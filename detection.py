import cv2
import numpy as np
import onnxruntime as ort
import subprocess
from multiprocessing import Process, Queue, Event
import os
import json
from datetime import datetime
import time
import threading
import argparse
import config
from utils import save_detection_image, non_max_suppression
from sender import send_detection_to_server, start_send_thread, stop_send_thread_func

# Global for last send time
last_send_time = 0.0

# Define the path to the ONNX model file
onnx_path = config.MODEL_PATH

# Create ONNX Runtime inference session with CPU execution provider for running the model
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# Get model input details
input_details = session.get_inputs()[0]
input_name = input_details.name
input_shape = input_details.shape

# IOU threshold for Non-Maximum Suppression
IOU_THRESH = config.IOU_THRESHOLD

# Object detection process function that runs in a separate process for parallel execution


def detection_process(q, stop_event, not_sent=False, display=False, roi_x1=None, roi_y1=None, roi_x2=None, roi_y2=None):
    # Start the sender thread in this process if sending is enabled
    if not not_sent:
        start_send_thread()

    # Main detection loop - runs continuously until stopped
    frame_count = 0

    while True:
        if stop_event.is_set():
            break

        try:
            # Receive frame from queue
            frame = q.get(timeout=1.0)
            frame_count += 1

            # Skip frames for performance optimization
            if frame_count % config.FRAME_SKIP != 0:
                continue

            # Skip invalid frames
            if frame is None or frame.size == 0:
                continue

            frame_original = frame.copy()       # ảnh gốc để lưu và gửi

            # Apply ROI cropping if ROI is defined
            if roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None:
                # Convert ROI coordinates to integers
                roi_x1_int = int(roi_x1)
                roi_y1_int = int(roi_y1)
                roi_x2_int = int(roi_x2)
                roi_y2_int = int(roi_y2)

                # Ensure ROI coordinates are within frame bounds
                roi_x1_int = max(0, min(roi_x1_int, frame.shape[1] - 1))
                roi_y1_int = max(0, min(roi_y1_int, frame.shape[0] - 1))
                roi_x2_int = max(
                    roi_x1_int + 1, min(roi_x2_int, frame.shape[1]))
                roi_y2_int = max(
                    roi_y1_int + 1, min(roi_y2_int, frame.shape[0]))

                # Crop the frame to ROI
                frame = frame[roi_y1_int:roi_y2_int, roi_x1_int:roi_x2_int]

                # Skip if ROI is too small
                if frame.shape[0] < 10 or frame.shape[1] < 10:
                    continue

            # Preprocess frame for model input
            # Resize frame to model input size using config values to avoid symbolic shapes
            resized_frame = cv2.resize(
                frame, (config.INPUT_W_SIZE, config.INPUT_H_SIZE))

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Normalize pixel values to [0, 1]
            normalized_frame = rgb_frame.astype(np.float32) / 255.0

            # Transpose to channel-first format (NCHW)
            input_tensor = np.transpose(normalized_frame, (2, 0, 1))

            # Add batch dimension
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Run inference
            # [1, N, 85] - [batch, detections, 85 values per detection]
            outputs = session.run(None, {input_name: input_tensor})[0]

            # Process model outputs
            # Assuming YOLOv5 output format: [batch, num_boxes, 85] where 85 = 4 bbox + 1 obj + 80 classes
            # Remove the batch dimension from outputs for easier processing
            predictions = np.squeeze(outputs)

            # Ensure predictions is 2D [num_boxes, features]
            if predictions.ndim == 1:
                predictions = np.expand_dims(predictions, axis=0)

            # Filter predictions by confidence threshold
            boxes = []
            scores = []
            class_ids = []
            class_names = []  # for send to server
            confidences = []  # for send to server
            xs = []
            ys = []
            ws = []
            hs = []
            for pred in predictions:
                confidence = pred[4]  # Objectness score
                if confidence < config.DETECTION_THRESHOLD:
                    continue
                # Get class scores (offsets 5-84 in YOLO output format)
                class_scores = pred[5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]

                # Combine objectness and class score
                final_score = confidence * class_score

                if final_score < config.DETECTION_THRESHOLD:
                    continue

                # Extract bounding box coordinates (center x, center y, width, height)
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

                # Convert from normalized coordinates to pixel coordinates
                x1 = int((cx - w / 2) * frame.shape[1] / config.INPUT_W_SIZE)
                y1 = int((cy - h / 2) * frame.shape[0] / config.INPUT_H_SIZE)
                x2 = int((cx + w / 2) * frame.shape[1] / config.INPUT_W_SIZE)
                y2 = int((cy + h / 2) * frame.shape[0] / config.INPUT_H_SIZE)

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(final_score))
                class_ids.append(class_id)

            # Perform Non-Maximum Suppression if we have any valid detections
            if len(boxes) > 0:
                idxs = non_max_suppression(boxes, scores, IOU_THRESH)
                # Process detections that survived NMS
                if len(idxs) > 0:
                    for i in idxs.flatten():
                        # Get class name
                        class_name = config.CLASS_NAMES[class_id]
                        # class_name = config.CLASS_NAMES2[class_id]

                        # Check if this object should be tracked
                        if config.TRACKED_OBJECTS and class_name not in config.TRACKED_OBJECTS:
                            continue

                        # Extract bounding box coordinates
                        x, y, w, h = boxes[i]
                        class_id = class_ids[i]
                        score = scores[i]

                        # Validate bbox
                        if not all(isinstance(coord, (int, float)) for coord in [x, y, w, h]) or w <= 0 or h <= 0:
                            continue

                        # Clamp bbox to frame bounds
                        x = max(0, min(x, frame.shape[1] - 1))
                        y = max(0, min(y, frame.shape[0] - 1))
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)

                        if w <= 0 or h <= 0:
                            continue

                        # Draw bounding boxes and labels for all detections that survived NMS
                        label = f"{class_name} {score:.2f}"
                        cv2.rectangle(frame_original, (int(x + (roi_x1_int if roi_x1_int is not None else 0)), int(y + (roi_y1_int if roi_y1_int is not None else 0))),
                                      (int(x + (roi_x1_int if roi_x1_int is not None else 0) + w), int(y + (roi_y1_int if roi_y1_int is not None else 0) + h)), (0, 255, 0), 2)
                        cv2.putText(frame_original, label, (int(x + (roi_x1_int if roi_x1_int is not None else 0)), int(y + (roi_y1_int if roi_y1_int is not None else 0) + 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        xs.append(
                            x + (roi_x1_int if roi_x1_int is not None else 0))
                        ys.append(
                            y + (roi_y1_int if roi_y1_int is not None else 0))
                        ws.append(w)
                        hs.append(h)
                        class_names.append(class_name)
                        confidences.append(score)
                    
                     # Draw roi
                    if roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None:
                        cv2.rectangle(frame_original, (int(roi_x1_int), int(roi_y1_int)),
                        (int(roi_x2_int), int(roi_y2_int)), (0, 0, 255), 1)

                    # Send every 1 second if any detections are present
                    current_time = time.time()
                    global last_send_time
                    if (current_time - last_send_time) > config.TIME_BETWEEN_SEND:
                        if not class_names:
                            continue
                        last_send_time = current_time
                        print(f"Detections found: {class_names}")
                        # Create timestamp
                        timestamp = datetime.now()

                        # Save detection image (using first detection for image saving)
                        print(f"Saving image for detections")
                        image_filename = save_detection_image(frame_original, class_names[0] if class_names else "unknown", confidences[0] if confidences else 0, (
                            xs[0] if xs else 0, ys[0] if ys else 0, ws[0] if ws else 0, hs[0] if hs else 0), timestamp)

                        if image_filename:
                            # Prepare detection data for server
                            detection_data = {
                                'timestamp': timestamp.isoformat(),
                                'class_name': class_names,
                                'confidence': confidences,
                                'image_path': os.path.basename(image_filename),
                                'bbox_x': xs,
                                'bbox_y': ys,
                                'bbox_width': ws,
                                'bbox_height': hs,
                                'metadata': {
                                    'frame_width': frame.shape[1],
                                    'frame_height': frame.shape[0],
                                    'detection_id': f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
                                }
                            }

                            # Queue detection for background sending (non-blocking) only if not disabled
                            if not not_sent:
                                print(f"Sending detections to server")
                                send_detection_to_server(detection_data)

                        # Draw bounding boxes and labels for detections that survived NMS (only if not sending, for display but since headless, optional)
                        # But since we moved draw inside if should_send, here only if not should_send
                        # But since we running headless so no need for this
                        # if not should_send:
                        #     # Create label text with class name and confidence score
                        #     label = f"{class_name} {score:.2f}"
                        #     # Draw green bounding box around detected object
                        #     cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                        #     # Draw label text above the bounding box
                        #     cv2.putText(frame, label, (int(x), int(y - 5)),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display window if enabled
            if display:
                cv2.imshow('Object Detection', frame_original)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to exit
                    stop_event.set()
                    break

        except Exception as e:
            print(f"Detection process error: {e}")
            continue

    # Cleanup
    if not not_sent:
        stop_send_thread_func()

    # Close display window if it was opened
    if display:
        cv2.destroyAllWindows()
