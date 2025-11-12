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
import client.config as config
from client.utils import save_detection_image, non_max_suppression
from client.sender import send_detection_to_server

# Global dict for tracking last detection time per class
last_detection_time = {}

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
def detection_process(q, stop_event, not_sent=False):
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

            # Preprocess frame for model input
            # Resize frame to model input size using config values to avoid symbolic shapes
            resized_frame = cv2.resize(frame, (config.INPUT_W_SIZE, config.INPUT_H_SIZE))

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Normalize pixel values to [0, 1]
            normalized_frame = rgb_frame.astype(np.float32) / 255.0

            # Transpose to channel-first format (NCHW)
            input_tensor = np.transpose(normalized_frame, (2, 0, 1))

            # Add batch dimension
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Run inference
            outputs = session.run(None, {input_name: input_tensor})[0] # [1, N, 85] - [batch, detections, 85 values per detection]

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
            if boxes:
                idxs = non_max_suppression(boxes, scores, IOU_THRESH)
                # Process detections that survived NMS
                if len(idxs) > 0:
                    for i in idxs.flatten():
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

                        # Get class name
                        class_name = config.CLASS_NAMES2[class_id]

                        # Check if this object should be tracked
                        if config.TRACKED_OBJECTS and class_name not in config.TRACKED_OBJECTS:
                            continue

                        # Check tracking: send only for new objects or reappearances after timeout
                        current_time = time.time()
                        should_send = (class_name not in last_detection_time or
                                       (current_time - last_detection_time[class_name]) > config.TRACK_TIMEOUT)

                        if should_send:
                            last_detection_time[class_name] = current_time
                            print(class_name)
                            # Create timestamp
                            timestamp = datetime.now()

                            # Draw bounding boxes and labels for detections that survived NMS
                            # Create label text with class name and confidence score
                            label = f"{class_name} {score:.2f}"
                            # Draw green bounding box around detected object
                            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                            # Draw label text above the bounding box
                            cv2.putText(frame, label, (int(x), int(y + 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            # Save detection image
                            print(f"Saving image for {class_name}")
                            image_filename = save_detection_image(frame, class_name, score, (x, y, w, h), timestamp)

                            if image_filename:
                                # Prepare detection data for server
                                detection_data = {
                                    'timestamp': timestamp.isoformat(),
                                    'class_name': class_name,
                                    'confidence': score,
                                    'image_path': os.path.basename(image_filename),
                                    'bbox_x': x,
                                    'bbox_y': y,
                                    'bbox_width': w,
                                    'bbox_height': h,
                                    'metadata': {
                                        'frame_width': frame.shape[1],
                                        'frame_height': frame.shape[0],
                                        'detection_id': f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{class_name}"
                                    }
                                }

                                # Queue detection for background sending (non-blocking) only if not disabled
                                if not not_sent:
                                    print(f"Sending detection for {class_name}")
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

            # No display window; run headless

        except Exception as e:
            print(f"Detection process error: {e}")
            continue
