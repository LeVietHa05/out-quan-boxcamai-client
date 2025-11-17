import cv2
import os
from datetime import datetime
import numpy as np
import config as config

def save_detection_image(frame, class_name, score, bbox, timestamp):
    """Save a cropped image of the detected object to disk"""
    
    # Generate filename with timestamp
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp_str}_{class_name}_{score:.2f}.jpg"
    filepath = os.path.join(config.IMAGES_DIR, filename)

    # Save the cropped image
    try:
        cv2.imwrite(filepath, frame)
        return filename
    except Exception as e:
        print(f"Error saving detection image: {e}")
        return None

def non_max_suppression(boxes, scores, threshold):
    """Apply Non-Maximum Suppression to filter overlapping bounding boxes"""
    # If no boxes, return empty array
    if len(boxes) == 0:
        return np.array([])

    # Convert to numpy arrays if not already
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)

    # Sort by scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep)
