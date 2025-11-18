import os

# Server Configuration
SERVER_HOST = 'boxcamai.cloud'
SERVER_PORT = 443
SERVER_URL = f"https://{SERVER_HOST}:{SERVER_PORT}/api/detections"

# Database Configuration
# DATABASE_URL = 'sqlite:///server/detections.db' # no need the db in client

# Image Storage Configuration
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'captured_images')# Client images
SERVER_IMAGES_DIR = os.path.join(os.path.dirname(__file__),  'captured_images')  # Server images
MAX_IMAGES_PER_DETECTION = 5  # Maximum images to keep per detection class

# Detection Configuration
DETECTION_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3
FRAME_SKIP = 1
TIME_BETWEEN_SEND = 2.0

# Objects to track (empty list means track all detected objects)
TRACKED_OBJECTS = ['person', 'car', 'truck', 'bus', 'motorbike']  # ['person', 'car', 'truck', 'bus', 'motorbike']

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FRAMERATE = 30

# Video File Configuration (set to None to use camera, or provide path to local video file)
VIDEO_FILE_PATH = None # e.g., 'path/to/video.mp4'
# VIDEO_FILE_PATH = './videotest/video-test-2.mp4' # e.g., 'path/to/video.mp4'

# RTSP Configuration
RTSP_USER = 'admin'
RTSP_PASS = 'quan2004'
RTSP_IP = None
RTSP_PORT = '554'
RTSP_URL = "rtsp://admin:quan2004@172.16.1.245:554/cam/realmonitor?channel=1&subtype=1"
#go lai rtsv vao day la oke
# Model Configuration
MODEL_PATH = 'yolov5s.onnx'
# MODEL_PATH = 'best.onnx'
INPUT_W_SIZE = 320
INPUT_H_SIZE = 320
INPUT_SIZE = 320

# Rate limiting configuration
DETECTION_SEND_DELAY = 1  # Delay in seconds between sending detections to server

# Tracking configuration
TRACK_TIMEOUT = 5  # Timeout in seconds for object reappearance to trigger send

# Class names for the model
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

CLASS_NAMES2 = ['ambulance', 'bicycle', 'bird', 'bus', 'camel', 'car', 'cow', 'deer', 'drone', 'dump truck', 'excavators', 'goat', 'horse', 'motorcycle', 'person', 'sheep', 'truck', 'wheel loader']
# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
# os.makedirs(SERVER_IMAGES_DIR, exist_ok=True) #no need server image folder in client folder

# Client Configuration (for identification when sending to server)
CLIENT_NAME = 'raspberry_pi_1'  # Unique name for this client
CLIENT_LATITUDE = None  # GPS latitude (optional)
CLIENT_LONGITUDE = None  # GPS longitude (optional)
