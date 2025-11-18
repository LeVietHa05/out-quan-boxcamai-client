import cv2
import subprocess
import numpy as np
from multiprocessing import Process, Queue, Event
import argparse
import config
from detection import detection_process
from sender import start_send_thread, stop_send_thread_func
import requests


def video_capture_process(q, stop_event, source):
    if config.VIDEO_FILE_PATH:
        # Use OpenCV to read from local video file
        cap = cv2.VideoCapture(config.VIDEO_FILE_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file {config.VIDEO_FILE_PATH}")
            return
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("End of video file reached.")
                    break
                if frame is not None and not q.full():
                    q.put(frame)
        finally:
            cap.release()
    else:
        if source == 'rtsp':
            # Use OpenCV to read from RTSP stream
            rtspLink = f"rtsp://{config.RTSP_USER}:{config.RTSP_PASS}@{ip_address if not config.RTSP_IP else config.RTSP_IP}:{config.RTSP_PORT}/cam/realmonitor?channel=1&subtype=1"
            cap = cv2.VideoCapture(rtspLink)
            if not cap.isOpened():
                print(f"Error: Could not open RTSP stream {config.RTSP_URL}")
                return
            try:
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        print("RTSP stream ended.")
                        break
                    if frame is not None and not q.full():
                        q.put(frame)
            finally:
                cap.release()
        elif source == 'webcam':
            # Use OpenCV to read from local webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
            try:
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        print("Webcam capture ended.")
                        break
                    if frame is not None and not q.full():
                        q.put(frame)
            finally:
                cap.release()
        else:  # rpicam
            # Use rpicam-vid subprocess for camera
            cmd = [
                'rpicam-vid',
                '--width', str(config.CAMERA_WIDTH),
                '--height', str(config.CAMERA_HEIGHT),
                '--framerate', str(config.CAMERA_FRAMERATE),
                '--codec', 'mjpeg',
                '--inline',
                '--timeout', '0',
                '-o', '-',
                '--nopreview'
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
            buffer = b""
            try:
                while not stop_event.is_set():
                    data = proc.stdout.read(1024)
                    if not data:
                        break
                    buffer += data
                    while b'\xff\xd9' in buffer:
                        split_idx = buffer.index(b'\xff\xd9') + 2
                        jpg_data = buffer[:split_idx]
                        buffer = buffer[split_idx:]
                        jpg = np.frombuffer(jpg_data, dtype=np.uint8)
                        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                        if frame is not None and not q.full():
                            q.put(frame)
            finally:
                proc.terminate()
                proc.wait()


def get_info():
    try:
        response = requests.get(
            f'http://{config.SERVER_HOST}:{config.SERVER_PORT}/api/clients/by-name/{config.CLIENT_NAME}',
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get client info: HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting client info: {e}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Object Detection Client')
    parser.add_argument('--not-sent', action='store_true',
                        help='Run without sending detections to server')
    parser.add_argument('--video', type=str,
                        help='Path to video file for testing')
    parser.add_argument('--rtsp', action='store_true',
                        help='Use RTSP stream for video')
    parser.add_argument('--rpicam', action='store_true',
                        help='Use rpicam for video')
    parser.add_argument('--webcam', action='store_true',
                        help='Use local webcam for video')
    parser.add_argument('--display', action='store_true',
                        help='Display detection results in a window')
    args = parser.parse_args()

    not_sent = args.not_sent

    # Get client info from server
    client_info = get_info()
    if client_info:
        print(f"Client info retrieved: {client_info}")
        is_detect_enabled = client_info.get('is_detect_enabled', True)
        roi_x1 = client_info.get('roi_x1')
        roi_y1 = client_info.get('roi_y1')
        roi_x2 = client_info.get('roi_x2')
        roi_y2 = client_info.get('roi_y2')
        global ip_address
        ip_address = client_info.get('ip_address')

        # Override not_sent based on server setting
        if not is_detect_enabled:
            not_sent = True
            print("Detection disabled by server")
    else:
        print("Could not retrieve client info, using default settings")
        is_detect_enabled = True
        roi_x1 = roi_y1 = roi_x2 = roi_y2 = None

    # Determine video source
    if args.webcam:
        source = 'webcam'
    elif args.rtsp:
        source = 'rtsp'
    elif args.rpicam:
        source = 'rpicam'
    else:
        source = 'rpicam'  # default

    # Note: Sender thread will be started in the detection process
    # if not not_sent:
    #     start_send_thread()

    # Create queue for inter-process communication
    frame_queue = Queue(maxsize=10)
    stop_event = Event()

    # Start detection process
    detection_proc = Process(target=detection_process, args=(
        frame_queue, stop_event, not_sent, args.display, roi_x1, roi_y1, roi_x2, roi_y2))
    detection_proc.start()

    # Override config if video file specified via args
    if args.video:
        config.VIDEO_FILE_PATH = args.video

    # Start video capture process
    capture_proc = Process(target=video_capture_process,
                           args=(frame_queue, stop_event, source))
    capture_proc.start()

    try:
        print("Starting object detection...")
        # Wait for processes
        capture_proc.join()
        detection_proc.join()

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Cleanup
        stop_event.set()
        capture_proc.join(timeout=5)
        detection_proc.join(timeout=5)
        # Note: Sender thread cleanup is now handled in detection process
        # if not not_sent:
        #     stop_send_thread_func()
        print("Detection client stopped")


if __name__ == '__main__':
    main()
