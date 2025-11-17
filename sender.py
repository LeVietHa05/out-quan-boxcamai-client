import requests
import json
import threading
from multiprocessing import Queue, Event
import time
import config 
import os

# Global queue and thread for asynchronous sending
detection_queue = Queue()
send_thread = None
stop_send_thread = Event()

def send_detection_to_server(detection_data):
    """Add detection data to queue for asynchronous sending"""
    # Add client information to detection data
    
    detection_data['client_name'] = config.CLIENT_NAME 
    detection_data['client_latitude'] = config.CLIENT_LATITUDE 
    detection_data['client_longitude'] = config.CLIENT_LONGITUDE 

    detection_queue.put(detection_data)

def send_worker():
    """Background worker to send detections to server"""
    while not stop_send_thread.is_set():
        try:
            # Get detection data from queue with timeout
            detection_data = detection_queue.get(timeout=1.0)

            print(f"Sending {detection_data['class_name']} to server")

            # Prepare image file path
            image_path_full = os.path.join(config.IMAGES_DIR, detection_data['image_path'])
            if not os.path.exists(image_path_full):
                print(f"Image file not found: {image_path_full}")
                detection_queue.task_done()
                continue

            # Send to server as multipart
            try:
                with open(image_path_full, 'rb') as img_file:
                    files = {'image': (detection_data['image_path'], img_file, 'image/jpeg')}
                    data = {'json_data': json.dumps(detection_data)}
                    response = requests.post(
                        f'http://{config.SERVER_HOST}:{config.SERVER_PORT}/api/detections',
                        files=files,
                        data=data,
                        timeout=10
                    )
                if response.status_code in (200, 201):
                    print(f"Detection sent successfully: {detection_data['class_name']}")
                else:
                    print(f"Failed to send detection: HTTP {response.status_code}")
                    print(f"{response.json()}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending detection: {e}")

            # Mark task as done
            detection_queue.task_done()

        except:
            # Timeout or other exception, continue loop
            continue

def start_send_thread():
    """Start the background sending thread"""
    global send_thread
    if send_thread is None or not send_thread.is_alive():
        stop_send_thread.clear()
        send_thread = threading.Thread(target=send_worker, daemon=True)
        send_thread.start()
        print("Detection sending thread started")

def stop_send_thread_func():
    """Stop the background sending thread"""
    stop_send_thread.set()
    if send_thread and send_thread.is_alive():
        send_thread.join(timeout=5)
        print("Detection sending thread stopped")
