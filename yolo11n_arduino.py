import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

class Config:
    def __init__(self):
        # Model parameters
        self.model_path = 'best_ncnn_model'

        # Basic detection parameter
        self.conf_threshold = 0.10  # Confidence threshold for YOLO
        
        # ROI parameters
        self.mask_path = 'Masked Region 2.jpg'
        self.mask_threshold = 127
        self.mask_max_value = 255
        
        # Serial communication parameters
        self.serial_port = '/dev/cu.usbmodem101'  # Change based on your system
        self.baud_rate = 9600

# Initialize configuration
config = Config()

# Initialize Serial Communication with Arduino
try:
    arduino = serial.Serial(port=config.serial_port, baudrate=config.baud_rate, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    print("Arduino connected successfully!")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None

def send_signal_to_arduino(signal):
    """Send command to Arduino over Serial."""
    if arduino:
        arduino.write(f"{signal}\n".encode())
        time.sleep(0.1) 

def check_box_in_roi(box_coords, mask):
    """Check if the detected bounding box is within the Region of Interest (ROI)"""
    x1, y1, x2, y2 = box_coords
    box_region = mask[y1:y2, x1:x2]
    
    if box_region.size == 0:
        return False

    overlap_percentage = np.count_nonzero(box_region) / box_region.size
    return overlap_percentage >= 0.1

def process_frame(frame, mask, model):
    """Process a single webcam frame with YOLO"""
    results = model(frame, conf=config.conf_threshold, verbose=False)
    annotated_frame = frame.copy()
    object_detected = False
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0])

            if check_box_in_roi((x1, y1, x2, y2), mask):
                object_detected = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                conf_text = f'Conf: {conf:.2f}'
                cv2.putText(annotated_frame, conf_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if object_detected:
        send_signal_to_arduino("0")  # Stop motor
        print("Object detected! Sending STOP signal to Arduino.")
    else:
        send_signal_to_arduino("1")  # Start motor
        print("No object detected. Sending START signal to Arduino.")
    
    return annotated_frame

def process_webcam():
    """Process webcam feed with YOLO and display real-time detection results"""
    print("\nLoading YOLO model...")
    model = YOLO(config.model_path, task='detect')
    print("Model loaded successfully!")

    print(f"Loading mask from {config.mask_path}")
    mask = cv2.imread(config.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Error: Mask image could not be loaded!")
        return
    
    print("Resizing mask to match webcam feed size...")
    webcam_width, webcam_height = 640, 480
    mask = cv2.resize(mask, (webcam_width, webcam_height))
    _, mask = cv2.threshold(mask, config.mask_threshold, config.mask_max_value, cv2.THRESH_BINARY)

    print("Mask processed successfully!")
    print("Starting webcam. Press 'q' to exit.")

    cap = cv2.VideoCapture(0)
    cap.set(3, webcam_width)
    cap.set(4, webcam_height)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (webcam_width, webcam_height))
        annotated_frame = process_frame(frame, mask, model)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display FPS on the top right
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(annotated_frame, fps_text, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.imshow("Webcam Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam processing stopped.")

if __name__ == "__main__":
    process_webcam()