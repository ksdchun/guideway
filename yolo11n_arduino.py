
# ===============================================================================
# YOLO Object Detection with Arduino Control System
# ===============================================================================
# This application combines computer vision with hardware control to detect
# objects within specified regions and control an Arduino-based motor system.
# ===============================================================================

import os 
import datetime

# Disable hardware transforms in OpenCV to fix slow camera initialization issues
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Core dependencies
import cv2                # OpenCV for computer vision and camera access
import numpy as np        # NumPy for efficient array operations
import time               # Time utilities for delays and timestamps
import serial             # Serial communication with Arduino
from ultralytics import YOLO  # YOLO object detection model
import json               # JSON parsing for configuration

# GUI components
import tkinter as tk
from tkinter import messagebox, ttk
import sv_ttk             # Sun Valley theme for modern UI appearance
from PIL import Image, ImageTk, ImageDraw  # Image processing for overlays

class Config:
    """Configuration management system with persistent storage.
    
    This class handles loading, storing, and providing access to all application settings.
    It uses a JSON file for persistence and provides sensible defaults for all parameters.
    """
    def __init__(self):
        # Define configuration file path relative to script location
        self.config_file = os.path.join(os.path.dirname(__file__), 'gui_config.json')
        cfg = {}
        
        # Attempt to load existing configuration
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load GUI config: {e}")
        
        # Initialize all parameters with defaults or loaded values
        # YOLO model settings
        self.model_path = cfg.get('model_path', os.path.join(os.path.dirname(__file__), 'weights', 'YOLO11n.pt'))
        self.conf_threshold = cfg.get('conf_threshold', 0.45)  # Confidence threshold for detections
        
        # Region of interest settings
        self.region_json_path = cfg.get('region_json_path', os.path.join(os.path.dirname(__file__), 'regions.json'))
        self.mask_threshold = cfg.get('mask_threshold', 127)  # Binary threshold for mask creation
        self.mask_max_value = cfg.get('mask_max_value', 255)  # Maximum value for binary mask
        self.overlap_threshold = cfg.get('overlap_threshold', 0.1)  # Minimum overlap to consider detection in ROI
        
        # Arduino communication settings
        self.serial_port = cfg.get('serial_port', '/dev/cu.usbmodem101')
        self.baud_rate = cfg.get('baud_rate', 9600)
        self.motor_speed = cfg.get('motor_speed', 255)  # Default to full speed (0-255)
        
        # Recording and screenshot settings
        self.recordings_dir = cfg.get('recordings_dir', os.path.join(os.path.dirname(__file__), 'recordings'))
        self.screenshots_dir = cfg.get('screenshots_dir', os.path.join(os.path.dirname(__file__), 'screenshots'))
        self.auto_screenshot = cfg.get('auto_screenshot', True)  # Take screenshots automatically on detection
        
        # Ensure required directories exist
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.recordings_dir):
            try:
                os.makedirs(self.recordings_dir)
                print(f"[Info] Created recordings directory: {self.recordings_dir}")
            except Exception as e:
                print(f"[Warning] Failed to create recordings directory: {e}")
                
        # Create screenshots directory if it doesn't exist
        if not os.path.exists(self.screenshots_dir):
            try:
                os.makedirs(self.screenshots_dir)
                print(f"[Info] Created screenshots directory: {self.screenshots_dir}")
            except Exception as e:
                print(f"[Warning] Failed to create screenshots directory: {e}")

    def save(self):
        """Save current configuration to JSON file.
        
        This method persists all current settings to the configuration file,
        allowing them to be restored on next application launch.
        """
        # Create a dictionary with all current configuration values
        cfg = {
            # YOLO model settings
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            
            # Region of interest settings
            'region_json_path': self.region_json_path,
            'mask_threshold': self.mask_threshold,
            'mask_max_value': self.mask_max_value,
            'overlap_threshold': self.overlap_threshold,
            
            # Arduino communication settings
            'serial_port': self.serial_port,
            'baud_rate': self.baud_rate,
            'motor_speed': self.motor_speed,
            
            # Recording and screenshot settings
            'recordings_dir': self.recordings_dir,
            'screenshots_dir': self.screenshots_dir,
            'auto_screenshot': self.auto_screenshot
        }
        
        # Write configuration to file with error handling
        try:
            with open(self.config_file, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"[Info] Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"[Error] Failed to save GUI config: {e}")




# ===============================================================================
# Application Initialization
# ===============================================================================

# Initialize configuration system
config = Config()
print(f"[Info] Configuration loaded from {config.config_file}")

# Initialize Serial Communication with Arduino
try:
    arduino = serial.Serial(port=config.serial_port, baudrate=config.baud_rate, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize (required for reliable communication)
    print(f"[Success] Arduino connected on port {config.serial_port} at {config.baud_rate} baud")
except Exception as e:
    print(f"[Error] Failed to connect to Arduino: {e}")
    print("[Warning] Running in degraded mode without hardware control")
    arduino = None  # Set to None to allow application to run without Arduino

# ===============================================================================
# Computer Vision Functions
# ===============================================================================

def check_box_in_roi(box_coords, mask):
    """Check if the detected bounding box is within the Region of Interest (ROI).
    
    Args:
        box_coords (tuple): Coordinates of detection box (x1, y1, x2, y2)
        mask (numpy.ndarray): Binary mask image where white areas (255) represent ROIs
        
    Returns:
        bool: True if the box overlaps with the ROI by at least the threshold percentage
    """
    # Extract box coordinates
    x1, y1, x2, y2 = box_coords
    
    # Extract the portion of the mask that corresponds to the bounding box
    box_region = mask[y1:y2, x1:x2]
    
    # If box is outside the mask or has zero area, it's not in the ROI
    if box_region.size == 0:
        return False

    # Calculate what percentage of the box overlaps with the masked region
    # by counting non-zero pixels and dividing by total pixels in the box
    overlap_percentage = np.count_nonzero(box_region) / box_region.size
    
    # Consider it a match if overlap exceeds the configured threshold
    return overlap_percentage >= config.overlap_threshold

# Define inference resolution - smaller than display resolution for speed optimization
# Higher resolution = better detection accuracy but slower performance
INFER_WIDTH, INFER_HEIGHT = 640, 360  # 16:9 aspect ratio at lower resolution

def process_frame(frame, mask, model):
    """Process a single webcam frame with YOLO object detection.
    
    This function handles the core computer vision pipeline:
    1. Resize frame for faster inference
    2. Run YOLO detection on the resized frame
    3. Scale detection coordinates back to original frame size
    4. Check if detections are within the ROI
    5. Annotate the frame with detection boxes and confidence values
    
    Args:
        frame (numpy.ndarray): Original webcam frame
        mask (numpy.ndarray): Binary mask defining regions of interest
        model (YOLO): Loaded YOLO model instance
        
    Returns:
        tuple: (annotated_frame, object_detected)
            - annotated_frame: Frame with detection boxes drawn
            - object_detected: Boolean indicating if objects were detected in ROI
    """
    # Get original dimensions and downscale for faster inference
    orig_h, orig_w = frame.shape[:2]
    small = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
    
    # Calculate scaling factors to map detection coordinates back to original frame
    scale_x = orig_w / INFER_WIDTH
    scale_y = orig_h / INFER_HEIGHT
    
    # Run YOLO inference on the smaller frame
    results = model(small, conf=config.conf_threshold, verbose=False)
    
    # Create a copy of the original frame for annotation
    annotated_frame = frame.copy()
    object_detected = False
    
    # Process each detection
    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates (in small frame space)
            x1_s, y1_s, x2_s, y2_s = box.xyxy[0].cpu().numpy()
            
            # Scale coordinates back to original frame space
            x1, y1, x2, y2 = (int(x1_s * scale_x), int(y1_s * scale_y), 
                              int(x2_s * scale_x), int(y2_s * scale_y))
            
            # Get detection confidence
            conf = float(box.conf[0])

            # Check if the detection is within our region of interest
            if check_box_in_roi((x1, y1, x2, y2), mask):
                object_detected = True
                
                # Draw green rectangle around the detected object
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
                # Add confidence text above the box
                conf_text = f'Conf: {conf:.2f}'
                cv2.putText(annotated_frame, conf_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Return the annotated frame and detection flag
    return annotated_frame, object_detected

def process_webcam_gui():
    """Main GUI function with mask selector and 720p live preview.
    
    This function initializes and runs the complete application GUI, including:
    - Loading region definitions from JSON
    - Setting up the control panel with all parameters
    - Initializing the webcam and video display
    - Creating the frame processing loop
    - Handling all user interactions and events
    """
    # ===============================================================================
    # Load region definitions from JSON file
    # ===============================================================================
    try:
        with open(config.region_json_path, 'r') as f:
            data = json.load(f)
            print(f"[Info] Loaded region definitions from {config.region_json_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load regions.json: {e}")
        print(f"[Error] Failed to load region definitions: {e}")
        return
        
    # Extract available mask models from the JSON data
    models = list(data.keys())
    if not models:
        messagebox.showerror("Error", "No mask models found in regions.json")
        print("[Error] No mask models found in regions.json")
        return
    # ===============================================================================
    # Initialize main application window
    # ===============================================================================
    root = tk.Tk()
    root.title("Guideway")
    # Start maximized for optimal visibility
    root.state('zoomed')
    # Apply Sun Valley theme for modern appearance
    sv_ttk.set_theme("light")
    
    # ===============================================================================
    # Create left control panel
    # ===============================================================================
    # Fixed-width control panel to hold all UI controls
    ctrl = tk.Frame(root, width=188)  # Width accommodates all controls and labels
    ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=3)
    ctrl.pack_propagate(False)  # Prevent panel from resizing based on contents

    # ===============================================================================
    # Mask Settings Section
    # ===============================================================================
    model_frame = ttk.LabelFrame(ctrl, text="Mask Settings")
    model_frame.pack(fill=tk.X, padx=5, pady=(10,5))
    
    # Mask model selection dropdown
    ttk.Label(model_frame, text="Mask Model:").pack(padx=5, pady=(5,0))
    mask_var = tk.StringVar(value=models[0])  # Default to first model
    combo = ttk.Combobox(model_frame, textvariable=mask_var, values=models, state="readonly")
    combo.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Horizontal frame for visualization options
    mask_viz_frame = tk.Frame(model_frame)
    mask_viz_frame.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Visualization control variables
    overlay_var = tk.BooleanVar(value=True)  # Show region overlay by default
    detect_preview_var = tk.BooleanVar(value=False)  # Don't show detection preview by default
    
    # Visualization checkboxes
    ttk.Checkbutton(mask_viz_frame, text="Overlay", variable=overlay_var, 
                   style="Small.TCheckbutton").pack(side=tk.LEFT, padx=(0,2))
    ttk.Checkbutton(mask_viz_frame, text="Preview", variable=detect_preview_var, 
                   style="Small.TCheckbutton").pack(side=tk.LEFT, padx=5)

    # ===============================================================================
    # Detection Parameters Section
    # ===============================================================================
    detection_frame = ttk.LabelFrame(ctrl, text="Detection Parameters")
    detection_frame.pack(fill=tk.X, padx=5, pady=(5,5))
    
    # Confidence threshold slider (0.0-1.0)
    threshold_var = tk.DoubleVar(value=config.conf_threshold)
    ttk.Label(detection_frame, text="Confidence:").pack(padx=5, pady=(5,0))
    tk.Scale(detection_frame, variable=threshold_var, from_=0.0, to=1.0, 
             resolution=0.01, orient='horizontal').pack(fill=tk.X, padx=5, pady=(0,5))
    # Auto-save config when threshold changes
    threshold_var.trace_add('write', lambda *args: (setattr(config, 'conf_threshold', threshold_var.get()), 
                                                  config.save()))
    
    # Overlap threshold slider (0.0-1.0)
    overlap_var = tk.DoubleVar(value=config.overlap_threshold)
    ttk.Label(detection_frame, text="Overlap:").pack(padx=5, pady=(0,0))
    tk.Scale(detection_frame, variable=overlap_var, from_=0.0, to=1.0, 
             resolution=0.01, orient='horizontal').pack(fill=tk.X, padx=5, pady=(0,5))
    # Auto-save config when overlap threshold changes
    overlap_var.trace_add('write', lambda *args: (setattr(config, 'overlap_threshold', overlap_var.get()), 
                                                config.save()))
    
    # ===============================================================================
    # Motor Control Section
    # ===============================================================================
    motor_frame = ttk.LabelFrame(ctrl, text="Motor Control")
    motor_frame.pack(fill=tk.X, padx=5, pady=(5,5))
    
    # Motor speed slider (0-255 for Arduino PWM)
    speed_var = tk.IntVar(value=config.motor_speed)
    ttk.Label(motor_frame, text="Motor Speed:").pack(padx=5, pady=(5,0))
    speed_scale = tk.Scale(motor_frame, variable=speed_var, from_=0, to=255, 
                          resolution=1, orient='horizontal')
    speed_scale.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Update config when speed value changes
    speed_var.trace_add('write', lambda *args: (setattr(config, 'motor_speed', speed_var.get()), 
                                              config.save()))
    
    # Apply speed to running motor when slider is released
    # This prevents sending too many commands while dragging
    speed_scale.bind("<ButtonRelease-1>", lambda event: apply_speed())

    # ===============================================================================
    # Run Mode Section
    # ===============================================================================
    run_frame = ttk.LabelFrame(ctrl, text="Run Mode")
    run_frame.pack(fill=tk.X, padx=5, pady=(5,10))
    
    # Status indicator label
    status_label = ttk.Label(run_frame, text="Status: Stopped")
    status_label.pack(padx=5, pady=(5,5))
    
    # Start/Stop control buttons
    ttk.Button(run_frame, text="Start", command=lambda: start_auto()).pack(fill=tk.X, padx=5, pady=(2,2))
    ttk.Button(run_frame, text="Stop", command=lambda: stop_auto()).pack(fill=tk.X, padx=5, pady=(2,5))
    
    # Horizontal frame for auto-resume checkbox
    check_frame = tk.Frame(run_frame)
    check_frame.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Auto-resume control - when enabled, system will automatically restart
    # after an object is no longer detected
    auto_resume_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(check_frame, text="Auto Resume", variable=auto_resume_var, 
                   style="Small.TCheckbutton").pack(side=tk.LEFT, padx=(0,1))
    
    # ===============================================================================
    # Recording & Screenshots Section
    # ===============================================================================
    recording_frame = ttk.LabelFrame(ctrl, text="Recording & Screenshots")
    recording_frame.pack(fill=tk.X, padx=5, pady=(5,10))
    
    # Recording status indicator
    recording_status = ttk.Label(recording_frame, text="Status: Not Recording")
    recording_status.pack(padx=5, pady=(5,5))
    
    # Recording control buttons
    record_btn = ttk.Button(recording_frame, text="Start Recording", 
                           command=lambda: start_recording())
    record_btn.pack(fill=tk.X, padx=5, pady=(2,2))
    
    stop_record_btn = ttk.Button(recording_frame, text="Stop Recording", 
                               command=lambda: stop_recording(), state=tk.DISABLED)
    stop_record_btn.pack(fill=tk.X, padx=5, pady=(2,5))
    
    # Horizontal frame for auto-screenshot checkbox
    screenshot_check_frame = tk.Frame(recording_frame)
    screenshot_check_frame.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Auto-screenshot control - when enabled, system will automatically take
    # screenshots when objects are detected
    auto_screenshot_var = tk.BooleanVar(value=config.auto_screenshot)
    ttk.Checkbutton(screenshot_check_frame, text="Auto Screenshot", 
                   variable=auto_screenshot_var, 
                   style="Small.TCheckbutton").pack(side=tk.LEFT, padx=0)
    auto_screenshot_var.trace_add('write', lambda *args: setattr(config, 
                                                              'auto_screenshot', 
                                                              auto_screenshot_var.get()))
    
    # ===============================================================================
    # Custom Styles
    # ===============================================================================
    # Create custom styles for UI elements to optimize space and appearance
    style = ttk.Style()
    style.configure("Small.TButton", font=("TkDefaultFont", 13))  # Slightly larger buttons
    style.configure("Small.TCheckbutton", font=("TkDefaultFont", 8))  # Smaller checkbuttons

    # ===============================================================================
    # Control State Variables
    # ===============================================================================
    # Core control state for detection and motor operation
    running = False        # Whether detection and motor control is active
    last_signal = None     # Last command sent to Arduino (1=run, 0=stop)
    current_speed = config.motor_speed  # Current motor speed setting
    
    # Video recording state
    recording = False      # Whether recording is currently active
    video_writer = None    # OpenCV VideoWriter object when recording
    
    # Frame references for processing and display
    # Defined at outer scope to be accessible across all functions
    current_frame = None   # Most recent frame from camera
    annotated_frame = None # Frame with detection boxes drawn
    # ===============================================================================
    # Arduino Control Functions
    # ===============================================================================
    def send_and_set(sig, speed=None):
        """Send command to Arduino and update internal state.
        
        This function handles all communication with the Arduino using a simple
        text-based protocol in the format "signal:speed\n".
        
        Args:
            sig (str): Signal to send - '1' for run, '0' for stop
            speed (int, optional): Motor speed (0-255). If None, uses current_speed.
        """
        nonlocal last_signal, current_speed
        
        # Use provided speed or current speed if none specified
        if speed is None:
            speed = current_speed
        else:
            # Update current speed if explicitly provided
            current_speed = speed
            
        print(f"[GUI] send_and_set called with: {sig}, speed: {speed}")
        
        # Send command to Arduino if connected
        if arduino:
            # Format command as sig:speed (e.g., "1:200" for running at speed 200)
            command = f"{sig}:{speed}\n"
            arduino.write(command.encode())  # Convert to bytes and send
            time.sleep(0.1)  # Short delay to ensure command is processed
            print(f"[Arduino] Sent {command.strip()}")
        else:
            # Graceful handling when Arduino is not connected
            print(f"[Warning] Arduino not connected. Can't send {sig}:{speed}")
            
        # Update UI to reflect current state
        status_label.config(text=f"Status: {'Running' if sig=='1' else 'Stopped'}")
        
        # Store last signal for state tracking
        last_signal = sig
    def start_auto():
        """Start automatic detection and motor operation.
        
        This function is called when the Start button is pressed.
        It activates detection processing and sends the run command to Arduino.
        """
        print("[GUI] Start pressed")
        nonlocal running
        running = True  # Enable detection processing
        send_and_set('1', speed_var.get())  # Send run command with current speed setting
    def stop_auto():
        """Stop automatic detection and motor operation.
        
        This function is called when the Stop button is pressed.
        It disables detection processing, turns off auto-resume,
        and sends the stop command to Arduino.
        """
        print("[GUI] Stop pressed")
        nonlocal running
        running = False  # Disable detection processing
        auto_resume_var.set(False)  # Disable auto-resume to prevent immediate restart
        send_and_set('0', 0)  # Always stop with speed 0 for safety
    def pause_auto():
        """Pause operation when object is detected.
        
        Unlike stop_auto(), this function does not disable auto-resume,
        allowing the system to automatically restart when the object
        is no longer detected (if auto-resume is enabled).
        
        Also triggers automatic screenshot if that option is enabled.
        """
        print("[GUI] Paused on detection")
        nonlocal running, annotated_frame
        running = False  # Pause detection processing
        send_and_set('0', 0)  # Always stop with speed 0 for safety
        
        # Take a screenshot when object is detected and system stops
        # This provides visual documentation of what triggered the stop
        if auto_screenshot_var.get() and annotated_frame is not None:
            print("[Screenshot] Taking screenshot with detection")
            take_screenshot(annotated_frame)
        
    # ===============================================================================
    # Recording Functions
    # ===============================================================================
    def start_recording():
        """Start recording video to an MP4 file.
        
        Creates a timestamped MP4 file in the recordings directory and
        initializes an OpenCV VideoWriter to capture frames. Updates UI
        to reflect recording state.
        """
        nonlocal recording, video_writer
        
        # Guard against multiple recording starts
        if recording:
            return
            
        # Generate timestamp for filename (YYYYMMDD_HHMMSS format)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(config.recordings_dir, f"recording_{timestamp}.mp4")
        
        # Create VideoWriter object with MP4 codec
        # mp4v = MPEG-4 codec, 20.0 = framerate, (width, height) = frame size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (webcam_width, webcam_height))
        
        # Verify VideoWriter was initialized successfully
        if video_writer.isOpened():
            print(f"[Recording] Started: {video_path}")
            recording = True  # Set recording state
            
            # Update UI elements
            recording_status.config(text=f"Status: Recording")
            record_btn.config(state=tk.DISABLED)  # Disable start button
            stop_record_btn.config(state=tk.NORMAL)  # Enable stop button
        else:
            # Handle initialization failure
            messagebox.showerror("Error", "Failed to create video writer.")
            video_writer = None
    
    def stop_recording():
        """Stop video recording and release resources.
        
        Properly closes the VideoWriter, resets recording state,
        and updates UI elements.
        """
        nonlocal recording, video_writer
        
        # Guard against stopping when not recording
        if not recording or video_writer is None:
            return
        
        # Release VideoWriter resources
        video_writer.release()  # Properly close the video file
        video_writer = None  # Clear reference for garbage collection
        recording = False  # Reset recording state
        
        print("[Recording] Stopped")
        
        # Update UI elements
        recording_status.config(text="Status: Not Recording")
        record_btn.config(state=tk.NORMAL)  # Re-enable start button
        stop_record_btn.config(state=tk.DISABLED)  # Disable stop button
    
    # Location directory selection functions removed as UI buttons were removed
            
    def take_screenshot(frame_to_save):
        """Take a screenshot and save it to the screenshots directory.
        
        Creates a timestamped JPG file with optional region overlay.
        The overlay shows the detection regions as semi-transparent red polygons.
        
        Args:
            frame_to_save (numpy.ndarray): The frame to save as a screenshot
        """
        # Generate timestamp for filename (YYYYMMDD_HHMMSS format)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(config.screenshots_dir, f"screenshot_{timestamp}.jpg")
        
        # Apply overlay to screenshot if overlay is enabled
        if overlay_var.get():
            # Convert OpenCV BGR frame to PIL RGBA Image for transparency support
            pil_frame = Image.fromarray(cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)).convert("RGBA")
            
            # Create a transparent overlay image of the same size
            overlay_img = Image.new("RGBA", (webcam_width, webcam_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_img)
            
            # Draw the polygon overlay for each region
            regions = data.get(mask_var.get(), {}).get("regions", {})
            for pts in regions.values():
                # Convert region points to coordinate pairs
                coords = [(x, y) for x, y in pts]
                # Draw filled polygon with semi-transparent red (RGBA: 255,0,0,128)
                draw.polygon(coords, fill=(255, 0, 0, 128))
            
            # Composite the original image with the transparent overlay
            composite = Image.alpha_composite(pil_frame, overlay_img)
            
            # Convert back to OpenCV format (PIL RGBA → PIL RGB → NumPy array → OpenCV BGR)
            frame_with_overlay = cv2.cvtColor(np.array(composite.convert('RGB')), cv2.COLOR_RGB2BGR)
            
            # Save the composited image
            cv2.imwrite(screenshot_path, frame_with_overlay)
        else:
            # Save without overlay - direct OpenCV write
            cv2.imwrite(screenshot_path, frame_to_save)
            
        print(f"[Screenshot] Saved: {screenshot_path}")

    # ===============================================================================
    # Video Display and Camera Setup
    # ===============================================================================
    # Define display resolution - 720p HD (16:9 aspect ratio)
    webcam_width, webcam_height = 1280, 720
    
    # Create canvas for video display
    canvas = tk.Canvas(root, width=webcam_width, height=webcam_height)
    canvas.pack(side=tk.RIGHT)
    
    # Create a single image item that will be reused for each frame
    # This is more efficient than recreating image objects for each frame
    img_item = canvas.create_image(0, 0, anchor='nw', image=None)
    
    # Initialize webcam - index 1 typically refers to the first external camera
    # (index 0 is usually the built-in webcam on laptops)
    cap = cv2.VideoCapture(1)
    
    # Set camera resolution to match display resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam.")
        root.destroy()  # Close application if camera can't be accessed
        return
    # ===============================================================================
    # Region Mask Generation
    # ===============================================================================
    def build_mask(model_name):
        """Build a binary mask image from region definitions.
        
        Creates a mask where white areas (255) represent regions of interest
        and black areas (0) represent ignored regions.
        
        Args:
            model_name (str): Name of the model/configuration to use from regions.json
            
        Returns:
            numpy.ndarray: Binary mask image matching webcam resolution
        """
        # Get region definitions for the selected model
        regions = data.get(model_name, {}).get("regions", {})
        
        # Create empty mask matching webcam resolution
        m = np.zeros((webcam_height, webcam_width), dtype=np.uint8)
        
        # Fill each polygon region with the maximum mask value
        for pts in regions.values():
            poly = np.array(pts, dtype=np.int32)  # Convert points to numpy array
            cv2.fillPoly(m, [poly], config.mask_max_value)  # Fill polygon
            
        # Apply threshold to create binary mask
        # This ensures the mask only contains values 0 and max_value
        _, m = cv2.threshold(m, config.mask_threshold, config.mask_max_value, cv2.THRESH_BINARY)
        return m
    
    # Create initial mask from selected model
    # Use dictionary to allow updating by reference in callback functions
    mask_dict = {'mask': build_mask(mask_var.get())}
    
    # Callback function to rebuild mask when model selection changes
    def on_model_change(*args):
        mask_dict['mask'] = build_mask(mask_var.get())
        print(f"[Info] Mask updated to model: {mask_var.get()}")
        
    # Register callback to track model selection changes
    mask_var.trace_add('write', on_model_change)
    # ===============================================================================
    # YOLO Model Initialization and Frame Processing Loop
    # ===============================================================================
    # Load YOLO model from configured path
    print(f"[Info] Loading YOLO model from {config.model_path}")
    model = YOLO(config.model_path)
    
    def update_frame():
        """Main processing loop that runs at approximately 33 FPS.
        
        This function handles:
        1. Capturing frames from the webcam
        2. Running object detection when appropriate
        3. Implementing auto-stop and auto-resume logic
        4. Updating the display with appropriate visualization
        5. Recording video if active
        6. Scheduling the next frame update
        """
        nonlocal running, last_signal, current_frame, annotated_frame
        
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            # Handle camera read failure
            print("[Warning] Failed to read frame from camera")
            # Schedule next attempt and return early
            root.after(30, update_frame)
            return
        
        # Resize the current frame to ensure consistent dimensions
        current_frame = cv2.resize(frame, (webcam_width, webcam_height))
        
        # Determine if we should run inference based on current state:
        # - When actively running (normal operation)
        # - When preview is enabled (to see detections without motor control)
        # - When auto-resume is enabled (to detect when objects leave)
        do_infer = running or detect_preview_var.get() or auto_resume_var.get()
        
        if do_infer:
            # Process frame with YOLO detection
            annotated_frame, detected = process_frame(current_frame, mask_dict['mask'], model)
        else:
            # Skip detection to save processing power
            detected = False
            # Still create a copy for consistent behavior with screenshots
            annotated_frame = current_frame.copy()

        # -----------------------------------------------------------------------
        # Auto-stop logic: Stop on detection if currently running
        # -----------------------------------------------------------------------
        if running and detected and last_signal != '0':
            pause_auto()  # Pause operation when object detected

        # -----------------------------------------------------------------------
        # Auto-resume logic: Restart when detection stops (if enabled)
        # -----------------------------------------------------------------------
        if auto_resume_var.get() and not detected and last_signal == '0':
            start_auto()  # Resume operation when object no longer detected

        # -----------------------------------------------------------------------
        # Frame Selection and Visualization
        # -----------------------------------------------------------------------
        # Choose which frame to display based on current mode:
        # - Annotated frame with detection boxes when running or preview enabled
        # - Raw camera frame otherwise
        if running or detect_preview_var.get():
            frame_out = annotated_frame  # Show frame with detection boxes
        else:
            frame_out = frame  # Show raw camera frame
        
        # Apply semi-transparent region overlay if enabled
        has_overlay = overlay_var.get()
        if has_overlay:
            # Convert OpenCV BGR frame to PIL RGBA Image for transparency support
            pil_frame = Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)).convert("RGBA")
            
            # Create a transparent overlay image of the same size
            overlay_img = Image.new("RGBA", (webcam_width, webcam_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_img)
            
            # Get region definitions for the selected model
            regions = data.get(mask_var.get(), {}).get("regions", {})
            
            # Draw each region as a semi-transparent red polygon
            for pts in regions.values():
                coords = [(x, y) for x, y in pts]
                draw.polygon(coords, fill=(255, 0, 0, 128))  # Semi-transparent red
                
            # Composite the original image with the transparent overlay
            composite = Image.alpha_composite(pil_frame, overlay_img)
            
            # Convert to Tkinter-compatible PhotoImage
            imgtk = ImageTk.PhotoImage(composite)
            
            # Also prepare a version for video recording (PIL → NumPy → OpenCV)
            display_frame = np.array(composite.convert('RGB'))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else:
            # Without overlay - simpler conversion path
            pil_frame = Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(pil_frame)
            display_frame = frame_out  # Use selected frame directly
            
        # -----------------------------------------------------------------------
        # Video Recording
        # -----------------------------------------------------------------------
        if recording and video_writer is not None:
            # Record frame with consistent appearance as the display
            if has_overlay:
                # Use the already-prepared overlay version
                video_writer.write(display_frame)
            else:
                # Without overlay, choose appropriate frame based on mode
                if running or detect_preview_var.get():
                    # Record frame with detection boxes when in detection mode
                    video_writer.write(annotated_frame)
                else:
                    # Record raw camera frame otherwise
                    video_writer.write(frame)
        
        # -----------------------------------------------------------------------
        # Update Display and Schedule Next Frame
        # -----------------------------------------------------------------------
        # Update the canvas with the new image
        canvas.itemconfig(img_item, image=imgtk)
        
        # Keep a reference to prevent garbage collection of the image
        # (Tkinter would otherwise lose the image due to Python's reference counting)
        canvas.imgtk = imgtk
        
        # Schedule the next frame update in 30ms (approx. 33 FPS)
        root.after(30, update_frame)
    # ===============================================================================
    # Cleanup and Utility Functions
    # ===============================================================================
    def on_close():
        """Clean up resources when the application is closed.
        
        This function ensures proper release of camera and video resources,
        saves the current configuration, and destroys the GUI.
        """
        nonlocal recording, video_writer
        
        print("[Info] Shutting down application")
        
        # Properly close video recording if active
        if recording and video_writer is not None:
            video_writer.release()
            print("[Recording] Stopped due to application close")
            
        # Release camera resources
        cap.release()
        print("[Info] Released camera resources")
        
        # Save current configuration
        config.save()
        print("[Info] Saved configuration")
        
        # Destroy the GUI
        root.destroy()
    
    def apply_speed():
        """Apply the current speed setting to the motor.
        
        This function is called when the speed slider is released.
        It updates the current speed and sends it to the Arduino if running.
        """
        nonlocal current_speed
        current_speed = speed_var.get()
        
        # Only send the speed command if we're currently running
        # This prevents unintended motor starts when just adjusting the slider
        if last_signal == '1':
            send_and_set('1', current_speed)
            
        print(f"[Motor] Speed set to: {current_speed}")
    
    # ===============================================================================
    # Application Startup
    # ===============================================================================
    # Register window close handler
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # Start the frame processing loop
    print("[Info] Starting frame processing loop")
    update_frame()
    
    # Enter the Tkinter main event loop
    root.mainloop()

# ===============================================================================
# Main Entry Point
# ===============================================================================
if __name__ == "__main__":
    print("[Info] Starting Guideway application")
    process_webcam_gui()