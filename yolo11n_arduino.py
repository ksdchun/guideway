import os 
import datetime

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Fix for external camera took long time to load.

import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO
import json
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import sv_ttk
from PIL import Image, ImageTk, ImageDraw

class Config:
    def __init__(self):
        # load/save GUI config
        self.config_file = os.path.join(os.path.dirname(__file__), 'gui_config.json')
        cfg = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load GUI config: {e}")
        # assign parameters with defaults or loaded config
        self.model_path = cfg.get('model_path', os.path.join(os.path.dirname(__file__), 'weights', 'YOLO11n.pt'))
        self.conf_threshold = cfg.get('conf_threshold', 0.45)
        self.region_json_path = cfg.get('region_json_path', os.path.join(os.path.dirname(__file__), 'regions.json'))
        self.mask_threshold = cfg.get('mask_threshold', 127)
        self.mask_max_value = cfg.get('mask_max_value', 255)
        self.overlap_threshold = cfg.get('overlap_threshold', 0.1)
        self.serial_port = cfg.get('serial_port', '/dev/cu.usbmodem101')
        self.baud_rate = cfg.get('baud_rate', 9600)
        self.motor_speed = cfg.get('motor_speed', 255)  # Default to full speed
        self.recordings_dir = cfg.get('recordings_dir', os.path.join(os.path.dirname(__file__), 'recordings'))
        self.screenshots_dir = cfg.get('screenshots_dir', os.path.join(os.path.dirname(__file__), 'screenshots'))
        self.auto_screenshot = cfg.get('auto_screenshot', True)
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.recordings_dir):
            try:
                os.makedirs(self.recordings_dir)
            except Exception as e:
                print(f"[Warning] Failed to create recordings directory: {e}")
                
        # Create screenshots directory if it doesn't exist
        if not os.path.exists(self.screenshots_dir):
            try:
                os.makedirs(self.screenshots_dir)
            except Exception as e:
                print(f"[Warning] Failed to create screenshots directory: {e}")

    def save(self):
        """Save GUI config to file"""
        cfg = {
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'region_json_path': self.region_json_path,
            'mask_threshold': self.mask_threshold,
            'mask_max_value': self.mask_max_value,
            'overlap_threshold': self.overlap_threshold,
            'serial_port': self.serial_port,
            'baud_rate': self.baud_rate,
            'motor_speed': self.motor_speed,
            'recordings_dir': self.recordings_dir,
            'screenshots_dir': self.screenshots_dir,
            'auto_screenshot': self.auto_screenshot
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save GUI config: {e}")

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

def check_box_in_roi(box_coords, mask):
    """Check if the detected bounding box is within the Region of Interest (ROI)"""
    x1, y1, x2, y2 = box_coords
    box_region = mask[y1:y2, x1:x2]
    
    if box_region.size == 0:
        return False

    overlap_percentage = np.count_nonzero(box_region) / box_region.size
    return overlap_percentage >= config.overlap_threshold

# inference resolution (smaller for speed)
INFER_WIDTH, INFER_HEIGHT = 640, 360

def process_frame(frame, mask, model):
    """Process a single webcam frame with YOLO"""
    # downscale for faster inference
    orig_h, orig_w = frame.shape[:2]
    small = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
    scale_x = orig_w / INFER_WIDTH
    scale_y = orig_h / INFER_HEIGHT
    results = model(small, conf=config.conf_threshold, verbose=False)
    annotated_frame = frame.copy()
    object_detected = False
    
    for r in results:
        for box in r.boxes:
            x1_s, y1_s, x2_s, y2_s = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (int(x1_s * scale_x), int(y1_s * scale_y), int(x2_s * scale_x), int(y2_s * scale_y))
            conf = float(box.conf[0])

            if check_box_in_roi((x1, y1, x2, y2), mask):
                object_detected = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                conf_text = f'Conf: {conf:.2f}'
                cv2.putText(annotated_frame, conf_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # return annotated frame and detection flag; signaling moved to GUI loop
    return annotated_frame, object_detected

def process_webcam_gui():
    """Tkinter GUI with mask selector and 720p live preview."""
    # Load region definitions
    try:
        with open(config.region_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load regions.json: {e}")
        return
    models = list(data.keys())
    if not models:
        messagebox.showerror("Error", "No mask models found.")
        return
    # Setup main window
    root = tk.Tk()
    root.title("Guideway")
    # Start maximized
    root.state('zoomed')
    sv_ttk.set_theme("light")
    # Left control panel for mask selection
    ctrl = tk.Frame(root, width=188)  # Increased width to accommodate recording controls
    ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=3)
    ctrl.pack_propagate(False)

    # Group controls into labeled fram
    model_frame = ttk.LabelFrame(ctrl, text="Mask Settings")
    model_frame.pack(fill=tk.X, padx=5, pady=(10,5))
    ttk.Label(model_frame, text="Mask Model:").pack(padx=5, pady=(5,0))
    mask_var = tk.StringVar(value=models[0])
    combo = ttk.Combobox(model_frame, textvariable=mask_var, values=models, state="readonly")
    combo.pack(fill=tk.X, padx=5, pady=(0,5))
    # Create a frame for mask visualization options to arrange them horizontally
    mask_viz_frame = tk.Frame(model_frame)
    mask_viz_frame.pack(fill=tk.X, padx=5, pady=(0,5))
    
    overlay_var = tk.BooleanVar(value=True)
    detect_preview_var = tk.BooleanVar(value=False)  # Moved from Run Mode section
    
    ttk.Checkbutton(mask_viz_frame, text="Overlay", variable=overlay_var, style="Small.TCheckbutton").pack(side=tk.LEFT, padx=(0,2))
    ttk.Checkbutton(mask_viz_frame, text="Preview", variable=detect_preview_var, style="Small.TCheckbutton").pack(side=tk.LEFT, padx=5)

    detection_frame = ttk.LabelFrame(ctrl, text="Detection Parameters")
    detection_frame.pack(fill=tk.X, padx=5, pady=(5,5))
    threshold_var = tk.DoubleVar(value=config.conf_threshold)
    ttk.Label(detection_frame, text="Confidence:").pack(padx=5, pady=(5,0))
    tk.Scale(detection_frame, variable=threshold_var, from_=0.0, to=1.0, resolution=0.01, orient='horizontal').pack(fill=tk.X, padx=5, pady=(0,5))
    threshold_var.trace_add('write', lambda *args: (setattr(config, 'conf_threshold', threshold_var.get()), config.save()))
    overlap_var = tk.DoubleVar(value=config.overlap_threshold)
    ttk.Label(detection_frame, text="Overlap:").pack(padx=5, pady=(0,0))
    tk.Scale(detection_frame, variable=overlap_var, from_=0.0, to=1.0, resolution=0.01, orient='horizontal').pack(fill=tk.X, padx=5, pady=(0,5))
    overlap_var.trace_add('write', lambda *args: (setattr(config, 'overlap_threshold', overlap_var.get()), config.save()))
    
    # Motor control frame
    motor_frame = ttk.LabelFrame(ctrl, text="Motor Control")
    motor_frame.pack(fill=tk.X, padx=5, pady=(5,5))
    speed_var = tk.IntVar(value=config.motor_speed)
    ttk.Label(motor_frame, text="Motor Speed:").pack(padx=5, pady=(5,0))
    speed_scale = tk.Scale(motor_frame, variable=speed_var, from_=0, to=255, resolution=1, orient='horizontal')
    speed_scale.pack(fill=tk.X, padx=5, pady=(0,5))
    # Update config when speed changes
    speed_var.trace_add('write', lambda *args: (setattr(config, 'motor_speed', speed_var.get()), config.save()))
    # Apply speed when slider is released
    speed_scale.bind("<ButtonRelease-1>", lambda event: apply_speed())

    run_frame = ttk.LabelFrame(ctrl, text="Run Mode")
    run_frame.pack(fill=tk.X, padx=5, pady=(5,10))
    status_label = ttk.Label(run_frame, text="Status: Stopped")
    status_label.pack(padx=5, pady=(5,5))
    ttk.Button(run_frame, text="Start", command=lambda: start_auto()).pack(fill=tk.X, padx=5, pady=(2,2))
    ttk.Button(run_frame, text="Stop", command=lambda: stop_auto()).pack(fill=tk.X, padx=5, pady=(2,5))
    # Create a frame for the checkbox to arrange it horizontally
    check_frame = tk.Frame(run_frame)
    check_frame.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Initialize the auto resume variable
    auto_resume_var = tk.BooleanVar(value=False)
    
    # Add checkbox with updated label (Auto Resume instead of Resume)
    ttk.Checkbutton(check_frame, text="Auto Resume", variable=auto_resume_var, style="Small.TCheckbutton").pack(side=tk.LEFT, padx=(0,1))
    
    # Recording and Screenshot Frame
    recording_frame = ttk.LabelFrame(ctrl, text="Recording & Screenshots")
    recording_frame.pack(fill=tk.X, padx=5, pady=(5,10))
    
    recording_status = ttk.Label(recording_frame, text="Status: Not Recording")
    recording_status.pack(padx=5, pady=(5,5))
    
    record_btn = ttk.Button(recording_frame, text="Start Recording", command=lambda: start_recording())
    record_btn.pack(fill=tk.X, padx=5, pady=(2,2))
    
    stop_record_btn = ttk.Button(recording_frame, text="Stop Recording", command=lambda: stop_recording(), state=tk.DISABLED)
    stop_record_btn.pack(fill=tk.X, padx=5, pady=(2,5))
    
    # Add Screenshot checkbox to the recording frame
    screenshot_check_frame = tk.Frame(recording_frame)
    screenshot_check_frame.pack(fill=tk.X, padx=5, pady=(0,5))
    
    # Initialize the auto screenshot variable
    auto_screenshot_var = tk.BooleanVar(value=config.auto_screenshot)
    ttk.Checkbutton(screenshot_check_frame, text="Auto Screenshot", variable=auto_screenshot_var, style="Small.TCheckbutton").pack(side=tk.LEFT, padx=0)
    auto_screenshot_var.trace_add('write', lambda *args: setattr(config, 'auto_screenshot', auto_screenshot_var.get()))
    
    # Create styles for small buttons and checkbuttons
    style = ttk.Style()
    style.configure("Small.TButton", font=("TkDefaultFont", 13))
    style.configure("Small.TCheckbutton", font=("TkDefaultFont", 8))

    # control variables
    running = False
    last_signal = None
    current_speed = config.motor_speed
    
    # Video recording variables
    recording = False
    video_writer = None
    
    # Define frame variables at the outer scope so they're accessible to all functions
    current_frame = None
    annotated_frame = None
    def send_and_set(sig, speed=None):
        nonlocal last_signal, current_speed
        if speed is None:
            speed = current_speed
        else:
            current_speed = speed
            
        print(f"[GUI] send_and_set called with: {sig}, speed: {speed}")
        if arduino:
            # Format command as sig:speed (e.g., "1:200" for running at speed 200)
            command = f"{sig}:{speed}\n"
            arduino.write(command.encode())
            time.sleep(0.1)
            print(f"[Arduino] Sent {command.strip()}")
        else:
            print(f"[Warning] Arduino not connected. Can't send {sig}:{speed}")
        status_label.config(text=f"Status: {'Running' if sig=='1' else 'Stopped'}")
        last_signal = sig
    def start_auto():
        print("[GUI] Start pressed")
        nonlocal running
        running = True
        send_and_set('1', speed_var.get())
    def stop_auto():
        print("[GUI] Stop pressed")
        nonlocal running
        running = False
        auto_resume_var.set(False)
        send_and_set('0', 0)  # Always stop with speed 0
    def pause_auto():
        """Pause on detection without disabling Auto-Resume."""
        print("[GUI] Paused on detection")
        nonlocal running, annotated_frame
        running = False
        send_and_set('0', 0)  # Always stop with speed 0
        
        # Take a screenshot when object is detected and car stops
        if auto_screenshot_var.get() and annotated_frame is not None:
            print("[Screenshot] Taking screenshot with detection")
            take_screenshot(annotated_frame)
        
    def start_recording():
        """Start recording video to a file"""
        nonlocal recording, video_writer
        if recording:
            return
            
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(config.recordings_dir, f"recording_{timestamp}.mp4")
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (webcam_width, webcam_height))
        
        if video_writer.isOpened():
            print(f"[Recording] Started: {video_path}")
            recording = True
            recording_status.config(text=f"Status: Recording")
            record_btn.config(state=tk.DISABLED)
            stop_record_btn.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "Failed to create video writer.")
            video_writer = None
    
    def stop_recording():
        """Stop recording video"""
        nonlocal recording, video_writer
        if not recording or video_writer is None:
            return
            
        video_writer.release()
        video_writer = None
        recording = False
        print("[Recording] Stopped")
        recording_status.config(text="Status: Not Recording")
        record_btn.config(state=tk.NORMAL)
        stop_record_btn.config(state=tk.DISABLED)
    
    # Location directory selection functions removed as UI buttons were removed
            
    def take_screenshot(frame_to_save):
        """Take a screenshot and save it to the screenshots directory"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(config.screenshots_dir, f"screenshot_{timestamp}.jpg")
        
        # Apply overlay to screenshot if overlay is enabled
        if overlay_var.get():
            # Convert frame to PIL Image for overlay
            pil_frame = Image.fromarray(cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay_img = Image.new("RGBA", (webcam_width, webcam_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_img)
            
            # Draw the polygon overlay
            regions = data.get(mask_var.get(), {}).get("regions", {})
            for pts in regions.values():
                coords = [(x, y) for x, y in pts]
                draw.polygon(coords, fill=(255, 0, 0, 128))
            
            # Composite the image with overlay
            composite = Image.alpha_composite(pil_frame, overlay_img)
            frame_with_overlay = cv2.cvtColor(np.array(composite.convert('RGB')), cv2.COLOR_RGB2BGR)
            
            # Save the composited image
            cv2.imwrite(screenshot_path, frame_with_overlay)
        else:
            # Save without overlay
            cv2.imwrite(screenshot_path, frame_to_save)
            
        print(f"[Screenshot] Saved: {screenshot_path}")

    # Video display canvas
    webcam_width, webcam_height = 1280, 720
    canvas = tk.Canvas(root, width=webcam_width, height=webcam_height)
    canvas.pack(side=tk.RIGHT)
    # single image item for reuse (avoid creating per frame)
    img_item = canvas.create_image(0, 0, anchor='nw', image=None)
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam.")
        root.destroy()
        return
    # Build mask for the selected model
    def build_mask(model_name):
        regions = data.get(model_name, {}).get("regions", {})
        m = np.zeros((webcam_height, webcam_width), dtype=np.uint8)
        for pts in regions.values():
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(m, [poly], config.mask_max_value)
        _, m = cv2.threshold(m, config.mask_threshold, config.mask_max_value, cv2.THRESH_BINARY)
        return m
    mask_dict = {'mask': build_mask(mask_var.get())}
    def on_model_change(*args):
        mask_dict['mask'] = build_mask(mask_var.get())
    mask_var.trace_add('write', on_model_change)
    # Frame update loop
    model = YOLO(config.model_path)
    def update_frame():
        nonlocal running, last_signal, current_frame, annotated_frame
        ret, frame = cap.read()
        if not ret:
            return
        
        # Resize the current frame
        current_frame = cv2.resize(frame, (webcam_width, webcam_height))
        
        # Determine if we should run inference (for detection, preview or auto-resume)
        do_infer = running or detect_preview_var.get() or auto_resume_var.get()
        if do_infer:
            annotated_frame, detected = process_frame(current_frame, mask_dict['mask'], model)
        else:
            detected = False
            annotated_frame = current_frame.copy()  # Ensure annotated_frame exists for screenshots

        # Stop on detection if currently running
        if running and detected and last_signal != '0':
            pause_auto()

        # Auto-resume when detection stops
        if auto_resume_var.get() and not detected and last_signal == '0':
            start_auto()

        # Choose frame to display: annotated if in running or preview mode, else raw
        if running or detect_preview_var.get():
            frame_out = annotated_frame
        else:
            frame_out = frame
        
        # conditional semi-transparent overlay
        has_overlay = overlay_var.get()
        if has_overlay:
            pil_frame = Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay_img = Image.new("RGBA", (webcam_width, webcam_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_img)
            regions = data.get(mask_var.get(), {}).get("regions", {})
            for pts in regions.values():
                coords = [(x, y) for x, y in pts]
                draw.polygon(coords, fill=(255, 0, 0, 128))
            composite = Image.alpha_composite(pil_frame, overlay_img)
            imgtk = ImageTk.PhotoImage(composite)
            display_frame = np.array(composite.convert('RGB'))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else:
            pil_frame = Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(pil_frame)
            display_frame = frame_out
            
        # Record video if recording is active
        if recording and video_writer is not None:
            # Use the same overlay setting as the main display
            if has_overlay:
                video_writer.write(display_frame)
            else:
                # Always record detected objects in frame
                if running or detect_preview_var.get():
                    video_writer.write(annotated_frame)
                else:
                    video_writer.write(frame)
        
        canvas.itemconfig(img_item, image=imgtk)
        canvas.imgtk = imgtk
        root.after(30, update_frame)
    def on_close():
        nonlocal recording, video_writer
        if recording and video_writer is not None:
            video_writer.release()
        cap.release()
        config.save()
        root.destroy()
    # Function to apply current speed without changing run state
    def apply_speed():
        nonlocal current_speed
        current_speed = speed_var.get()
        # Only send the speed if we're currently running
        if last_signal == '1':
            send_and_set('1', current_speed)
        print(f"[Motor] Speed set to: {current_speed}")
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()
    root.mainloop()

if __name__ == "__main__":
    process_webcam_gui()