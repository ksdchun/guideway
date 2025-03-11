# Repository for Guideway Project
This repo contains the codes for detecting the objects and controlling the car.

# Prerequisite
Please install ultralytics library by following this guide here: https://docs.ultralytics.com/quickstart/#install-ultralytics \
The program requires pyserial to communicate with Arduino using USB port, so please install it using: `pip install pyserial` \
Please also install Arduino IDE in order to upload the program to Arduino: https://www.arduino.cc/en/software 

# How to run
## Step 1: Upload `control_motor.ino` to Arduino board
Connect the USB cable from PC to Arduino board then open the file `control_motor.ino` and upload it to the board. \
In the Arduino code, the motor speed can be adjusted by changing the first parameter in the `driveMotor()` function. \
The speed range is 0-255, where 0 is stopped and 255 is full speed:

```cpp
void loop() {
    // Check if a command is available from Python
    if (Serial.available() > 0) {
        command = Serial.read();
        Serial.println("Received: " + String(command)); // Debugging output
    }
    // Run motor at full power when "1", stop when "0"
    if (command == '1') {
        driveMotor(255, 1);  // Full speed forward - you can change 255 to a lower value for slower speed
    } else if (command == '0') {
        driveMotor(0, 1);     // Stop motor
    }
}
```

## Step 2: Run the object detection code
You can simply run the object detection code (`yolo11n_arduino.py`) using the command:
```bash
python yolo11n_arduino.py
```

### Configuration Parameters
The program can be customized by modifying parameters in the Config class. Below is an explanation of all available configuration options:
### Model Parameters
- `model_path`: Path to your YOLO model (default: 'best_ncnn_model')
- `conf_threshold`: Confidence threshold for object detection (0.0-1.0) - lower values detect more objects but may increase false positives (default: 0.10)

### ROI (Region of Interest) Parameters
- `mask_path`: Path to the mask image that defines the detection region (default: 'Masked Region 2.jpg')
- `mask_threshold`: Pixel intensity threshold for creating binary mask (0-255) - pixels above this value will be set to `mask_max_value`, others to 0 (default: 127)
- `mask_max_value`: Maximum value to use in the binary mask (default: 255)

### Serial Communication Parameters
- `serial_port`: Serial port for Arduino connection - change based on your system:
  - Windows: typically 'COM3', 'COM4', etc.
  - macOS: typically '/dev/cu.usbmodem101' or similar
  - Linux: typically '/dev/ttyACM0' or '/dev/ttyUSB0'
- `baud_rate`: Communication speed with Arduino (default: 9600)

### How to Modify Configuration

To change any parameter, modify the corresponding value in the `Config` class:

```bash
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
```

