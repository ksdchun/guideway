# Guideway: YOLO Object Detection with Arduino Control

## Overview

Guideway is a sophisticated computer vision application that combines YOLO (You Only Look Once) object detection with Arduino-based motor control. The system is designed to detect objects within specified regions of interest and automatically control a motor system based on these detections.

This document provides a comprehensive explanation of the system architecture, functionality, and usage.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Key Components](#key-components)
   - [Configuration System](#configuration-system)
   - [Computer Vision Pipeline](#computer-vision-pipeline)
   - [Arduino Control System](#arduino-control-system)
   - [User Interface](#user-interface)
   - [Recording System](#recording-system)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Guide](#usage-guide)
5. [Technical Implementation Details](#technical-implementation-details)
6. [File Structure](#file-structure)
7. [Troubleshooting](#troubleshooting)

## System Architecture

The system is built with a modular architecture consisting of five main components:

1. **Configuration Management** - Persistent settings via JSON
2. **Computer Vision Pipeline** - YOLO-based object detection with region masking
3. **Hardware Interface** - Serial communication with Arduino
4. **User Interface** - Modern Tkinter GUI with Sun Valley theme
5. **Recording System** - Video and screenshot capabilities

![System Architecture Diagram](system_architecture.png)

## Key Components

### Configuration System

The `Config` class manages all application settings with persistent storage in `gui_config.json`. It handles:

- **Model Settings**
  - Path to YOLO model file
  - Confidence threshold for detections

- **Region Settings**
  - Path to region definitions file
  - Mask threshold parameters
  - Overlap threshold for detection

- **Arduino Settings**
  - Serial port configuration
  - Baud rate
  - Motor speed control

- **Recording Settings**
  - Directories for videos and screenshots
  - Auto-screenshot preferences

All settings are loaded at startup with sensible defaults and saved automatically when changed through the UI.

### Computer Vision Pipeline

The application uses:

- **YOLO11n Model** - Ultralytics YOLO implementation for object detection
- **Region Masking** - Polygon-based regions of interest to limit detection areas
- **Overlap Detection** - Configurable threshold for determining if objects are within ROIs
- **Optimized Processing** - Downscaled inference for performance with proper rescaling

The pipeline follows these steps:
1. Capture frame from webcam
2. Resize to smaller dimensions for faster inference
3. Run YOLO detection on the resized frame
4. Scale detection coordinates back to original frame size
5. Check if detections overlap with defined regions of interest
6. Annotate the frame with detection boxes and confidence values
7. Return detection status for control logic

### Arduino Control System

The system communicates with an Arduino via serial port to control motors:

- **Command Protocol** - Simple text-based protocol (`signal:speed\n`)
- **Speed Control** - Variable motor speed (0-255)
- **Auto-Resume** - Optional automatic resumption when objects leave detection zone

The Arduino is expected to parse commands in the format `signal:speed\n` where:
- `signal` is either `1` (run) or `0` (stop)
- `speed` is a value from 0-255 for PWM control

### User Interface

The modern GUI provides:

- **Mask Selection** - Dropdown for different region configurations
- **Detection Parameters** - Adjustable confidence and overlap thresholds
- **Motor Controls** - Speed adjustment and run mode controls
- **Recording Controls** - Video recording and screenshot functionality
- **Visual Feedback** - Live camera feed with optional mask overlay and detection visualization

The interface is built with Tkinter and enhanced with the Sun Valley theme (sv_ttk) for a modern appearance.

### Recording System

The application supports:

- **Video Recording** - MP4 format with timestamp-based filenames
- **Auto Screenshots** - Automatic screenshots on detection events
- **Overlay Options** - Configurable visualization of detection regions

All recordings and screenshots are saved with timestamps for easy organization.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- OpenCV
- NumPy
- Ultralytics YOLO
- Tkinter
- PIL (Pillow)
- sv_ttk
- pySerial
- Arduino board with appropriate firmware

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/guideway.git
   cd guideway
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Connect your Arduino and upload the appropriate firmware.

4. Create or modify the `regions.json` file to define your detection regions.

5. Run the application:
   ```
   python yolo11n_arduino.py
   ```

## Usage Guide

### Setting Up Regions

The `regions.json` file defines the regions of interest for detection. The format is:

```json
{
  "model_name": {
    "regions": {
      "region1": [[x1, y1], [x2, y2], ...],
      "region2": [[x1, y1], [x2, y2], ...]
    }
  }
}
```

You can define multiple models with different region configurations.

### Basic Operation

1. **Select a Mask Model** - Choose from the dropdown menu
2. **Adjust Parameters** - Set confidence and overlap thresholds
3. **Set Motor Speed** - Adjust the speed slider
4. **Start Detection** - Click the Start button
5. **Enable Auto-Resume** - Check this option to automatically restart after detection

### Recording and Screenshots

- **Start/Stop Recording** - Use the recording buttons to capture video
- **Auto Screenshot** - Enable to automatically capture images when objects are detected

## Technical Implementation Details

### Threading Model

The application uses a single-threaded architecture with Tkinter's event loop. The frame processing happens within this loop at approximately 33 frames per second (30ms interval).

### Performance Optimization

- Inference is performed at reduced resolution (640×360) for speed
- The original frame is maintained for display and recording
- Detection coordinates are scaled back to the original resolution

### Error Handling

The application implements graceful degradation for various failure scenarios:
- Arduino connection issues
- Missing configuration files
- Failed directory creation
- Camera initialization problems

### Memory Management

- Image objects are reused to prevent memory leaks
- Video writers are properly released when recording stops
- Resources are cleaned up on application exit

## File Structure

- `yolo11n_arduino.py` - Main application file
- `gui_config.json` - Configuration storage
- `regions.json` - Region definitions
- `weights/` - Directory containing YOLO model files
- `recordings/` - Directory for saved videos
- `screenshots/` - Directory for saved images

## Troubleshooting

### Common Issues

1. **Arduino Not Connecting**
   - Check the serial port setting in the configuration
   - Verify the Arduino is properly connected
   - Ensure the correct firmware is uploaded

2. **Camera Not Working**
   - Check if another application is using the camera
   - Try changing the camera index (currently set to 1)
   - Verify camera drivers are installed

3. **Detection Not Working**
   - Check that the YOLO model file exists
   - Verify regions are properly defined
   - Adjust confidence and overlap thresholds

4. **Recording Issues**
   - Ensure the recordings directory exists and is writable
   - Check available disk space
   - Verify OpenCV is compiled with video codecs

### Logs

The application outputs informative logs with prefixes to help diagnose issues:
- `[Info]` - General information
- `[Warning]` - Non-critical issues
- `[Error]` - Critical problems
- `[Success]` - Successful operations
- `[GUI]` - User interface events
- `[Arduino]` - Communication with Arduino
- `[Recording]` - Video recording events
- `[Screenshot]` - Screenshot events

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLO implementation
- Sun Valley theme for Tkinter
- OpenCV community for computer vision tools
