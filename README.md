# Raspberry Pi Computer Vision Tools

This repository contains two computer vision tools designed for the Raspberry Pi:

- Liquid Level Monitor: Tracks liquid levels in transparent tubes using AprilTags and gradient analysis.
- Analog Gauge Reader: Reads values from analog dial gauges (like pressure or temperature gauges).



## Hardware Requirements

- Raspberry Pi  
  Raspberry Pi 4 or 5 recommended (running Raspberry Pi OS Bookworm).

- Camera
  - Primary: Raspberry Pi Camera Module (e.g., IMX462 / IMX290) via CSI
  - Fallback: USB Webcam (supported automatically if PiCamera fails)



## Installation Guide

### 1. System Dependencies (Raspberry Pi)

The picamera2 library comes pre-installed on Raspberry Pi OS, but you need to ensure your system is up to date and has the necessary build tools.

    sudo apt update
    sudo apt install python3-libcamera python3-kms++ python3-prctl libcap-dev



### 2. Camera Configuration

Ensure your camera is enabled in:

    /boot/firmware/config.txt

For an IMX462 / IMX290, ensure this line exists:

    dtoverlay=imx290,clock-frequency=37125000

Then reboot:

    sudo reboot



### 3. Python Environment Setup (Crucial)

Because Picamera2 is a system library and OpenCV is a PyPI library, you must create a virtual environment with access to system site packages.

#### Step A: Create the Environment

    # Create venv with access to system libs (like picamera2)
    python3 -m venv --system-site-packages .venv

    # Activate it
    source .venv/bin/activate

#### Step B: Install Python Dependencies

    # Install OpenCV with Contrib modules (for ArUco / AprilTags)
    pip install -r requirements.txt

#### Step C: Fix Binary Incompatibility

OpenCV often installs a newer version of NumPy that conflicts with the system-level Picamera2 library.

    pip uninstall numpy -y



## Tool 1: AprilTag Liquid Level Monitor

Tracks liquid levels in transparent tubes using AprilTags for dynamic ROI alignment and a Gradient Mass algorithm for robust meniscus detection.

### Features

- Dynamic ROI  
  Uses two AprilTags (Top and Bottom) to automatically find and straighten the test tube view.

- Gradient Mass Detection  
  Uses morphological filtering to detect liquid levels even in cloudy or low-contrast fluids.

- Bubble Rejection  
  Intelligent filtering ignores small bubbles and vertical scratches.

### Running the Liquid Monitor

Open liquid_detection/liquid_level_detection_pi_(NEW).py and run the program

### Controls (Liquid Monitor)

- q: Quit

### Configuration

You can adjust these variables to your current setup:

    TUBE_CAPACITY: Total volume of your tube in ml (default: 15.0)
    SMOOTHING_WINDOW: How many frames are used to find the median volume
    GRADIENT_FILTER_WIDTH: How wide the features need to cover to be recogonized
    GRADIENT_SUM_WIDTH: what area of the middle needs to be used to calc the gradient change



## Tool 2: Analog Gauge Reader

Reads values from circular analog gauges by detecting the dial face and the needle angle.

### Features

- Automatic Dial Detection  
  Uses Hough Circle Transform to lock onto the gauge face.

- Needle Tracking  
  Detects the pointer needle using line segments and calculates the angle relative to the center.

- Auto-Calibration  
  Sets the zero point automatically based on the initial needle position (or reset via keypress).

### Running the Gauge Reader

With your virtual environment activated:

    python dial_detection/dial_detection_pi_camera.py

### Controls (Gauge Reader)

- q: Quit
- r: Reset dial detection and re-calibrate the zero angle (useful if you move the camera)

### Configuration

Open analog_gauge_reader.py to adjust:

    DIAL_RANGE_DEGREES: The total angular sweep of your specific gauge (e.g., 270 degrees)



## Troubleshooting

### "Status: Waiting for Tags..."

- Ensure both AprilTags are visible (Liquid Monitor).

### "Picamera2 not found"

- Ensure you created the venv with --system-site-packages.

### Window Glitching

- If the visualization panels bounce, ensure your monitor resolution supports the window size (default 1600x900).
