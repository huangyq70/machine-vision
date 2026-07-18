# Raspberry Pi Computer Vision Tools

This repository contains two computer vision tools designed for the Raspberry Pi:

- Liquid Level Monitor: Measures the liquid volume in a test tube and can drive a
  pump to stop at a precise target volume. It uses two AprilTags to rectify the
  tube (perspective + real-world mm scale), a multi-cue meniscus detector that
  rejects graduation markings and works across lighting conditions, a measured
  calibration curve for accurate volume (handles conical bottoms and any tube),
  and a predictive, non-overshooting pump control loop over USB/serial.
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

### 2. Camera Drivers

ArduCam provides the best guide (at least for the camera I was provided) on the driver installation steps. Follow the link below and just go step by step in enabling the camera
[Visit Here](https://docs.arducam.com/Raspberry-Pi-Camera/Pivariety-Camera/Quick-Start-Guide/#step-1-download-the-installation-script)

Make sure to test the camera and ensure that it actually opens with the command

    rpicam-still -t 0


**If you are not using the ArduCam IMX462 Camera, please use the driver installation instructions for your specific camera. If the script does not work then, please let me know as soon as possible**

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



## Tool 1: AprilTag Liquid Level Monitor + Pump Control

Measures the liquid volume in a test tube and (optionally) drives a pump to a
precise target volume. The pipeline is split into a hardware-free core library
(`liquid_detection/liquid_level.py`) plus runnable programs, so the detection
math can be unit-tested off the Pi.

### What makes it accurate and precise

- **AprilTag rectification with a real mm scale.** The two tags (top + bottom)
  are used to warp the tube column into an upright, fixed-size strip and to
  derive millimetres-per-pixel from the known 16 mm tag size. This removes
  camera tilt and perspective and measures in physical units instead of raw,
  distorted pixels. The strip is referenced to the tags' *inner edges*, so it
  contains only the tube and never locks onto the tag borders.

- **Multi-cue meniscus detector that rejects graduation marks.** Instead of a
  single gradient, three cues are fused per row: an *edge* cue (the surface
  line), a median-based *region step* (sustained brightness change between air
  and liquid), and a *texture step* (crisp markings/background above vs.
  refraction-blurred liquid below). Printed graduation ticks produce a big edge
  spike but almost no region/texture step, so they are suppressed — this is what
  lets it work on graduated tubes and with clear liquids. A weighted centroid
  gives a sub-pixel surface location.

- **Measured calibration curve for volume.** A `height_mm → volume_ml` curve
  (built once with `calibrate_tube.py`) linearises the printed scale directly
  and absorbs the conical/round bottom, the exact bore, and any optical bias.
  Geometric (cylinder + cone/hemisphere) and linear models are also available.

- **Temporal filtering.** Confidence gating, outlier rejection, a median +
  EMA filter, a least-squares flow-rate estimate, and a stability flag produce
  a smooth, low-jitter reading suitable for closed-loop control.

- **Predictive, non-overshooting pump stop.** The controller keeps the pump
  running until `volume ≥ target − flow_rate × stop_latency − margin`, so the
  liquid already in transit lands on the target instead of overshooting. It
  streams the live volume continuously and latches off once the target is hit.

### Calibrate your tube (once, for best accuracy)

    python liquid_detection/calibrate_tube.py --config liquid_detection/config.example.json \
           --out calibration.csv

Add known amounts of liquid (syringe/burette, or fill to printed lines), press
`c` and type the true volume for 6–12 levels spanning empty→full, then press
`s`. See `liquid_detection/calibration.example.csv` for the format.

### Running the Liquid Monitor + pump

    python liquid_detection/liquid_level_monitor.py \
           --config liquid_detection/config.example.json \
           --calibration calibration.csv \
           --port /dev/ttyACM0 --target 25

- `--port` is the pump's USB serial device (often `/dev/ttyACM0` or
  `/dev/ttyUSB0`). Omit it for a dry run that prints the pump commands.
- `--target` is the volume in mL to stop at; `--drain` reverses the direction.
- No calibration yet? Pass `--capacity 50` for a rough linear model to see it
  working, then calibrate for real accuracy.

### Controls (Liquid Monitor)

- `q`: Quit
- `space`: Start / pause dosing toward the target
- `+` / `-`: Raise / lower the target by 0.5 mL
- `d`: Toggle fill / drain direction
- `r`: Reset the temporal filter

### Pump wiring and protocol

The Pi talks to the pump over the USB cable as a serial link. Set the command
strings in `pump_controller.py` (`PumpConfig.cmd_run`, `cmd_stop`,
`cmd_setpoint`, `stream_format`) to match your pump's firmware. Defaults send
readable ASCII lines (`RUN\n`, `STOP\n`, `V:12.34\n`) and the live volume is
streamed continuously so the pump/host always has the latest reading. The old
`print` / `autoPrint` / `stopPrint` text queries are still answered for
backward compatibility.

### Configuration

Edit `liquid_detection/config.example.json` (or pass your own with `--config`):

    tube.tag_size_mm          Inner black square size of your AprilTags (mm)
    tube.strip_height_px      Resolution of the rectified tube strip
    tube.width_to_tag_ratio   Strip half-width as a multiple of the tag size
    meniscus.step_band_mm     Band size for the region/texture step (> tick spacing)
    meniscus.w_edge/region/texture   Cue weights in the fused score
    meniscus.min_confidence   Minimum confidence to accept a reading

Command-line flags on the monitor cover `--stop-latency`, `--tolerance`,
`--geometry`, and `--capacity`.

### Tests

The detection math is covered by hardware-free synthetic tests (sub-pixel
accuracy, graduation-mark rejection, calibration interpolation, tube geometry,
and the predictive pump stop):

    python liquid_detection/tests/test_liquid_level.py

### Legacy scripts

The earlier prototypes remain for reference:
`liquid_level_detection_pi_(NEW).py` (gradient-mass),
`liquid_detection_threshold_(OLD).py`, and
`liquid_level_detection_communication_protocol.py`. The new modular pipeline
above supersedes them.



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
