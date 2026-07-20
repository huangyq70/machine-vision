#!/usr/bin/env bash
#
# tanda_launch.sh -- launches the liquid-level monitor as a desktop app.
#
# Edit the SETTINGS below for your setup, then double-click the desktop icon
# (created by install_desktop_app.sh) to run it.

# ----------------------- SETTINGS (edit these) -----------------------
PORT="/dev/ttyUSB0"     # pump serial port (see: ls /dev/ttyUSB* /dev/ttyACM*)
BAUD="9600"             # pump baud rate
CAPACITY="12"           # tube capacity in mL
EXTRA_ARGS=""           # e.g. "--smooth 45 --deadband 0.2 --interval 0.5"
SHOW_WINDOW="yes"       # "yes" = show the video window; "no" = headless
# ---------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR" || exit 1

# Use the project's virtualenv python if it exists, else system python3.
PY="$REPO_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Only pass --port if the device is actually present (else run detection-only).
PORT_ARG=""
if [ -e "$PORT" ]; then
    PORT_ARG="--port $PORT --baud $BAUD"
else
    echo "NOTE: $PORT not found -> running detection only (no pump output)."
fi

WINDOW_ARG=""
[ "$SHOW_WINDOW" = "no" ] && WINDOW_ARG="--no-window"

echo "Starting Tanda Liquid Monitor..."
"$PY" liquid_detection/liquid_level_legacy.py \
    --capacity "$CAPACITY" $PORT_ARG $WINDOW_ARG $EXTRA_ARGS

# Keep the terminal open so any message is readable after it exits.
echo ""
echo "Program exited. Press Enter to close this window."
read -r _
