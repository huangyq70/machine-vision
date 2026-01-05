import cv2
import numpy as np
import math
import time
from picamera2 import Picamera2 # <-- NEW: Import picamera2

# --- Configuration ---

# ** USER: Set this to the total angular range of your dial in degrees. **
# Example: If your dial goes from 0 to 100, and the angle between the
# "0" mark and the "100" mark is 270 degrees, set this to 270.
DIAL_RANGE_DEGREES = 270.0

# --- Helper Functions ---

def get_angle(p1, p2, cx, cy):
    """Calculates the angle (in degrees) of a line segment relative to the center."""
    # Use atan2 for a 4-quadrant angle
    angle = math.atan2(p1[1] - cy, p1[0] - cx)
    angle_deg = math.degrees(angle)
    # Ensure angle is 0-360
    return (angle_deg + 360) % 360

def distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_dial_circle(frame):
    """
    Detects the main dial circle in the frame.
    Returns (cx, cy, r) or None if not found.
    """
    height, width = frame.shape[:2]
    
    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Tweak parameters for HoughCircles
    # dp: Inverse ratio of resolution. 1 is same as original.
    # minDist: Minimum distance between detected centers.
    # param1: Upper threshold for the Canny edge detector.
    # param2: Threshold for circle center detection. (Higher = "pickier")
    # minRadius/maxRadius: Size of circles to look for.
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=height // 4, 
        param1=100, 
        param2=60,  # <-- Increased to be pickier
        minRadius=int(height * 0.15), 
        maxRadius=int(height * 0.48)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        fully_visible_circles = []
        for (cx_u, cy_u, r_u) in circles[0, :]:
            # --- FIX for RuntimeWarning: overflow ---
            # Cast to standard Python int to prevent unsigned integer overflow/underflow
            # when checking if the circle is within frame boundaries.
            cx, cy, r = int(cx_u), int(cy_u), int(r_u)

            # Ensure the circle is fully within the frame boundaries
            if cx - r > 0 and cx + r < width and cy - r > 0 and cy + r < height:
                fully_visible_circles.append((cx, cy, r))
        
        if fully_visible_circles:
            # Find the largest, fully-visible circle
            largest_circle = max(fully_visible_circles, key=lambda x: x[2])
            return largest_circle
            
    return None

def get_needle_angle(frame, dial_circle):
    """
    Detects the needle lines and calculates the angle of the main pointer.
    Returns (angle, tip_point) or (None, None).
    """
    cx, cy, r = dial_circle
    
    # --- Preprocessing for Line Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    
    # Create a mask to only look for lines *inside* the dial
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), r - 5, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(blurred, blurred, mask=mask)
    
    # Use Canny edge detection
    edges = cv2.Canny(masked_image, 50, 150, apertureSize=3)
    
    # Use HoughLinesP to find line segments
    # minLineLength: Only find lines at least this long
    # maxLineGap: Gaps allowed in a single line
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi / 180, 
        threshold=50, 
        minLineLength=int(r * 0.2), # Find even shorter segments
        maxLineGap=20
    )

    if lines is not None:
        potential_needles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            p1 = (x1, y1)
            p2 = (x2, y2)
            center = (cx, cy)
            
            # Check distances of endpoints from the center
            dist1 = distance(p1, center)
            dist2 = distance(p2, center)
            
            min_dist = min(dist1, dist2)
            max_dist = max(dist1, dist2)

            # --- Needle Filtering Logic ---
            # We want a line that:
            # 1. Starts "super close" to the center (min_dist < r * 0.2)
            # 2. Extends far out to be the pointer (max_dist > r * 0.6)
            
            if min_dist < r * 0.2 and max_dist > r * 0.6:
                # This is a strong candidate for the needle
                
                # Get the point that is *farthest* from the center (the tip)
                tip_point = p1 if dist1 > dist2 else p2
                
                # Calculate the angle of that tip
                angle = get_angle(tip_point, center, cx, cy)
                
                # Store the angle, the tip's distance, and the tip's coordinates
                potential_needles.append((angle, max_dist, tip_point))

        if potential_needles:
            # We might have multiple segments. Find the *longest* one.
            best_needle = max(potential_needles, key=lambda item: item[1])
            best_angle, _, best_tip = best_needle
            return best_angle, best_tip

    return None, None


# --- Main Application ---

def main_video_loop():
    """Main function to run the video loop and dial detection."""
    
    # --- picamera2 Setup ---
    try:
        picam2 = Picamera2()
        # --- FIX ---
        # Removed 'if not picam2.cameras:' as it's not compatible
        # with all picamera2 versions.
        print("picamera2 object created.")
        
        # Configure for preview (faster) or video
        config = picam2.create_preview_configuration(main={"size": (1280, 720)})
        picam2.configure(config)
        picam2.start()
        print("picamera2 started.")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize picamera2.")
        print(f"Error details: {e}")
        print("Please check camera connection and libcamera setup.")
        return

    # --- State variables ---
    dial_circle = None    # Store the (cx, cy, r) of the dial
    calibrated = False  # Becomes true once we set the zero_angle
    zero_angle = None     # The angle of the needle at calibration
    last_time = time.time()
    fps = 0
    w, h = (1280, 720)    # Default width/height, will be updated

    print("Starting video loop...")
    print("Controls:")
    print("  r: Reset dial detection and zero calibration")
    print("  q: Quit")

    try:
        w = picam2.camera_properties['PixelArraySize'][0]
        h = picam2.camera_properties['PixelArraySize'][1]
        print(f"Camera resolution set to: {w}x{h}")
    except Exception as e:
        print(f"Could not get camera resolution, using default {w}x{h}. Error: {e}")

    # Create the window *before* the loop
    cv2.namedWindow('Dial Reader')

    while True:
        
        # --- 1. Frame Capture & Processing ---
        
        # --- Capture frame from picamera2 ---
        frame_rgb = picam2.capture_array()
        # OpenCV (and all our functions) expects BGR format.
        # --- FIX ---
        # Corrected typo from COLOR_RGB_BGR to COLOR_RGB2BGR
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time

        # --- Main Detection Logic ---
        if dial_circle is None:
            cv2.putText(frame, "Detecting dial...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            dial_circle = find_dial_circle(frame)
            if dial_circle:
                print(f"Dial found at {dial_circle}")

        if dial_circle:
            cx, cy, r = dial_circle
            # Draw the detected circle
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            needle_angle, needle_tip = get_needle_angle(frame, dial_circle)

            if needle_angle is not None:
                if not calibrated:
                    zero_angle = needle_angle
                    calibrated = True
                    print(f"Calibrated! Zero angle set to: {zero_angle:.2f} deg")

                # Draw the zero line (yellow)
                zero_x = int(cx + r * 0.9 * math.cos(math.radians(zero_angle)))
                zero_y = int(cy + r * 0.9 * math.sin(math.radians(zero_angle)))
                cv2.line(frame, (cx, cy), (zero_x, zero_y), (0, 255, 255), 2)

                # Draw the needle line (blue)
                cv2.line(frame, (cx, cy), needle_tip, (255, 0, 0), 3)

                # Calculate and display the value
                current_deg = (needle_angle - zero_angle + 360) % 360
                value = current_deg / DIAL_RANGE_DEGREES
                
                cv2.putText(frame, f"Value: {value:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Angle: {current_deg:.1f} deg", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            else:
                cv2.putText(frame, "Detecting needle...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # --- Display Info ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # --- 2. Display ---
        cv2.imshow('Dial Reader', frame)
        
        # --- 3. Handle Keys ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("'q' pressed. Exiting.")
            break
        elif key == ord('r'):
            print("Resetting dial and zero angle...")
            dial_circle = None
            zero_angle = None
            calibrated = False
        
    # --- Cleanup ---
    picam2.stop()
    cv2.destroyAllWindows()
    print("Program exited.")

if __name__ == '__main__':
    main_video_loop()

