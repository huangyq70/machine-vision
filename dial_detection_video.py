import cv2
import numpy as np
import math
import time

# --- Configuration ---

# ** USER: Set this to the total angular range of your dial in degrees. **
# For example, if your dial goes from 0 to 100, and the angle between
# the '0' mark and the '100' mark is 270 degrees, set this to 270.0.
# The user's original name was DEGREE_IS_ONE, which is renamed here for clarity.
DIAL_RANGE_DEGREES = 250.0

# Set the video source. 0 is typically the default webcam.
# You can also use a file path like 'my_video.mp4'.
VIDEO_SOURCE = 'videos/dial_video_2.mp4'

# --- End Configuration ---


def find_dial_circle(frame):
    """
    Finds the largest, fully-visible circle (the dial) in a frame.
    
    Args:
        frame: The input video frame (BGR).

    Returns:
        (cx, cy, r) tuple for the circle, or None if not found.
    """
    height, width = frame.shape[:2]
    
    # Preprocessing
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    # Detect circles
    # --- TUNED PARAMETERS V3 ---
    # dp=1: Use full resolution for accumulator.
    # param1=50: Canny high threshold (usually fine)
    # param2=60: Increased from 50. Requires an even stronger "vote" for a circle.
    # minRadius/maxRadius: Tightly constrained based on sample image.
    #                      Assumes dial radius is ~35-48% of frame height.
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=60, 
                               minRadius=int(height*0.35), maxRadius=int(height*0.48))

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
            # Return the largest fully-visible circle
            largest_circle = max(fully_visible_circles, key=lambda x: x[2])
            return largest_circle

    return None

def get_needle_angle(frame, circle):
    """
    Finds the needle's angle within the detected dial area.
    
    Args:
        frame: The input video frame (BGR).
        circle: The (cx, cy, r) tuple for the dial.

    Returns:
        (angle_degrees, needle_tip_coords) or (None, None) if not found.
    """
    cx, cy, r = circle
    
    # --- Preprocessing inside a circular mask ---
    # Create a mask to isolate the dial area, reducing noise
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    # Apply blur only within the masked region
    blurred_image = cv2.GaussianBlur(masked_gray, (9, 9), 2)

    # Use Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)
    
    # --- Detect Lines (Needle) ---
    # --- TUNED PARAMETERS V3 ---
    # threshold=80: Lowered to find more line segments.
    # minLineLength=int(r*0.2): Reduced to find *segments* of the needle.
    # maxLineGap=int(r*0.1): Increased to help *connect* broken segments.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, 
                            minLineLength=int(r*0.2), maxLineGap=int(r*0.1))

    if lines is not None:
        potential_needles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check distance of each endpoint to the circle's center
            dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
            dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
            
            min_dist = min(dist1, dist2)
            max_dist = max(dist1, dist2)

            # --- NEW NEEDLE LOGIC (V4) ---
            # Filter for lines that start *near* the center and point *outwards*.
            # This avoids the central hub and small tick marks.
            #
            # ** FIX for direction flipping **
            # We now require max_dist > r * 0.6 (was 0.3).
            # This ensures we *only* detect the long "pointer" end of the
            # needle and *ignore* the short "tail" end, which was
            # causing the angle to flip 180 degrees.
            if min_dist < r * 0.3 and max_dist > r * 0.5:
                # Store the endpoint that is *furthest* from the center
                if dist1 > dist2:
                    far_point = (x1, y1)
                else:
                    far_point = (x2, y2)
                
                # Store the far point and its distance
                potential_needles.append((far_point, max_dist))
        
        # If we found any valid needle segments
        if potential_needles:
            # The true needle tip is the one *furthest* from the center
            best_needle = max(potential_needles, key=lambda item: item[1])
            needle_tip_x, needle_tip_y = best_needle[0]
            
            # --- Calculate Angle ---
            # Use atan2 for a 4-quadrant angle
            angle_rad = math.atan2(needle_tip_y - cy, needle_tip_x - cx)
            
            # Convert to degrees (0-360), where 0 is to the right
            angle_deg = (math.degrees(angle_rad) + 360) % 360
            
            return angle_deg, (needle_tip_x, needle_tip_y)
            
    return None, None


def main_video_loop():
    """
    Main function to capture video, detect dial, calibrate, and show value.
    """
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    # --- State variables ---
    dial_circle = None  # Stores (cx, cy, r) of the dial
    zero_angle = None   # Stores the calibrated "zero" angle
    
    last_fps_time = time.time()
    frame_count = 0
    fps = 0

    # --- New variables for frame-by-frame control ---
    is_paused = True         # Start in paused state
    read_new_frame = True    # Flag to control when to read a new frame
    frame = None             # Holds the current frame

    print("Starting video loop...")
    print("Press 'p' to toggle pause/play.")
    print("Press 'n' to advance one frame (while paused).")
    print("Press 'r' to reset dial detection and zero calibration.")
    print("Press 'q' to quit.")

    while True:
        # --- Frame Reading Logic ---
        if read_new_frame:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error.")
                break
            
            # If we are in "play" mode, keep reading new frames.
            # If we are paused, stop reading after this one.
            if not is_paused:
                read_new_frame = True
            else:
                read_new_frame = False
        
        # If no frame has been read yet, skip
        if frame is None:
            continue
            
        display_frame = frame.copy()

        # --- Frame Rate Calculation (only updates when playing) ---
        if not is_paused:
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time

        # --- Main Logic (runs on the current frame, paused or not) ---

        # 1. Find Dial (only once, or if reset)
        # This assumes the camera and dial are stationary.
        # Running circle detection every frame is very slow.
        if dial_circle is None:
            dial_circle = find_dial_circle(frame)
            if dial_circle is None:
                cv2.putText(display_frame, "Detecting dial...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        # 2. If Dial is found, find needle and calculate value
        if dial_circle is not None:
            cx, cy, r = dial_circle
            
            # Draw the detected dial
            cv2.circle(display_frame, (cx, cy), r, (0, 255, 0), 3)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)

            # Find the needle
            current_angle, needle_tip = get_needle_angle(frame, dial_circle)

            if current_angle is not None:
                # 3. Calibrate Zero Position (if not already set)
                if zero_angle is None:
                    zero_angle = current_angle
                    print(f"CALIBRATED: Zero angle set to {zero_angle:.2f} degrees.")
                
                # 4. Calculate and Display Value
                
                # Draw the zero line (Yellow)
                zero_rad = math.radians(zero_angle)
                zero_x = int(cx + (r * 0.9) * math.cos(zero_rad))
                zero_y = int(cy + (r * 0.9) * math.sin(zero_rad))
                cv2.line(display_frame, (cx, cy), (zero_x, zero_y), (0, 255, 255), 2)

                # Draw the current needle line (Blue)
                cv2.line(display_frame, (cx, cy), needle_tip, (255, 0, 0), 3)

                # Calculate the difference, handling 360-degree wrapping
                # This assumes the dial moves clockwise from zero
                degrees_from_zero = (current_angle - zero_angle + 360) % 360
                
                # Calculate the final value as a ratio of the total range
                # A value of 1.0 means it has traveled the full DIAL_RANGE_DEGREES
                dial_value = (degrees_from_zero / DIAL_RANGE_DEGREES) * 10
                
                # Display text
                cv2.putText(display_frame, f"Value: {dial_value:.3f} mPa", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Angle: {degrees_from_zero:.1f} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(display_frame, "Detecting needle...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Display FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display Paused status
        if is_paused:
            cv2.putText(display_frame, "PAUSED", (display_frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the final image
        cv2.imshow('Dial Reader', display_frame)

        # --- User Controls ---
        # Wait indefinitely if paused (0), or 1ms if playing
        wait_duration = 0 if is_paused else 1
        key = cv2.waitKey(wait_duration) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting detection and calibration...")
            dial_circle = None
            zero_angle = None
            read_new_frame = True # Need to read a new frame after reset
        elif key == ord('p'):
            is_paused = not is_paused
            if not is_paused:
                # When unpausing, reset FPS counter and start reading frames
                read_new_frame = True
                last_fps_time = time.time()
                frame_count = 0
            print("Paused" if is_paused else "Playing")
        elif key == ord('n'):
            if is_paused:
                # If paused, just read one new frame
                read_new_frame = True

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited.")

if __name__ == '__main__':
    main_video_loop()




