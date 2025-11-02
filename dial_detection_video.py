import cv2
import numpy as np
import math
import time

# --- Configuration ---

# ** USER: Set this to the total angular range of your dial in degrees. **
# For example, if your dial goes from 0 to 100, and the angle between
# the '0' mark and the '100' mark is 270 degrees, set this to 270.0.
# The user's original name was DEGREE_IS_ONE, which is renamed here for clarity.
DIAL_RANGE_DEGREES = 270.0

# Set the video source. 0 is typically the default webcam.
# You can also use a file path like 'my_video.mp4'.
VIDEO_SOURCE = './videos/IMG_7751.mp4'

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
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=int(height/2))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        fully_visible_circles = []
        for (cx, cy, r) in circles[0, :]:
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
    # Adjust minLineLength relative to the radius for robustness
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=r//3, maxLineGap=20)

    if lines is not None:
        potential_needles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check distance of each endpoint to the circle's center
            dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
            dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
            
            # Store line with the minimum distance of one of its endpoints
            min_dist = min(dist1, dist2)
            potential_needles.append(((x1, y1, x2, y2), min_dist, dist1, dist2))
        
        # We need at least two lines to form the composite (average) needle
        if len(potential_needles) >= 2:
            # Sort lines by distance to center (closest first)
            sorted_needles = sorted(potential_needles, key=lambda item: item[1])
            
            # Get the two lines closest to the center
            line1_coords, _, l1_dist1, l1_dist2 = sorted_needles[0]
            line2_coords, _, l2_dist1, l2_dist2 = sorted_needles[1]

            # Find the endpoint of each line that is FURTHEST from the center
            far_point1 = (line1_coords[0], line1_coords[1]) if l1_dist1 > l1_dist2 else (line1_coords[2], line1_coords[3])
            far_point2 = (line2_coords[0], line2_coords[1]) if l2_dist1 > l2_dist2 else (line2_coords[2], line2_coords[3])

            # Calculate the midpoint of these two "tip" points
            needle_tip_x = int((far_point1[0] + far_point2[0]) / 2)
            needle_tip_y = int((far_point1[1] + far_point2[1]) / 2)
            
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

    print("Starting video loop...")
    print("Press 'r' to reset dial detection and zero calibration.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error.")
            break
            
        display_frame = frame.copy()

        # --- Frame Rate Calculation ---
        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = frame_count / (current_time - last_fps_time)
            frame_count = 0
            last_fps_time = current_time

        # --- Main Logic ---

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
                dial_value = degrees_from_zero / DIAL_RANGE_DEGREES
                
                # Display text
                cv2.putText(display_frame, f"Value: {dial_value:.3f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Angle: {degrees_from_zero:.1f} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(display_frame, "Detecting needle...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Display FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the final image
        cv2.imshow('Dial Reader', display_frame)

        # --- User Controls ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting detection and calibration...")
            dial_circle = None
            zero_angle = None

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited.")

if __name__ == '__main__':
    main_video_loop()
