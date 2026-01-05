import cv2
import numpy as np
import math

def detect_dial_and_needle(image_path):
    """
    Loads an image, finds the main dial (circle), draws all detected lines,
    and then draws a composite blue line representing the direction of the two
    lines closest to the dial's center.

    Args:
        image_path (str): Local file path of the image.
    """
    try:
        # --- Step 1: Load the Image ---
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Error: Could not load image from {image_path}")
            print("Please make sure the file path is correct and the image exists.")
            return

        image_with_detections = original_image.copy()

        image_with_final_prediction = original_image.copy()
        
        height, width = original_image.shape[:2]

        # --- Step 2: Preprocessing ---
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

        # --- Step 3: Detect Circles using the blurred Image ---
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=50, param2=30, minRadius=50, maxRadius=int(height/2))

        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            fully_visible_circles = []
            for (cx_u, cy_u, r_u) in circles[0, :]:
                # --- FIX for RuntimeWarning: overflow ---
                # Cast to standard Python int to prevent unsigned integer overflow/underflow
                # when checking if the circle is within frame boundaries.
                cx, cy, r = int(cx_u), int(cy_u), int(r_u)

                if cx - r > 0 and cx + r < width and cy - r > 0 and cy + r < height:
                    fully_visible_circles.append((cx, cy, r))

            if fully_visible_circles:
                # Find the largest circle
                largest_circle = max(fully_visible_circles, key=lambda x: x[2])
                cx, cy, r = largest_circle
                
                # Draw the detected circle and its center
                cv2.circle(image_with_detections, (cx, cy), r, (0, 255, 0), 3)
                cv2.circle(image_with_detections, (cx, cy), 5, (0, 255, 0), -1)

                cv2.circle(image_with_final_prediction, (cx, cy), r, (0, 255, 0), 3)
                cv2.circle(image_with_final_prediction, (cx, cy), 5, (0, 255, 0), -1)
                print("Dial (circle) detected!")

                # --- Step 4: Detect and Filter Lines (Needle) ---
                # Use Canny edge detection on the blurred image for better line finding
                edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)
                # Adjust minLineLength relative to radius
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=r//2, maxLineGap=20)
                
                if lines is not None:
                    # Draw all detected lines in red first
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(image_with_detections, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    print(f"Found and drew {len(lines)} lines.")

                    # Identify the two lines closest to the center to calculate the composite line
                    potential_needles = [] # Store lines with their distances for sorting
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Check distance of each endpoint to the circle's center
                        dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
                        dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
                        
                        min_dist = min(dist1, dist2)
                        # Store coordinates, min distance, and both endpoint distances
                        potential_needles.append(((x1, y1, x2, y2), min_dist, dist1, dist2))
                    
                    # We need at least two lines to form the composite line
                    if len(potential_needles) >= 2:
                        # Sort the potential needles by distance in ascending order (closest first)
                        sorted_needles = sorted(potential_needles, key=lambda item: item[1])
                        
                        # --- Calculate and draw the composite blue line ---
                        # Get data for the two closest lines
                        line1_coords, _, l1_dist1, l1_dist2 = sorted_needles[0]
                        line2_coords, _, l2_dist1, l2_dist2 = sorted_needles[1]

                        # For each line, find the endpoint that is FURTHEST from the center
                        far_point1 = (line1_coords[0], line1_coords[1]) if l1_dist1 > l1_dist2 else (line1_coords[2], line1_coords[3])
                        far_point2 = (line2_coords[0], line2_coords[1]) if l2_dist1 > l2_dist2 else (line2_coords[2], line2_coords[3])

                        # Calculate the midpoint between these two furthest points
                        midpoint_x = int((far_point1[0] + far_point2[0]) / 2)
                        midpoint_y = int((far_point1[1] + far_point2[1]) / 2)
                        
                        # Draw the thick blue line from the midpoint to the center
                        cv2.line(image_with_final_prediction, (midpoint_x, midpoint_y), (cx, cy), (255,0,0), 4)
                        print("Drew composite blue line based on the two closest needles.")
                    else:
                        print("Not enough lines found (need at least 2) to draw composite blue line.")
                else:
                    print("Dial detected, but no lines were found.")

            else:
                print("No circles were found that are fully on screen.")
        else:
            print("No circles were detected.")

        # --- Step 5: Show the Images ---
        cv2.imshow('Blurred Image', blurred_image)
        cv2.imshow('Detected Dial and Needle', image_with_detections)
        cv2.imshow('Final Prediction', image_with_final_prediction)

        print("Press any key to close the windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # ** USER: Update this path to your image file **
    image_file = 'images/IMG_7750.jpg' 
    detect_dial_and_needle(image_file)
