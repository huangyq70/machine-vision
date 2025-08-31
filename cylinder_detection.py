import cv2
import numpy as np

def detect_meniscus(image_path):
    """
    Loads an image, isolates a yellow liquid, finds the tallest contour
    (to avoid side noise), and identifies the bottom of the meniscus curve.

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

        # --- Step 2: Color Segmentation for Yellow ---
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the color yellow in HSV
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([30, 255, 255])

        # Create a mask that isolates only the yellow pixels
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # --- NEW: More aggressive cleaning to remove side noise ---
        # This erodes thin connections and then restores the main body
        kernel = np.ones((5, 5), np.uint8)
        # First, close small holes in the main body
        closed_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        # Erode to sever thin connections to side noise
        eroded_mask = cv2.erode(closed_mask, kernel, iterations=1)
        # Dilate to restore the main liquid body to its approximate original size
        cleaned_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # --- Step 3: Find and Select the Best Contour ---
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # --- MODIFIED: Find the TALLEST contour, not the largest by area ---
            tallest_contour = None
            max_height = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small contours that are clearly noise
                if h > max_height and h > 20: # Added a min height threshold
                    max_height = h
                    tallest_contour = contour

            # Ensure we found a valid contour
            if tallest_contour is not None:
                # Draw the selected contour on our detection image
                cv2.drawContours(image_with_detections, [tallest_contour], -1, (0, 255, 0), 2)

                # --- Step 4: Identify the Meniscus Reading Point (using the tallest contour) ---
                x, y, w, h = cv2.boundingRect(tallest_contour)
                
                meniscus_points = []
                # Scan each vertical line within the bounding box
                for col in range(x, x + w):
                    # Find all points of the contour that lie on this vertical line
                    col_points = tallest_contour[tallest_contour[:, :, 0] == col]
                    if col_points.size > 0:
                        # Find the highest point (minimum y) on this line
                        top_point_index = np.argmin(col_points[:, 1])
                        top_point = col_points[top_point_index]
                        meniscus_points.append(top_point)

                if meniscus_points:
                    # The reading point is the lowest point of the meniscus curve (max y)
                    meniscus_point_array = max(meniscus_points, key=lambda p: p[1])
                    
                    center_x, center_y = meniscus_point_array
                    
                    cv2.circle(image_with_detections, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(image_with_detections, f'Meniscus: ({center_x}, {center_y})', 
                                (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    print(f"Meniscus detected at: ({center_x}, {center_y})")
            else:
                 print("No sufficiently tall contours found.")

        else:
            print("No contours were found in the cleaned yellow mask.")
        
        # --- Step 5: Show the Images ---
        cv2.imshow("Cleaned Yellow Mask", cleaned_mask)
        cv2.imshow("Detected Meniscus", image_with_detections)

        print("Press any key to close the windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Using the specified image path for the cylinder
    image_file = 'images/cylinder.JPG'
    detect_meniscus(image_file)