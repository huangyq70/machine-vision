import cv2
import numpy as np
import math

def show_all_lines(image_path):
    """
    Loads an image, finds all lines using HoughLinesP, and displays them.

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

        image_with_lines = original_image.copy()
        
        # --- Step 2: Preprocessing ---
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
        
        # --- Step 3: Edge Detection ---
        # Use Canny edge detection on the blurred image
        edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)

        # --- Step 4: Detect and Draw All Lines ---
        # Note: minLineLength and maxLineGap are set to arbitrary values
        # You may need to tune these!
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Draw all detected lines in red
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(f"Found and drew {len(lines)} lines.")
        else:
            print("No lines were found with the current settings.")

        # --- Step 5: Show the Images ---
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Canny Edges', edges)
        cv2.imshow('All Detected Lines', image_with_lines)

        print("Press any key to close the windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # ** USER: Update this path to your image file **
    image_file = 'images/IMG_7750.JPG' 
    show_all_lines(image_file)
