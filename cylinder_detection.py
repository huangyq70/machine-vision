import cv2
import numpy as np

# Global variable to hold the image for easy access by the trackbar callback
img_global = None

def find_and_show_curved_lines(image, canny_t1, canny_t2, hough_thresh, hough_min_len, hough_max_gap):
    """
    Identifies and displays only the curved lines from an input image using tunable parameters.
    """
    # --- 1. Pre-process the Image ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. Find All Edges using parameters from trackbars ---
    all_edges = cv2.Canny(blurred, canny_t1, canny_t2)

    # --- 3. Find and Mask the Straight Lines ---
    straight_lines_mask = np.zeros_like(all_edges) 
    
    # Use the Probabilistic Hough Line Transform with parameters from trackbars
    lines = cv2.HoughLinesP(
        all_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=hough_min_len,
        maxLineGap=hough_max_gap
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(straight_lines_mask, (x1, y1), (x2, y2), 255, 2)

    # --- 4. Isolate the Curved Lines ---
    curved_lines = cv2.subtract(all_edges, straight_lines_mask)

    # --- 5. Display the Results in persistent windows ---
    cv2.imshow('All Edges (Canny)', all_edges)
    cv2.imshow('Detected Straight Lines', straight_lines_mask)
    cv2.imshow('Detected Curved Lines Only', curved_lines)

def on_trackbar_change(val):
    """
    Callback function that is triggered when a trackbar value changes.
    It reads all trackbar values and re-runs the detection.
    """
    # Defensive check for the global image
    if img_global is None:
        return

    # Get current positions of all trackbars
    canny_t1 = cv2.getTrackbarPos('Canny Thresh 1', 'Parameters')
    canny_t2 = cv2.getTrackbarPos('Canny Thresh 2', 'Parameters')
    hough_thresh = cv2.getTrackbarPos('Hough Threshold', 'Parameters')
    # Ensure minLineLength is at least 1 to prevent errors
    hough_min_len = max(1, cv2.getTrackbarPos('Hough Min Length', 'Parameters'))
    hough_max_gap = cv2.getTrackbarPos('Hough Max Gap', 'Parameters')

    # Make a copy of the original image to work on
    image_copy = img_global.copy()
    find_and_show_curved_lines(image_copy, canny_t1, canny_t2, hough_thresh, hough_min_len, hough_max_gap)


if __name__ == '__main__':
    # --- Load an Image from a File Path ---
    # We will use the meniscus image you provided.
    image_path = 'templates/template_5.jpg'
    
    img_global = cv2.imread(image_path)
    img_global = cv2.resize(img_global, (0, 0), fx=0.25, fy=0.25)

    if img_global is not None:
        # Create windows that will persist
        cv2.namedWindow('Original Image')
        cv2.namedWindow('Detected Curved Lines Only')
        cv2.namedWindow('Parameters')
        
        # --- Create Trackbars for Real-Time Parameter Tuning ---
        # Canny Edge Detector thresholds
        cv2.createTrackbar('Canny Thresh 1', 'Parameters', 50, 255, on_trackbar_change)
        cv2.createTrackbar('Canny Thresh 2', 'Parameters', 150, 255, on_trackbar_change)
        # Hough Line Transform parameters
        cv2.createTrackbar('Hough Threshold', 'Parameters', 50, 255, on_trackbar_change)
        cv2.createTrackbar('Hough Min Length', 'Parameters', 50, 255, on_trackbar_change)
        cv2.createTrackbar('Hough Max Gap', 'Parameters', 10, 100, on_trackbar_change)

        # Display the original image
        cv2.imshow('Original Image', img_global)
        
        # Trigger the initial detection with default trackbar values
        on_trackbar_change(0)

        print("Tuning Instructions:")
        print("1. Adjust the sliders in the 'Parameters' window.")
        print("2. 'Canny Thresh' sliders control the initial edge detection.")
        print("3. 'Hough' sliders control what is considered a straight line.")
        print("4. Press any key in an image window to exit.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load image from path: {image_path}")
        print("Please make sure the file exists and is in the same directory.")