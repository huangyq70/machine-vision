import cv2
import numpy as np

# isolate_curved_line function remains the same as before...
def isolate_curved_line(image, lower_bound, upper_bound):
    """
    Takes an image of a shape (like a meniscus), finds the curved parts,
    and returns a binary image containing only the single longest curve.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    all_edges = cv2.Canny(blurred, lower_bound, upper_bound)
    
    straight_lines_mask = np.zeros_like(all_edges)
    lines = cv2.HoughLinesP(all_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(straight_lines_mask, (x1, y1), (x2, y2), 255, 3)

    curved_line = cv2.subtract(all_edges, straight_lines_mask)
    contours, _ = cv2.findContours(curved_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        longest_contour = max(contours, key=lambda c: cv2.arcLength(c, False))
        output_mask = np.zeros_like(all_edges)
        cv2.drawContours(output_mask, [longest_contour], -1, 255, 1)
        return output_mask
    else:
        return np.zeros_like(all_edges)

def detect_meniscus_by_shape(main_image_path, template_image_path):
    """
    Finds a meniscus in a larger image by matching the shape of its curved line.
    This version includes filtering to remove horizontal scale lines.
    """
    # --- Step 1: Load the images ---
    main_image = cv2.imread(main_image_path)
    template = cv2.imread(template_image_path)

    # Make images a manageable size
    main_image = cv2.resize(main_image, (0,0), fx=0.25, fy=0.25)
    template = cv2.resize(template, (0,0), fx=0.25, fy=0.25)

    if main_image is None or template is None:
        print("Error: Could not load one or both images.")
        return

    # --- Step 1.5: Isolate the Tube by Finding Vertical Walls ---
    # This section remains unchanged...
    print("Attempting to isolate the tube from the main image...")
    gray_for_lines = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    edges_for_lines = cv2.Canny(gray_for_lines, 50, 150)
    lines = cv2.HoughLinesP(edges_for_lines, rho=1, theta=np.pi/180, threshold=80, minLineLength=100, maxLineGap=20)
    vertical_lines_x = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if 70 < abs(angle) < 120:
                vertical_lines_x.append((x1 + x2) // 2)
    if len(vertical_lines_x) >= 2:
        unique_x = sorted(list(set(vertical_lines_x)))
        min_dist = float('inf')
        best_pair = None
        for i in range(len(unique_x) - 1):
            dist = unique_x[i+1] - unique_x[i]
            if dist < min_dist:
                min_dist = dist
                best_pair = (unique_x[i], unique_x[i+1])
        if best_pair:
            left_wall, right_wall = best_pair
            padding = 5
            if (right_wall - padding) > (left_wall + padding):
                crop_x1 = max(0, left_wall + padding)
                crop_x2 = min(main_image.shape[1], right_wall - padding)
                main_image = main_image[:, crop_x1:crop_x2]
                print(f"Tube walls found. Image cropped between x={crop_x1} and x={crop_x2}.")
                cv2.imshow("Cropped Tube", main_image)
            else:
                print("Warning: Detected tube walls are too close together to crop. Proceeding with uncropped image.")
    else:
        print("Warning: Could not robustly identify tube walls. Proceeding with uncropped image.")
    
    # --- Step 2: Isolate the template shape and get edges from main image ---
    print("Processing images to detect shape...")
    template_shape = isolate_curved_line(template, 60, 150)
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    main_blurred = cv2.GaussianBlur(main_gray, (5, 5), 0)
    main_edges = cv2.Canny(main_blurred, 50, 150)
    
    # --- NEW: Step 3: Filter Out Scale Lines from the Main Image Edges ---
    print("Filtering out horizontal lines from main image...")
    
    # Find all contours in the edge map
    contours, _ = cv2.findContours(main_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new blank mask to draw only the contours we want to keep
    filtered_edges = np.zeros_like(main_edges)

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Define what a "tick mark" looks like.
        # It's a rectangle that is very wide but not very tall.
        # You can adjust these values for your specific images.
        is_likely_tick_mark = (aspect_ratio > 3.0) and (h < 15)

        # If the contour is NOT a tick mark, we keep it
        if not is_likely_tick_mark:
            # Draw this "good" contour onto our blank mask
            cv2.drawContours(filtered_edges, [contour], -1, 255, 1)

    # --- Step 4: Perform Template Matching on the Filtered Edges ---
    template_h, template_w = template_shape.shape[:2]
    
    if filtered_edges.shape[0] < template_h or filtered_edges.shape[1] < template_w:
        print("Error: The main image is smaller than the template image after processing.")
        return

    # Use the 'filtered_edges' instead of the original 'main_edges'
    result = cv2.matchTemplate(filtered_edges, template_shape, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    print(f"Shape match confidence: {max_val:.2f}")

    detection_image = main_image.copy()

    # --- Step 5: Display the Results ---
    confidence_threshold = 0.2
    if max_val > confidence_threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        reading_point = (top_left[0] + template_w // 2, top_left[1] + template_h)
        
        cv2.rectangle(detection_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(detection_image, f'Reading: ({reading_point[0]}, {reading_point[1]})', (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"Meniscus reading point found at {reading_point}")
    else:
        print("No confident shape match found.")

    cv2.imshow("1. Original Main Image Edges", main_edges)
    cv2.imshow("2. Filtered Edges (Tick Marks Removed)", filtered_edges) # <-- New window to see the result
    cv2.imshow("3. Template Shape to Match", template_shape)
    cv2.imshow("4. Final Detection Result", detection_image)

    print("\nPress any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_file = 'images/Clear-Images/IMG_3444.jpg' 
    template_file = 'templates/template_5.jpg' 
    
    detect_meniscus_by_shape(image_file, template_file)
