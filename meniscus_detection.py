import cv2
import numpy as np

def detect_meniscus_with_template(main_image_path, template_image_path):
    """
    Finds a meniscus in a larger image by matching a template image.

    This method is robust to low contrast and noise, as it looks for a
    pattern rather than relying on sharp edges.

    Args:
        main_image_path (str): Path to the image of the cylinder.
        template_image_path (str): Path to the small, cropped template image
                                     of the meniscus.
    """
    # --- Step 1: Load the images ---
    main_image = cv2.imread(main_image_path)
    template = cv2.imread(template_image_path)

    if main_image is None:
        print(f"Error: Could not load main image from {main_image_path}")
        return
    if template is None:
        print(f"Error: Could not load template image from {template_image_path}")
        print("Please ensure you have created a 'meniscus_template.png' file.")
        return

    detection_image = main_image.copy()
    
    # Convert images to grayscale for matching
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Get the width and height of the template
    template_h, template_w = template_gray.shape[:2]
    main_h, main_w = main_gray.shape[:2]

    # --- NEW: Add a check to prevent the OpenCV error ---
    # The main image must be larger than the template to perform matching.
    if main_h < template_h or main_w < template_w:
        print("Error: The main image is smaller than the template image.")
        print(f"Main image dims: {main_w}x{main_h}, Template dims: {template_w}x{template_h}")
        print("Please use a larger main image or create a smaller template from the current main image.")
        return

    # --- Step 2: Perform Template Matching ---
    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    print(f"Template match confidence: {max_val:.2f}")

    # Set a confidence threshold
    if max_val > 0.8: # You can adjust this threshold (0.0 to 1.0)
        top_left = max_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        # --- Step 3: Identify the Reading Point ---
        # The reading is at the bottom-center of the matched template area
        reading_point = (top_left[0] + template_w // 2, top_left[1] + template_h)
        
        # Draw a rectangle around the detected area
        cv2.rectangle(detection_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Draw a prominent circle at the reading point
        cv2.circle(detection_image, reading_point, 7, (0, 0, 255), -1)
        cv2.putText(detection_image, f'Reading: {reading_point}',
                    (reading_point[0] + 10, reading_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"Meniscus reading point found at {reading_point}")

    else:
        print("No confident match found. The meniscus might not be visible or the template is a poor match.")


    # --- Step 4: Display the Result ---
    cv2.imshow("Detection Image", detection_image)

    print("Press any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You MUST create this template file yourself by cropping a good example
    template_file = 'templates/template_3.jpg' 
    image_file = 'images/cylinder.JPG'
    detect_meniscus_with_template(image_file, template_file)
