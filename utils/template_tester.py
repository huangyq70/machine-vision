import cv2
import numpy as np
import os

# --- Configuration ---
IMAGE_FOLDER = "./images/personal_images/"
IMAGE_PREFIX = "image_"
IMAGE_EXT = ".jpg"
TOTAL_IMAGES = 10  # Check up to 10 images
TEMPLATE_FILENAME = "./templates/tube_holder_template.jpg" # <--- YOU MUST CREATE THIS FILE

def main():
    # 1. Load Template
    if not os.path.exists(TEMPLATE_FILENAME):
        print(f"Error: Could not find '{TEMPLATE_FILENAME}'.")
        print("Please run 'create_template.py' first to generate it.")
        return

    template = cv2.imread(TEMPLATE_FILENAME)
    if template is None:
        print("Error: Failed to load template image.")
        return

    # Convert template to grayscale for matching
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    t_h, t_w = template_gray.shape[:2]

    print(f"Loaded template ({t_w}x{t_h}).")
    print("Press SPACE to check next image. Press 'q' to quit.")

    # 2. Loop through images
    cv2.namedWindow("Template Match Tester", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Template Match Tester", 800, 600)

    found_images = False

    for i in range(1, TOTAL_IMAGES + 1):
        filename = f"{IMAGE_PREFIX}{i}{IMAGE_EXT}"
        filepath = os.path.join(IMAGE_FOLDER, filename)
        
        if not os.path.exists(filepath):
            continue
        
        found_images = True
        img = cv2.imread(filepath)
        if img is None:
            print(f"Could not read {filename}")
            continue

        # Convert target image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- MATCHING ---
        # TM_CCOEFF_NORMED is best for lighting changes.
        # Result is a map of scores from -1 to 1.
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For TM_CCOEFF_NORMED, max_loc is the best match
        top_left = max_loc
        bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
        
        # --- VISUALIZATION ---
        # Determine color based on confidence
        # > 0.8 is excellent, < 0.5 is probably wrong
        if max_val > 0.8:
            color = (0, 255, 0) # Green (Good)
        elif max_val > 0.5:
            color = (0, 255, 255) # Yellow (Okay)
        else:
            color = (0, 0, 255) # Red (Bad match)

        cv2.rectangle(img, top_left, bottom_right, color, 3)
        
        label = f"Match: {max_val:.2f}"
        cv2.putText(img, label, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(img, f"Checking: {filename}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Template Match Tester", img)
        
        print(f"{filename}: Score {max_val:.2f}")

        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    if not found_images:
        print("No images found to test!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()