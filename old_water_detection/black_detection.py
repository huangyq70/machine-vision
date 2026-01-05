import cv2
import numpy as np
import os

# --- Configuration ---
IMAGE_FOLDER = "./images/personal_images/"
IMAGE_PREFIX = "image_"
IMAGE_EXT = ".jpg"
TOTAL_IMAGES = 13
DISPLAY_WIDTH = 600
DEFAULT_THRESHOLD = 30  # Adjusted based on your image brightness

def nothing(x):
    pass

def resize_to_width(image, target_width):
    h, w = image.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h))

def find_closest_black_walls(mask):
    """
    Scans the mask for BLACK pixels (value 0).
    Finds the row where the horizontal gap between the 
    Left Black Wall and Right Black Wall is SMALLEST.
    """
    h, w = mask.shape
    center_x = w // 2
    
    # Scan width: Middle 50% of the image
    scan_width = int(w * 0.5) 
    start_x = center_x - (scan_width // 2)
    end_x = center_x + (scan_width // 2)

    min_gap = float('inf')
    # Default to a safe spot (1/3 from bottom) if nothing found
    best_y = int(h * 0.66) 

    # Skip top/bottom 10% to avoid edge artifacts/labels
    margin_y = int(h * 0.1)
    
    for y in range(margin_y, h - margin_y):
        # Extract row slice
        row_slice = mask[y, start_x:end_x]
        
        # FIND BLACK PIXELS (Value 0)
        # Note: In standard thresholding of a bright image, dark objects are 0.
        black_indices = np.where(row_slice == 0)[0]
        
        if len(black_indices) < 2:
            continue
            
        # Define Center relative to the slice
        slice_center = (end_x - start_x) // 2
        
        # Split into Left Wall and Right Wall
        left_wall = black_indices[black_indices < slice_center]
        right_wall = black_indices[black_indices > slice_center]
        
        if len(left_wall) > 0 and len(right_wall) > 0:
            # Inner Edge of Left Wall (Rightmost pixel of the left blob)
            l_edge = left_wall[-1]
            # Inner Edge of Right Wall (Leftmost pixel of the right blob)
            r_edge = right_wall[0]
            
            # Calculate the WHITE gap between the black walls
            gap = r_edge - l_edge
            
            # We want the smallest gap (closest walls)
            # Gap must be positive (otherwise they are touching/overlapping)
            if gap < min_gap:
                min_gap = gap
                best_y = y

    return best_y

def main():
    print("Looking for images...")
    images = []
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")

    for i in range(1, TOTAL_IMAGES + 1):
        path = os.path.join(IMAGE_FOLDER, f"{IMAGE_PREFIX}{i}{IMAGE_EXT}")
        if os.path.exists(path):
            images.append(cv2.imread(path))
            print(f"Loaded {path}")

    if not images:
        print("No images found.")
        return

    window_name = "Black Wall Pinch Detector"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Threshold", window_name, DEFAULT_THRESHOLD, 255, nothing)

    current_idx = 0

    while True:
        img = images[current_idx].copy()
        h, w = img.shape[:2]
        
        thresh_val = cv2.getTrackbarPos("Threshold", window_name)
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Thresholding
        # We use THRESH_BINARY. 
        # Bright Background -> 255 (White)
        # Dark Walls/Liquid -> 0 (Black)
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 3. Clean Noise (Morph Open)
        # Helps remove random black specs that aren't the wall
        kernel = np.ones((3,3), np.uint8)
        # Invert logic for Morph: We want to remove small BLACK dots from WHITE bg?
        # Standard morph works on white foreground. 
        # Let's simple erode/dilate to ensure walls are solid.
        clean_mask = cv2.erode(mask, kernel, iterations=1)
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)
        
        # 4. Find where Black Walls are closest
        meniscus_y = find_closest_black_walls(clean_mask)
        
        # --- VISUALIZATION ---
        result_vis = img.copy()
        mask_vis = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

        # Dynamic Graphics Scaling
        font_scale = w / 500.0 
        thickness = int(w / 200.0)
        if thickness < 2: thickness = 2
        line_thickness = int(h / 150)
        if line_thickness < 3: line_thickness = 3

        # Draw Guess Line
        cv2.line(result_vis, (0, meniscus_y), (w, meniscus_y), (0, 255, 0), line_thickness)
        
        # Label
        label = "GUESS"
        cv2.putText(result_vis, label, (20, meniscus_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4)
        cv2.putText(result_vis, label, (20, meniscus_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # Combine
        combined = np.hstack((mask_vis, result_vis))
        final_view = resize_to_width(combined, DISPLAY_WIDTH * 2)
        
        cv2.imshow(window_name, final_view)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            current_idx = (current_idx + 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()