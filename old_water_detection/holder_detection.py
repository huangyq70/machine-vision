import cv2
import numpy as np
import os

# --- Configuration ---
IMAGE_FOLDER = "./images/personal_images/"
IMAGE_PREFIX = "image_"
IMAGE_EXT = ".jpg"
TOTAL_IMAGES = 14
DISPLAY_MAX_WIDTH = 2000
DISPLAY_MAX_HEIGHT = 1800

# --- ALIGNMENT SETTINGS ---
CENTER_TOLERANCE = 0.25 

def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def find_holder_structure(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 1. DARKNESS THRESHOLD (Find Black Objects) ---
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    # --- 2. VERTICAL PROJECTION (Sum Columns) ---
    col_sums = np.sum(dark_mask, axis=0)
    
    # Restrict Search to Safe Zone
    safe_zone_width = w * CENTER_TOLERANCE
    safe_min = int((w - safe_zone_width) / 2)
    safe_max = int((w + safe_zone_width) / 2)
    
    col_sums[:safe_min] = 0
    col_sums[safe_max:] = 0

    max_val = np.max(col_sums)
    graph_norm = np.zeros_like(col_sums)
    if max_val > 0:
        graph_norm = (col_sums / max_val) * (h * 0.8)

    # --- 3. FIND PEAKS (The Pillars) ---
    threshold = max_val * 0.3 
    strong_indices = np.where(col_sums > threshold)[0]
    
    holder_left = 0
    holder_right = w
    glass_x1 = 0
    glass_x2 = 0
    found = False
    is_centered = False

    # --- VISUALIZATION SETUP ---
    view_original = image.copy()
    view_mask = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
    
    # Base result view (Darkened)
    view_result = image.copy()
    view_result = cv2.addWeighted(view_result, 0.3, np.zeros_like(view_result), 0.7, 0)

    # 4. LOGIC CHECK
    if len(strong_indices) > 10:
        holder_left = strong_indices[0]
        holder_right = strong_indices[-1]
        
        holder_width = holder_right - holder_left
        # Sanity check: Holder shouldn't be tiny or the entire screen
        if holder_width > 20 and holder_width < (w * 0.95):
            found = True
            
            # Calculate Center
            center = (holder_left + holder_right) // 2
            
            # Define "Guess" Box (Middle 30% of holder)
            glass_width = int(holder_width * 0.30)
            glass_x1 = center - (glass_width // 2)
            glass_x2 = center + (glass_width // 2)

            # Check Center Alignment
            if safe_min <= center <= safe_max:
                is_centered = True
                status_text = "Holder Locked"
                status_color = (0, 255, 0)
            else:
                is_centered = False
                status_text = "OFF CENTER"
                status_color = (0, 0, 255)

            # --- DRAWING ---
            
            # Draw Safe Zone (Gray)
            cv2.line(view_result, (safe_min, 0), (safe_min, h), (100, 100, 100), 1)
            cv2.line(view_result, (safe_max, 0), (safe_max, h), (100, 100, 100), 1)

            # Draw Pillars (Blue)
            cv2.line(view_result, (holder_left, 0), (holder_left, h), (255, 0, 0), 3)
            cv2.line(view_result, (holder_right, 0), (holder_right, h), (255, 0, 0), 3)
            
            # Draw Center Line (Magenta)
            cv2.line(view_result, (center, 0), (center, h), (255, 0, 255), 2)

            # Draw Graph (Yellow)
            points = []
            for x in range(w):
                y = int(h - graph_norm[x])
                points.append((x, y))
            cv2.polylines(view_result, [np.array(points)], False, (0, 255, 255), 1)
            
            # Draw Guess Box (Green if found)
            cv2.rectangle(view_result, (glass_x1, 0), (glass_x2, h), (0, 255, 0), 2)
            
            # --- DETECT AND DRAW HORIZONTAL LINES IN RED ---
            if glass_x2 > glass_x1:
                roi = image[:, glass_x1:glass_x2]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Edge Detection on ROI
                edges = cv2.Canny(roi_gray, 50, 150)
                
                # Find lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, 
                                      minLineLength=glass_width*0.4, maxLineGap=5)
                
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Calculate angle to ensure it's horizontal
                        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                        if angle < 20 or angle > 160:
                            # Draw red line adjusted to main image coordinates
                            cv2.line(view_result, (glass_x1 + x1, y1), (glass_x1 + x2, y2), (0, 0, 255), 1)

            cv2.putText(view_result, status_text, (glass_x1 - 40, h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # --- IF NOT FOUND: RED SCREEN ---
    if not found:
        view_result[:] = (0, 0, 255) 
        status_text = "NO HOLDER DETECTED"
        cv2.putText(view_result, status_text, (20, h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(view_result, "Ensure holder is in center", (20, h // 2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Labels for Grid
    cv2.putText(view_original, "1. Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(view_mask, "2. Black Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(view_result, "3. Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- COMPOSITE GRID ---
    grid = np.hstack((view_original, view_mask, view_result))

    return grid

def main():
    print("Looking for images...")
    images = []
    for i in range(1, TOTAL_IMAGES + 1):
        path = os.path.join(IMAGE_FOLDER, f"{IMAGE_PREFIX}{i}{IMAGE_EXT}")
        if os.path.exists(path):
            images.append(cv2.imread(path))
            print(f"Loaded {path}")

    if not images:
        print("No images found.")
        return

    idx = 0
    cv2.namedWindow("Pipeline View")
    
    print("Press SPACE for next image. 'q' to quit.")

    while True:
        result_grid = find_holder_structure(images[idx])
        resized_grid = resize_to_fit(result_grid, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
        
        cv2.imshow("Pipeline View", resized_grid)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        if key == ord(' '):
            idx = (idx + 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()