import cv2
import numpy as np
import os

# --- Configuration ---
IMAGE_FOLDER = "./images/personal_images/"
IMAGE_PREFIX = "image_"
IMAGE_EXT = ".jpg"
TOTAL_IMAGES = 13
DISPLAY_MAX_WIDTH = 1000
DISPLAY_MAX_HEIGHT = 800

# --- SETTINGS ---
CENTER_TOLERANCE = 0.25 
NOISE_THRESHOLD = 10 
# Gamma < 1.0 makes image brighter (0.5 is very bright).
# Gamma > 1.0 makes image darker.
BRIGHTNESS_GAMMA = 0.6 

def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def apply_gamma(image, gamma=1.0):
    """Boosts brightness/contrast to help Canny see in the dark."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_holder_roi(image):
    """
    Finds the Holder Pillars using Darkness Projection.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Darkness Threshold
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    # 2. Vertical Projection
    col_sums = np.sum(dark_mask, axis=0)
    
    # Restrict Search to Center
    safe_w = w * CENTER_TOLERANCE
    safe_min = int((w - safe_w) / 2)
    safe_max = int((w + safe_w) / 2)
    col_sums[:safe_min] = 0
    col_sums[safe_max:] = 0

    max_val = np.max(col_sums)
    if max_val == 0: return 0, 0, 0, 0, False

    # 3. Find Pillars
    threshold = max_val * 0.3 
    strong_indices = np.where(col_sums > threshold)[0]
    
    if len(strong_indices) > 10:
        pillar_L = strong_indices[0]
        pillar_R = strong_indices[-1]
        holder_width = pillar_R - pillar_L
        
        if holder_width > 20 and holder_width < (w * 0.95):
            center = (pillar_L + pillar_R) // 2
            # Glass is middle 30% of holder
            glass_width = int(holder_width * 0.30)
            x1 = center - (glass_width // 2)
            x2 = center + (glass_width // 2)
            
            return x1, x2, pillar_L, pillar_R, True

    return 0, 0, 0, 0, False

def check_for_tube(image, x1, x2):
    """Checks for vertical edges inside the glass zone."""
    if x2 <= x1: return False
    roi = image[:, x1:x2]
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            angle = np.abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180 / np.pi)
            if angle > 80: return True
    return False

def find_cap_block(image, x1, x2):
    """
    Finds the Thread Cluster.
    """
    if x2 <= x1: return 0, 0
    
    h, w = image.shape[:2]
    roi = image[:, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Structure Detection
    edges = cv2.Canny(gray, 30, 100)
    
    # Find horizontal lines
    min_len = int((x2 - x1) * 0.3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=min_len, maxLineGap=5)
    
    horizontal_ys = []
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            angle = np.abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180 / np.pi)
            if angle < 10 or angle > 170: 
                horizontal_ys.append((ly1 + ly2) // 2)
    
    if not horizontal_ys: return 0, 0
        
    horizontal_ys.sort()
    
    # Cluster lines
    clusters = []
    current_cluster = [horizontal_ys[0]]
    for i in range(1, len(horizontal_ys)):
        if horizontal_ys[i] - horizontal_ys[i-1] < 15:
            current_cluster.append(horizontal_ys[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [horizontal_ys[i]]
    clusters.append(current_cluster)
    
    # Find the "Best" cluster
    best_cluster = []
    max_lines = 0
    
    for cluster in clusters:
        center_y = np.mean(cluster)
        if center_y < (h * 0.6):
            if len(cluster) > max_lines:
                max_lines = len(cluster)
                best_cluster = cluster
            
    if best_cluster:
        return min(best_cluster), max(best_cluster)
    
    return 0, 0

def detect_liquid_level(image, x1, x2, start_search_y):
    """
    Finds the strongest horizontal line INSIDE the ROI (x1:x2).
    """
    if x2 <= x1: return -1, 0, np.zeros_like(image[:, 0])

    h, w = image.shape[:2]
    
    roi = image[:, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(gray_roi, 30, 100)

    row_sums = np.sum(edges, axis=1)
    
    # --- MASKING ---
    safety_buffer = 10
    mask_limit = start_search_y + safety_buffer
    
    if mask_limit < h:
        row_sums[:mask_limit] = 0
    else:
        return -1, 0, row_sums 
        
    border_bottom = int(h * 0.05)
    row_sums[-border_bottom:] = 0
    
    best_y = np.argmax(row_sums)
    max_val = row_sums[best_y]
    
    roi_w = x2 - x1
    score = max_val / roi_w if roi_w > 0 else 0

    return best_y, score, row_sums

def detect_tube_top(image, x1, x2):
    """Finds the Rim."""
    if x2 <= x1: return 0
    h, w = image.shape[:2]
    roi = image[:, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobel_y)
    row_sums = np.sum(abs_sobel, axis=1)
    
    max_val = np.max(row_sums)
    if max_val == 0: return 0
    
    threshold = max_val * 0.5
    candidate_indices = np.where(row_sums > threshold)[0]
    
    if len(candidate_indices) == 0: return 0
    
    center_y = h // 2
    best_y = candidate_indices[0]
    min_dist = abs(best_y - center_y)
    
    for y in candidate_indices:
        dist = abs(y - center_y)
        if dist < min_dist:
            min_dist = dist
            best_y = y
            
    return best_y

def main():
    print("Looking for images...")
    images = []
    for i in range(1, TOTAL_IMAGES + 1):
        path = os.path.join(IMAGE_FOLDER, f"{IMAGE_PREFIX}{i}{IMAGE_EXT}")
        if os.path.exists(path):
            # --- NEW: Load and immediately brighten ---
            raw_img = cv2.imread(path)
            if raw_img is not None:
                bright_img = apply_gamma(raw_img, gamma=BRIGHTNESS_GAMMA)
                images.append(bright_img)
                print(f"Loaded {path} (Brightened)")

    if not images:
        print("No images found.")
        return

    idx = 0
    cv2.namedWindow("Final Water Level")
    print("Press SPACE for next image. 'q' to quit.")

    while True:
        img = images[idx]
        h, w = img.shape[:2]
        vis = img.copy()
        # --- REMOVED THE DIMMING LINE ---
        # vis = cv2.addWeighted(...) <--- GONE

        # 1. Find Holder
        gx1, gx2, pL, pR, holder_found = get_holder_roi(img)

        if holder_found:
            # 2. Check for Tube
            if check_for_tube(img, gx1, gx2):
                
                # 3. Detect Features
                rim_y = detect_tube_top(img, gx1, gx2)
                cap_top, cap_bottom = find_cap_block(img, gx1, gx2)
                
                block_bottom = cap_bottom if cap_bottom > 0 else rim_y
                block_top = rim_y if (rim_y > 0 and rim_y < block_bottom) else cap_top
                if block_top == 0: block_top = block_bottom - 20 

                # 4. Detect Level
                level_y, score, profile = detect_liquid_level(img, gx1, gx2, start_search_y=block_bottom)
                
                # --- VISUALIZATION ---
                
                dark_blue = (139, 0, 0)
                thickness_box = 4
                cv2.line(vis, (gx1, 0), (gx1, h), dark_blue, thickness_box)
                cv2.line(vis, (gx2, 0), (gx2, h), dark_blue, thickness_box)
                cv2.putText(vis, "Tube Guess", (gx1 - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_blue, 2)

                if block_bottom > 0:
                    overlay = vis.copy()
                    y1_red = max(0, block_top)
                    y2_red = min(h, block_bottom)
                    if y2_red > y1_red:
                        cv2.rectangle(overlay, (gx1, y1_red), (gx2, y2_red), (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
                        cv2.line(vis, (gx1, y2_red), (gx2, y2_red), (0, 0, 255), 2)
                        cv2.putText(vis, "IGNORED", (gx1 - 65, (y1_red+y2_red)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                center_x = (gx1 + gx2) // 2
                cv2.line(vis, (center_x, 0), (center_x, h), (255, 255, 0), 2)

                if np.max(profile) > 0:
                    roi_w = gx2 - gx1
                    graph_offset = gx2 + 10
                    norm_profile = (profile / np.max(profile)) * (roi_w * 2)
                    points = []
                    for y in range(h):
                        x_plot = int(graph_offset + norm_profile[y])
                        x_plot = min(x_plot, w - 1)
                        points.append((x_plot, y))
                    cv2.polylines(vis, [np.array(points)], False, (0, 255, 255), 1)

                if score > NOISE_THRESHOLD:
                    cv2.line(vis, (gx1 - 20, level_y), (gx2 + 20, level_y), (0, 255, 0), 4)
                    cv2.putText(vis, f"WATER {level_y}", (gx2 + 20, level_y + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "Signal Weak", (gx1, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            else:
                cv2.rectangle(vis, (gx1, 0), (gx2, h), (0, 0, 255), 2)
                cv2.putText(vis, "NO TUBE", (gx1 - 30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            vis[:] = (0, 0, 255)
            cv2.putText(vis, "NO HOLDER DETECTED", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        final_view = resize_to_fit(vis, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
        cv2.imshow("Final Water Level", final_view)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        if key == ord(' '):
            idx = (idx + 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()