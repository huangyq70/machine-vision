import cv2
import numpy as np
import os

# --- Configuration ---
IMAGE_FOLDER = "./images/april_tag/"
IMAGE_PREFIX = "image_"
IMAGE_EXT = ".jpg"
TOTAL_IMAGES = 14
DISPLAY_MAX_WIDTH = 1500  
DISPLAY_MAX_HEIGHT = 900
TUBE_CAPACITY = 15.0  # The total capacity of the tube in some unit (e.g., ml)

# --- ALIGNMENT SETTINGS ---
CENTER_TOLERANCE = 0.25 
DEFAULT_MENISCUS_THRESHOLD = 28
ROI_WIDTH_PERCENTAGE = 0.40 # Increase this (e.g., 0.50) to widen the yellow gap

def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def find_pinch_point(mask_roi):
    """
    Finds the narrowest point between walls (The Meniscus Pinch).
    """
    h, w = mask_roi.shape
    min_gap = float('inf')
    best_y = -1 
    margin_y = int(h * 0.1)
    
    for y in range(margin_y, h - margin_y):
        row_slice = mask_roi[y, :]
        black_indices = np.where(row_slice == 0)[0]
        
        if len(black_indices) < 2:
            continue
            
        slice_center = w // 2
        left_wall = black_indices[black_indices < slice_center]
        right_wall = black_indices[black_indices > slice_center]
        
        if len(left_wall) > 0 and len(right_wall) > 0:
            l_edge = left_wall[-1]
            r_edge = right_wall[0]
            gap = r_edge - l_edge
            
            # We want the smallest gap 
            if gap > 0 and gap < min_gap:
                min_gap = gap
                best_y = y

    return best_y

def find_tube_top(holder_slice_mask):
    """Scan vertical column sum to find the first horizontal line (Top Cap)"""
    row_sums = np.sum(holder_slice_mask, axis=1)
    # Threshold: At least 5 pixels of pillar must exist to count as 'Top'
    top_indices = np.where(row_sums > (255 * 5))[0]
    if len(top_indices) > 0:
        return top_indices[0]
    return None

def find_tube_bottom(roi_clean):
    """Scan ROI center strip to find where the gap closes (Bottom)"""
    h, w = roi_clean.shape
    if w == 0: return None
    
    mid_s = int(w * 0.4)
    mid_e = int(w * 0.6)
    center_strip = roi_clean[:, mid_s:mid_e]
    
    # If row average < 100, it's mostly black (structure)
    strip_means = np.mean(center_strip, axis=1)
    bridge_indices = np.where(strip_means < 100)[0]
    
    if len(bridge_indices) > 0:
        return bridge_indices[0]
    return None

def cluster_votes_complex(votes, tolerance=10):
    """
    Groups votes based on Meniscus Y proximity.
    Returns averages for (Meniscus, Top, Bottom) from the winning cluster.
    votes format: [(m_y, t_y, b_y), ...]
    """
    if not votes:
        return -1, None, None, 0
        
    # Sort by Meniscus Y (index 0)
    votes.sort(key=lambda x: x[0])
    
    clusters = []
    current_cluster = [votes[0]]
    
    for i in range(1, len(votes)):
        # Compare Meniscus Y of current vote vs previous vote
        if votes[i][0] - votes[i-1][0] <= tolerance:
            current_cluster.append(votes[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [votes[i]]
    clusters.append(current_cluster)
    
    # Find the cluster with the most votes
    best_cluster = max(clusters, key=len)
    
    # Calculate average Meniscus Y
    meniscus_vals = [v[0] for v in best_cluster]
    avg_m = int(sum(meniscus_vals) / len(meniscus_vals))
    
    # Calculate average Top Y (filter out Nones)
    top_vals = [v[1] for v in best_cluster if v[1] is not None]
    avg_t = int(sum(top_vals) / len(top_vals)) if top_vals else None
    
    # Calculate average Bottom Y (filter out Nones)
    bot_vals = [v[2] for v in best_cluster if v[2] is not None]
    avg_b = int(sum(bot_vals) / len(bot_vals)) if bot_vals else None
    
    return avg_m, avg_t, avg_b, len(best_cluster)

def process_image(image, meniscus_threshold, fixed_bottom=None, auto_thresh=False):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # PART 1: COLUMN DETECTION (Global)
    # ---------------------------------------------------------
    # We still use a global mask for finding the X-coordinates of the holder
    _, dark_mask_global = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    dark_mask_global = cv2.morphologyEx(dark_mask_global, cv2.MORPH_OPEN, kernel)

    col_sums = np.sum(dark_mask_global, axis=0)
    
    safe_zone_width = w * CENTER_TOLERANCE
    safe_min = int((w - safe_zone_width) / 2)
    safe_max = int((w + safe_zone_width) / 2)
    col_sums[:safe_min] = 0
    col_sums[safe_max:] = 0

    max_val = np.max(col_sums)
    threshold = max_val * 0.3 
    strong_indices = np.where(col_sums > threshold)[0]
    
    holder_left = 0
    holder_right = w
    glass_x1 = 0
    glass_x2 = 0
    found = False
    
    final_bottom_y = fixed_bottom
    tube_top_y = None
    
    # Status variables
    meniscus_y = -1
    vote_confidence = 0
    candidates_to_draw = []
    scan_range_text = ""

    # --- VISUALIZATION PREP ---
    view_original = image.copy()
    view_mask = cv2.cvtColor(dark_mask_global, cv2.COLOR_GRAY2BGR)
    view_mask[np.where((view_mask==[255,255,255]).all(axis=2))] = [100, 100, 100] 
    view_result = image.copy()
    view_result = cv2.addWeighted(view_result, 0.3, np.zeros_like(view_result), 0.7, 0) 

    # Dynamic scaling 
    draw_thickness = max(2, int(w / 300))
    line_thickness = max(2, int(w / 150))
    font_scale = max(0.6, w / 1000)

    if len(strong_indices) > 10:
        holder_left = strong_indices[0]
        holder_right = strong_indices[-1]
        holder_width = holder_right - holder_left

        if holder_width > 20 and holder_width < (w * 0.95):
            found = True
            center = (holder_left + holder_right) // 2
            
            glass_width = int(holder_width * ROI_WIDTH_PERCENTAGE)
            glass_x1 = center - (glass_width // 2)
            glass_x2 = center + (glass_width // 2)

            # Draw Pillars
            cv2.line(view_result, (holder_left, 0), (holder_left, h), (255, 0, 0), line_thickness)
            cv2.line(view_result, (holder_right, 0), (holder_right, h), (255, 0, 0), line_thickness)
            cv2.rectangle(view_mask, (glass_x1, 0), (glass_x2, h), (0, 255, 255), max(1, draw_thickness//2))

            # ---------------------------------------------------------
            # PART 2: ROI ANALYSIS
            # ---------------------------------------------------------
            if glass_x2 > glass_x1:
                roi_gray = gray[:, glass_x1:glass_x2]
                
                # --- DETECTION LOGIC (AUTO VS MANUAL) ---
                
                if auto_thresh:
                    # === AUTO: DARKEST PEAK (NO OFFSET) ===
                    try:
                        h_roi, w_roi = roi_gray.shape
                        # 1. Focus on bottom 60% (Liquid Zone)
                        bottom_half_roi = roi_gray[int(h_roi * 0.4):, :]
                        
                        if bottom_half_roi.size > 0:
                            # 2. Histogram Analysis
                            # Calculate histogram of the bottom area
                            hist = cv2.calcHist([bottom_half_roi], [0], None, [256], [0, 256])
                            
                            # 3. Find Dark Peak
                            # Look only in the "dark" range (0-100) for the highest peak
                            # This ignores the bright air/glass peaks completely
                            search_limit = 100
                            dark_region = hist[:search_limit]
                            peak_val = np.argmax(dark_region)
                            
                            # 4. Calculate Threshold
                            # Removed offset as requested
                            center_t = int(peak_val)
                            
                            # Safety Clamp
                            center_t = max(15, min(center_t, 90))
                        else:
                            center_t = meniscus_threshold
                    except Exception:
                        center_t = meniscus_threshold 
                    
                    # 2. Define Scan Range (Center +/- 15)
                    scan_min = max(5, center_t - 15)
                    scan_max = min(250, center_t + 15)
                    thresholds = range(scan_min, scan_max, 2) 
                    
                    scan_range_text = f"Scan: {scan_min}-{scan_max} (Peak: {center_t})"
                    votes = []
                    
                    for t in thresholds:
                        # ROI Threshold
                        _, t_thresh = cv2.threshold(roi_gray, t, 255, cv2.THRESH_BINARY)
                        t_clean = cv2.erode(t_thresh, np.ones((3,3), np.uint8), iterations=1)
                        t_clean = cv2.dilate(t_clean, np.ones((3,3), np.uint8), iterations=1)
                        
                        # Holder Threshold (Context aware top detection)
                        holder_slice = gray[:, holder_left:holder_right]
                        _, t_holder_mask = cv2.threshold(holder_slice, t, 255, cv2.THRESH_BINARY_INV)
                        
                        m_cand = find_pinch_point(t_clean)
                        
                        if m_cand != -1:
                            t_cand = find_tube_top(t_holder_mask)
                            b_cand = find_tube_bottom(t_clean)
                            votes.append((m_cand, t_cand, b_cand))
                            candidates_to_draw.append(m_cand) 
                            
                    # 3. Consensus (Gap-Based Clustering)
                    meniscus_y, voted_top, voted_bottom, vote_confidence = cluster_votes_complex(votes)
                    
                    # Apply voted values
                    if voted_top is not None: tube_top_y = voted_top
                    if voted_bottom is not None and final_bottom_y is None: final_bottom_y = voted_bottom
                    
                    status_text = f"AUTO: {vote_confidence} Votes | {scan_range_text}"
                    
                    # Visualization dummy
                    _, roi_thresh = cv2.threshold(roi_gray, center_t, 255, cv2.THRESH_BINARY)
                    
                else:
                    # === MANUAL: SINGLE SLIDER ===
                    # 1. Meniscus
                    _, roi_thresh = cv2.threshold(roi_gray, meniscus_threshold, 255, cv2.THRESH_BINARY)
                    roi_clean = cv2.erode(roi_thresh, np.ones((3,3), np.uint8), iterations=1)
                    roi_clean = cv2.dilate(roi_clean, np.ones((3,3), np.uint8), iterations=1)
                    meniscus_y = find_pinch_point(roi_clean)
                    
                    # 2. Top (Standard Logic)
                    holder_slice = dark_mask_global[:, holder_left:holder_right]
                    tube_top_y = find_tube_top(holder_slice)
                    
                    # 3. Bottom (Standard Logic)
                    if final_bottom_y is None:
                        # Use a slightly more permissive threshold for manual bottom detection
                        _, bottom_thresh = cv2.threshold(roi_gray, 60, 255, cv2.THRESH_BINARY)
                        roi_clean_b = cv2.erode(bottom_thresh, np.ones((3,3), np.uint8), iterations=1)
                        final_bottom_y = find_tube_bottom(roi_clean_b)

                    status_text = f"MANUAL: {meniscus_threshold}"

                # Update Mask Visualization (Panel 2)
                if not auto_thresh:
                    roi_vis = cv2.bitwise_not(roi_clean)
                    structure_locs = np.where(roi_vis > 0)
                    view_mask[structure_locs[0], structure_locs[1] + glass_x1] = (255, 255, 0)

                # ---------------------------------------------------------
                # VISUALIZE RESULTS
                # ---------------------------------------------------------
                
                # 1. Draw "Ghost" lines for meniscus candidates
                if auto_thresh:
                    for cand_y in candidates_to_draw:
                        cv2.line(view_result, (glass_x1, cand_y), (glass_x2, cand_y), (150, 150, 150), 1)

                # 2. Draw Final Meniscus (Green)
                if meniscus_y != -1:
                    cv2.line(view_result, (holder_left, meniscus_y), (holder_right, meniscus_y), (0, 255, 0), line_thickness)
                    cv2.putText(view_result, f"Lev: {meniscus_y}", (glass_x1 + 10, meniscus_y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), draw_thickness)

                # 3. Draw Tube Top
                if tube_top_y is not None:
                    cv2.line(view_result, (holder_left, tube_top_y), (holder_right, tube_top_y), (255, 0, 255), line_thickness)
                    cv2.putText(view_result, "TOP", (holder_left - int(80*font_scale), tube_top_y + int(30*font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), draw_thickness)

                # 4. Draw Tube Bottom
                if final_bottom_y is not None:
                    cv2.line(view_result, (holder_left, final_bottom_y), (holder_right, final_bottom_y), (255, 0, 255), line_thickness)
                    cv2.putText(view_result, "BOT", (holder_left - int(80*font_scale), final_bottom_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), draw_thickness)

                # ---------------------------------------------------------
                # D. Calculate Percentage and Volume
                # ---------------------------------------------------------
                if meniscus_y != -1 and tube_top_y is not None and final_bottom_y is not None:
                    total_pixels = final_bottom_y - tube_top_y
                    liquid_pixels = final_bottom_y - meniscus_y
                    
                    if total_pixels > 0:
                        percentage = liquid_pixels / total_pixels
                        percentage = max(0.0, min(1.0, percentage))
                        volume = percentage * TUBE_CAPACITY
                        
                        vol_text = f"{volume:.1f} / {TUBE_CAPACITY} ({percentage*100:.1f}%)"
                        cv2.putText(view_result, vol_text, (holder_right + 10, meniscus_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), draw_thickness)

    if not found:
        view_result[:] = (0, 0, 255) 
        cv2.putText(view_result, "NO HOLDER", (20, h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Labels
    cv2.putText(view_original, "1. Input", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), draw_thickness)
    cv2.putText(view_mask, "2. Mask/ROI", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), draw_thickness)
    cv2.putText(view_result, "3. Final", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), draw_thickness)
    
    # Status Label
    if 'status_text' not in locals(): status_text = "Processing..."
    cv2.putText(view_result, status_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), draw_thickness)

    pipeline = np.hstack((view_original, view_mask, view_result))
    return pipeline, final_bottom_y

def main():
    print("Looking for images...")
    images = []
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Directory {IMAGE_FOLDER} not found.")
        return

    for i in range(1, TOTAL_IMAGES + 1):
        path = os.path.join(IMAGE_FOLDER, f"{IMAGE_PREFIX}{i}{IMAGE_EXT}")
        if os.path.exists(path):
            images.append(cv2.imread(path))
            print(f"Loaded {path}")

    if not images:
        print("No images found.")
        return

    idx = 0
    window_name = "Combined Detection Pipeline"
    cv2.namedWindow(window_name)
    
    cached_bottom = None
    auto_thresh_mode = False

    def on_trackbar(val): pass
    cv2.createTrackbar("Threshold", window_name, DEFAULT_MENISCUS_THRESHOLD, 255, on_trackbar)

    print("Controls:")
    print(" [SPACE]: Next Image")
    print(" 'a': Toggle Auto-Threshold (Clustering Mode)")
    print(" 'q': Quit")

    while True:
        # Check if the user clicked the 'X' button
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed. Exiting.")
                break
        except Exception:
            pass

        current_fixed_bottom = cached_bottom if idx != 0 else None
        
        # Safe trackbar access
        try:
            slider_val = cv2.getTrackbarPos("Threshold", window_name)
        except cv2.error:
            print("Window likely closed. Exiting.")
            break
        
        pipeline_img, detected_bottom = process_image(
            images[idx], 
            slider_val, 
            fixed_bottom=current_fixed_bottom,
            auto_thresh=auto_thresh_mode
        )
        
        if idx == 0 and detected_bottom is not None:
            cached_bottom = detected_bottom
        
        final_display = resize_to_fit(pipeline_img, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
        cv2.imshow(window_name, final_display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): 
            break
        if key == ord(' '):
            idx = (idx + 1) % len(images)
            print(f"Showing Image {idx+1}")
        if key == ord('a'):
            auto_thresh_mode = not auto_thresh_mode
            print(f"Auto Threshold Mode: {'ON (Clustering)' if auto_thresh_mode else 'OFF (Manual)'}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()