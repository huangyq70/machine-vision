import cv2
import numpy as np
import os
import sys
import time
from collections import deque

# --- Hardware Support Check ---
try:
    from picamera2 import Picamera2
    USING_PI = True
except ImportError:
    print("Picamera2 not found. Falling back to standard webcam (cv2.VideoCapture).")
    USING_PI = False

# --- Configuration ---
DISPLAY_SIZE = (1600, 900)

# Physics / Geometry
TUBE_CAPACITY = 15.0  # Total capacity in ml
DEFAULT_MENISCUS_THRESHOLD = 28
SMOOTHING_WINDOW = 15 # For median filtering of volume

def resize_to_fit(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def calculate_volume_from_height(height_pct, max_capacity):
    """
    Calculates volume based on height percentage using a piece-wise function.
    Logic: The first 2/15 of height corresponds to 1 unit of volume (1ml).
    The rest scales linearly to max_capacity.
    """
    h_split = 2.0 / 15.0
    v_at_split = 1.0
    
    if height_pct <= 0: return 0.0
    
    if height_pct <= h_split:
        return (height_pct / h_split) * v_at_split
    else:
        h_remaining_range = 1.0 - h_split
        v_remaining_range = max_capacity - v_at_split
        height_above_split = height_pct - h_split
        return v_at_split + (height_above_split / h_remaining_range) * v_remaining_range

def find_pinch_point(mask_roi):
    """
    Finds the meniscus by looking for the narrowest gap in the thresholded mask.
    """
    h, w = mask_roi.shape
    min_gap = float('inf')
    best_y = -1 
    # Ignore top/bottom 5% of the strip to avoid edge artifacts
    margin_y = int(h * 0.05) 
    
    for y in range(margin_y, h - margin_y):
        row_slice = mask_roi[y, :]
        black_indices = np.where(row_slice == 0)[0]
        
        # Need at least 2 black pixels to define a gap
        if len(black_indices) < 2:
            continue
            
        # Simple gap metric: distance between first and last dark pixel
        l_edge = black_indices[0]
        r_edge = black_indices[-1]
        gap = r_edge - l_edge
        
        # We want the smallest gap that isn't zero
        if gap > 0 and gap < min_gap:
            min_gap = gap
            best_y = y

    return best_y

def cluster_votes_simple(votes, tolerance=10):
    """
    Groups Y-coordinates. Returns the average Y of the largest cluster.
    """
    if not votes: return -1, 0
    
    votes.sort()
    clusters = []
    current_cluster = [votes[0]]
    
    for i in range(1, len(votes)):
        if votes[i] - votes[i-1] <= tolerance:
            current_cluster.append(votes[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [votes[i]]
    clusters.append(current_cluster)
    
    best_cluster = max(clusters, key=len)
    mean_y = int(sum(best_cluster) / len(best_cluster))
    return mean_y, len(best_cluster)

def process_frame(frame, detector, auto_thresh=True, manual_thresh=30):
    """
    Main processing pipeline:
    1. Detect AprilTags for ROI
    2. Extract ROI
    3. Run Water Detection on ROI
    4. Visualize Side-by-Side (Raw ROI, Masks, Final)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- 1. AprilTag Detection ---
    corners, ids, rejected = detector.detectMarkers(gray)
    
    detected_tags = []
    roi_defined = False
    roi_coords = None # (x1, y1, x2, y2)
    
    view_result = frame.copy()
    
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(view_result, corners, ids)
        
        # Parse Tag Data
        for i, tag_id in enumerate(ids):
            c = corners[i][0]
            points = c.astype(int)
            x, y, tw, th = cv2.boundingRect(points)
            center_y = int(c[:, 1].mean())
            
            detected_tags.append({
                'center_y': center_y,
                'bbox_x': x,
                'bbox_w': tw,
                'bbox_top': y
            })
            
        # Sort Tags Vertically
        detected_tags.sort(key=lambda t: t['center_y'])
        top_tag = detected_tags[0]
        bottom_tag = detected_tags[-1]
        
        # --- 2. Define ROI from Tags ---
        # Top: Center of Top Tag
        roi_y1 = top_tag['center_y']
        # Bottom: Top edge of Bottom Tag
        roi_y2 = bottom_tag['bbox_top']
        # Sides: Walls of Bottom Tag
        roi_x1 = bottom_tag['bbox_x']
        roi_x2 = bottom_tag['bbox_x'] + bottom_tag['bbox_w']
        
        if roi_y2 > roi_y1 + 10: # Sanity check height
            roi_defined = True
            roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
            
            # Draw ROI Box (Yellow) on Final Result
            cv2.rectangle(view_result, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
            cv2.putText(view_result, "ROI", (roi_x1, roi_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- 3. Water Detection ---
    meniscus_global_y = -1
    debug_text = "Waiting for Tags..."
    debug_masks = []
    roi_color_viz = None # Store raw ROI for visualization
    
    center_t = manual_thresh 
    
    if roi_defined:
        # Extract the ROI
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_color_viz = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy() # Capture raw color ROI
        
        if roi_gray.size > 0:
            # Apply CLAHE to enhance contrast inside the tube
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi_enhanced = clahe.apply(roi_gray)
            
            # --- AUTO MODE: FIND DARKEST PEAK (WHOLE ROI) ---
            if auto_thresh:
                try:
                    hist = cv2.calcHist([roi_enhanced], [0], None, [256], [0, 256])
                    dark_region = hist[:100]
                    peak_val = np.argmax(dark_region)
                    center_t = int(peak_val)
                    center_t = max(15, min(center_t, 90))
                except:
                    center_t = manual_thresh

            # --- Scanning & Voting ---
            scan_range = 15 if auto_thresh else 5
            scan_step = 2
            
            scan_min = max(5, center_t - scan_range)
            scan_max = min(250, center_t + scan_range + 1)
            
            votes = []
            
            for t in range(scan_min, scan_max, scan_step):
                _, t_thresh = cv2.threshold(roi_enhanced, t, 255, cv2.THRESH_BINARY)
                t_clean = cv2.erode(t_thresh, np.ones((3,3), np.uint8), iterations=1)
                t_clean = cv2.dilate(t_clean, np.ones((3,3), np.uint8), iterations=1)
                
                # Find Y relative to ROI
                y_rel = find_pinch_point(t_clean)
                if y_rel != -1:
                    votes.append(y_rel)
                    if auto_thresh:
                        cv2.line(view_result, (roi_x1, roi_y1 + y_rel), (roi_x2, roi_y1 + y_rel), (100, 100, 100), 1)

            # --- Generate Visualization Masks (T-5, T, T+5) ---
            vis_thresholds = [center_t - 5, center_t, center_t + 5]
            for vt in vis_thresholds:
                safe_vt = max(0, min(255, vt))
                _, vt_thresh = cv2.threshold(roi_enhanced, safe_vt, 255, cv2.THRESH_BINARY)
                vt_clean = cv2.erode(vt_thresh, np.ones((3,3), np.uint8), iterations=1)
                vt_clean = cv2.dilate(vt_clean, np.ones((3,3), np.uint8), iterations=1)
                
                mask_viz = cv2.cvtColor(vt_clean, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_viz, f"T:{safe_vt}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                debug_masks.append(mask_viz)

            # --- Consensus ---
            best_y_rel, confidence = cluster_votes_simple(votes)
            
            if best_y_rel != -1:
                meniscus_global_y = roi_y1 + best_y_rel
                
                # Draw Final Meniscus (Green) on Result
                cv2.line(view_result, (roi_x1 - 20, meniscus_global_y), (roi_x2 + 20, meniscus_global_y), (0, 255, 0), 3)
                
                # Draw on Raw ROI Panel (Green)
                if roi_color_viz is not None:
                    cv2.line(roi_color_viz, (0, best_y_rel), (roi_color_viz.shape[1], best_y_rel), (0, 255, 0), 3)
                
                # Draw Top/Bottom on Result
                cv2.line(view_result, (roi_x1 - 10, roi_y1), (roi_x2 + 10, roi_y1), (255, 0, 255), 2)
                cv2.line(view_result, (roi_x1 - 10, roi_y2), (roi_x2 + 10, roi_y2), (255, 0, 255), 2)
                
                # Volume Calc
                total_h = roi_y2 - roi_y1
                liquid_h = roi_y2 - meniscus_global_y
                pct = max(0.0, min(1.0, liquid_h / total_h))
                vol = calculate_volume_from_height(pct, TUBE_CAPACITY)
                vol_pct = (vol / TUBE_CAPACITY) * 100
                
                label = f"{vol:.1f}ml ({vol_pct:.0f}%)"
                cv2.putText(view_result, label, (roi_x2 + 15, meniscus_global_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                scan_info = f"{scan_min}-{scan_max}" if auto_thresh else "+/- 5"
                debug_text = f"Votes: {confidence} | Thresh: {center_t} (Scan: {scan_info})"
            else:
                debug_text = "Detection Failed (No pinch point found)"
    else:
        if ids is not None and len(ids) == 1:
            debug_text = "Need 2 Tags (Top & Bottom)"
        elif ids is None:
            debug_text = "No Tags Found"
        else:
            debug_text = "ROI Error (Tags inverted?)"

    # Draw Debug Info at bottom
    cv2.putText(view_result, debug_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    mode_label = "AUTO (Peak)" if auto_thresh else f"MANUAL"
    cv2.putText(view_result, f"Mode: {mode_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # --- Construct Side-by-Side Visualization (Raw ROI, T-5, T, T+5, Final) ---
    target_h = h
    panels = []
    
    # 1. Raw ROI Panel
    if roi_color_viz is not None:
        # Resize to match target_h
        scale = target_h / roi_color_viz.shape[0]
        new_w = int(roi_color_viz.shape[1] * scale)
        raw_resized = cv2.resize(roi_color_viz, (new_w, target_h))
        cv2.putText(raw_resized, "Raw ROI", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        panels.append(raw_resized)
    else:
        # Placeholder
        placeholder_w = 150
        blank = np.zeros((target_h, placeholder_w, 3), dtype=np.uint8)
        cv2.putText(blank, "No ROI", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(blank)

    # 2. Threshold Masks (T-5, T, T+5)
    titles = [f"T-5 ({center_t-5})", f"T ({center_t})", f"T+5 ({center_t+5})"]
    if debug_masks and len(debug_masks) == 3:
        for i, m in enumerate(debug_masks):
            scale = target_h / m.shape[0]
            new_w = int(m.shape[1] * scale)
            m_resized = cv2.resize(m, (new_w, target_h))
            cv2.putText(m_resized, titles[i], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            panels.append(m_resized)
    else:
        for _ in range(3):
            placeholder_w = 150
            blank = np.zeros((target_h, placeholder_w, 3), dtype=np.uint8)
            panels.append(blank)

    # 3. Final Result Panel
    panels.append(view_result)
    
    # Stack horizontally
    final_composite = np.hstack(panels)

    return final_composite

def main():
    picam2 = None
    cap = None

    # --- 1. Camera Initialization ---
    if USING_PI:
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (1280, 720)})
            picam2.configure(config)
            picam2.start()
            print("Picamera2 started.")
        except Exception as e:
            print(f"FATAL: Could not start Picamera2: {e}")
            return
    else:
        print("Starting Webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ArUco Init
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        print("Error: cv2.aruco not installed.")
        return

    idx = 0
    window_name = "AprilTag Liquid Monitor (Live)"
    cv2.namedWindow(window_name)
    
    auto_thresh_mode = True
    
    def on_trackbar(val): pass
    cv2.createTrackbar("Threshold", window_name, DEFAULT_MENISCUS_THRESHOLD, 255, on_trackbar)

    print("Controls:")
    print(" 'a': Toggle Auto-Threshold")
    print(" 'q': Quit")

    while True:
        # Check window
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        except: pass
        
        # --- Capture Frame ---
        if USING_PI:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret: break

        manual_val = cv2.getTrackbarPos("Threshold", window_name)
        
        # Process
        result_frame = process_frame(
            frame, 
            detector, 
            auto_thresh=auto_thresh_mode, 
            manual_thresh=manual_val
        )
        
        display = resize_to_fit(result_frame, DISPLAY_SIZE)
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('a'):
            auto_thresh_mode = not auto_thresh_mode

    # Cleanup
    if USING_PI:
        picam2.stop()
    elif cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()