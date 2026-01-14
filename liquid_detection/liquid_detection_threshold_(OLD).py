import cv2
import numpy as np
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
# Physics / Geometry
TUBE_CAPACITY = 15.0  # Total capacity in ml
SMOOTHING_WINDOW = 15 # For median filtering of volume

# Auto-Threshold Scanning Parameters
SCAN_RANGE_BELOW = 10   # How many levels BELOW the peak to scan
SCAN_RANGE_ABOVE = 25  # How many levels ABOVE the peak to scan
SCAN_STEP = 3       # Step size for the scan loop

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

def find_meniscus_by_line(mask_roi):
    """
    Finds the meniscus by detecting the strongest horizontal line (edge) 
    in the binary mask using Hough Transform.
    """
    h, w = mask_roi.shape
    if h == 0 or w == 0: return -1
    
    # 1. Define Margins (Ignore top/bottom 10% to avoid Tag artifacts)
    margin = int(h * 0.1)
    
    # 2. Edge Detection
    # mask_roi is binary (0 or 255), so Canny finds the boundaries
    edges = cv2.Canny(mask_roi, 50, 150)
    
    # Mask out the top/bottom margins
    edges[:margin, :] = 0
    edges[h-margin:, :] = 0
    
    # 3. Find Lines (Probabilistic Hough Transform)
    # minLineLength: Line must be at least 40% of the ROI width to be considered
    # maxLineGap: Allow small gaps (10px) in the line
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=15, 
        minLineLength=int(w * 0.4), 
        maxLineGap=10
    )
    
    if lines is None:
        return -1
    
    best_y = -1
    max_length = -1
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Check slope (Must be roughly horizontal)
        dy = abs(y1 - y2)
        if dy > 5: continue 
        
        # Calculate horizontal length
        length = abs(x2 - x1)
        
        # We want the longest/straightest line
        if length > max_length:
            max_length = length
            best_y = int((y1 + y2) / 2)
            
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

def process_frame(frame, detector, vol_history=None):
    """
    Main processing pipeline.
    Layout: [Raw ROI] [Mask T (with Candidates)] [Main Feed]
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- 1. AprilTag Detection ---
    corners, ids, rejected = detector.detectMarkers(gray)
    
    detected_tags = []
    roi_defined = False
    
    # Start with a clean copy for drawing results
    view_result = frame.copy()
    
    # Variables for calculation
    roi_y1, roi_y2, roi_x1, roi_x2 = 0, 0, 0, 0
    
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
        roi_y1 = top_tag['center_y']
        roi_y2 = bottom_tag['bbox_top']
        roi_x1 = bottom_tag['bbox_x']
        roi_x2 = bottom_tag['bbox_x'] + bottom_tag['bbox_w']
        
        # Sanity Check: Ensure dimensions are positive and within frame
        if roi_y2 > roi_y1 + 10 and roi_x2 > roi_x1 + 10:
            roi_defined = True
            
            # Draw ROI Box (Yellow)
            cv2.rectangle(view_result, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
            cv2.putText(view_result, "ROI", (roi_x1, roi_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- 3. Water Detection ---
    meniscus_global_y = -1
    debug_text = "Status: Waiting for Tags..."
    thresh_debug_text = ""
    
    # Visuals containers
    roi_color_viz = None
    mask_viz_center = None
    center_t = 30 # Default
    
    if roi_defined:
        # Extract the ROI
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_color_viz = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        
        if roi_gray.size > 0:
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi_enhanced = clahe.apply(roi_gray)
            
            # --- AUTO MODE: FIND DARKEST PEAK ---
            try:
                # 1. Use Whole ROI (No crop)
                # 2. Histogram Analysis
                hist = cv2.calcHist([roi_enhanced], [0], None, [256], [0, 256])
                
                # 3. Find Dark Peak
                dark_region = hist[:100]
                peak_val = np.argmax(dark_region)
                
                center_t = int(peak_val)
                # Safety Clamp (15-90)
                center_t = max(15, min(center_t, 90))
            except:
                pass

            thresh_debug_text = f"Auto Thresh: {center_t}"

            # --- Scanning & Voting (Variable Clustering) ---
            scan_min = max(5, center_t - SCAN_RANGE_BELOW)
            scan_max = min(250, center_t + SCAN_RANGE_ABOVE + 1)
            
            votes = []
            
            for t in range(scan_min, scan_max, SCAN_STEP):
                _, t_thresh = cv2.threshold(roi_enhanced, t, 255, cv2.THRESH_BINARY)
                t_clean = cv2.erode(t_thresh, np.ones((3,3), np.uint8), iterations=1)
                t_clean = cv2.dilate(t_clean, np.ones((3,3), np.uint8), iterations=1)
                
                y_rel = find_meniscus_by_line(t_clean)
                if y_rel != -1:
                    votes.append(y_rel)
                    # Draw faint ghost lines on result
                    cv2.line(view_result, (roi_x1, roi_y1 + y_rel), (roi_x2, roi_y1 + y_rel), (100, 100, 100), 1)

            # --- Generate Visualization Mask for Center Threshold ---
            _, center_thresh_img = cv2.threshold(roi_enhanced, center_t, 255, cv2.THRESH_BINARY)
            center_clean = cv2.erode(center_thresh_img, np.ones((3,3), np.uint8), iterations=1)
            center_clean = cv2.dilate(center_clean, np.ones((3,3), np.uint8), iterations=1)
            mask_viz_center = cv2.cvtColor(center_clean, cv2.COLOR_GRAY2BGR)
            
            # --- Find and Draw Top 3 Candidate Lines on Mask ---
            h_roi_viz, w_roi_viz = mask_viz_center.shape[:2]
            margin = int(h_roi_viz * 0.1)
            edges_viz = cv2.Canny(center_clean, 50, 150)
            edges_viz[:margin, :] = 0
            edges_viz[h_roi_viz-margin:, :] = 0
            
            lines_viz = cv2.HoughLinesP(edges_viz, rho=1, theta=np.pi/180, threshold=15, 
                                      minLineLength=int(w_roi_viz * 0.4), maxLineGap=10)
            
            if lines_viz is not None:
                valid_lines = []
                for line in lines_viz:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) > 5: continue
                    length = abs(x2 - x1)
                    valid_lines.append((length, line[0]))
                
                # Sort by length descending
                valid_lines.sort(key=lambda x: x[0], reverse=True)
                
                # Draw top 3
                for i in range(min(3, len(valid_lines))):
                    l = valid_lines[i][1]
                    # Draw Line (Red)
                    cv2.line(mask_viz_center, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)
                    # Draw Rank Number
                    cv2.putText(mask_viz_center, str(i+1), (l[0], l[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(mask_viz_center, f"T:{center_t}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- Consensus ---
            best_y_rel, confidence = cluster_votes_simple(votes)
            
            if best_y_rel != -1:
                meniscus_global_y = roi_y1 + best_y_rel
                
                # Draw Final Meniscus (Green)
                cv2.line(view_result, (roi_x1 - 20, meniscus_global_y), (roi_x2 + 20, meniscus_global_y), (0, 255, 0), 3)
                
                # Draw on Raw ROI Viz
                if roi_color_viz is not None:
                    cv2.line(roi_color_viz, (0, best_y_rel), (roi_color_viz.shape[1], best_y_rel), (0, 255, 0), 3)
                
                # Draw Top and Bottom Limits (Purple)
                cv2.line(view_result, (roi_x1 - 10, roi_y1), (roi_x2 + 10, roi_y1), (255, 0, 255), 2) # Top
                cv2.line(view_result, (roi_x1 - 10, roi_y2), (roi_x2 + 10, roi_y2), (255, 0, 255), 2) # Bottom
                
                # --- Volume Calc ---
                total_h = roi_y2 - roi_y1
                liquid_h = roi_y2 - meniscus_global_y
                pct = max(0.0, min(1.0, liquid_h / total_h))
                
                # Apply Smoothing
                if vol_history is not None:
                    vol_history.append(pct)
                    # Median filter for stability
                    pct = np.median(vol_history)

                vol = calculate_volume_from_height(pct, TUBE_CAPACITY)
                vol_pct = (vol / TUBE_CAPACITY) * 100
                
                label = f"{vol:.1f}ml ({vol_pct:.0f}%)"
                cv2.putText(view_result, label, (roi_x2 + 15, meniscus_global_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                debug_text = f"Status: Locked ({confidence} votes)"
            else:
                debug_text = "Status: Detection Failed (No line found)"
    else:
        # Fallback messages
        if ids is not None and len(ids) == 1:
            debug_text = "Status: Need 2 Tags (Top & Bottom)"
        elif ids is None:
            debug_text = "Status: No Tags Found"
        else:
            debug_text = "Status: ROI Error (Tags inverted?)"

    # --- Draw Info Overlay on Main Result ---
    cv2.putText(view_result, debug_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    if thresh_debug_text:
        cv2.putText(view_result, thresh_debug_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # --- Construct Side-by-Side Visualization ---
    target_h = h
    FIXED_PANEL_W = 240
    side_panels = []
    
    # 1. Raw ROI Panel
    if roi_color_viz is not None:
        raw_resized = cv2.resize(roi_color_viz, (FIXED_PANEL_W, target_h))
        cv2.putText(raw_resized, "Raw ROI", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        side_panels.append(raw_resized)
    else:
        blank = np.zeros((target_h, FIXED_PANEL_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No ROI", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        side_panels.append(blank)

    # 2. Center Threshold Mask
    if mask_viz_center is not None:
        m_resized = cv2.resize(mask_viz_center, (FIXED_PANEL_W, target_h))
        side_panels.append(m_resized)
    else:
        blank = np.zeros((target_h, FIXED_PANEL_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No Data", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        side_panels.append(blank)

    # 3. Add Final Result Panel
    side_panels.append(view_result)
    
    final_composite = np.hstack(side_panels)
    return final_composite

def main():
    picam2 = None
    cap = None

    # --- 1. Camera Initialization ---
    if USING_PI:
        try:
            print("Attempting to initialize Picamera2...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (1280, 720)})
            picam2.configure(config)
            picam2.start()
            print("Picamera2 started successfully.")
        except Exception as e:
            print(f"WARNING: Picamera2 failed: {e}")
            picam2 = None

    if picam2 is None:
        print("Starting Webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("FATAL: Could not open any camera.")
            return
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

    window_name = "AprilTag Liquid Monitor (Live)"
    cv2.namedWindow(window_name)
    
    # Initialize Smoothing History
    vol_history = deque(maxlen=SMOOTHING_WINDOW)
    
    print("\nSystem Ready. Controls: 'q' to Quit.")

    while True:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        except: pass
        
        if picam2:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            break

        result_frame = process_frame(frame, detector, vol_history)
        cv2.imshow(window_name, result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    if picam2: picam2.stop()
    elif cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()