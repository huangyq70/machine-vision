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
TUBE_CAPACITY = 12.0  # Total capacity in ml
SMOOTHING_WINDOW = 15 # For median filtering of volume

# Gradient Analysis Settings
# 1. SPAN FILTER: How wide a line must be to be considered (0.80 = 80% of tube width)
GRADIENT_FILTER_WIDTH = 0.80 
# 2. SUMMATION WIDTH: Which part of the tube to measure (0.50 = Middle 50%)
GRADIENT_SUM_WIDTH = 0.20

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

def find_meniscus_gradient_mass(roi_gray):
    """
    Finds the meniscus by looking for a 'mass' of vertical change.
    1. Blurs horizontally to emphasize flat features.
    2. Calculates vertical derivative (Sobel Y).
    3. Filters for gradients that span at least X% of the width (removing bubbles).
    4. Sums rows within the CENTER X% STRIP to get a signal profile.
    5. Uses a moving window to find the area with the most energy (the meniscus mass).
    
    Returns: (best_y, visualization_image, max_score)
    """
    h, w = roi_gray.shape
    if h == 0 or w == 0: return -1, None, 0

    # 1. Horizontal Blur (Smear out vertical noise like bubbles)
    # Kernel (25, 1) means we average 25 pixels horizontally
    blur = cv2.boxFilter(roi_gray, -1, (25, 1))
    
    # 2. Vertical Derivative (Sobel Y)
    # Highlights horizontal edges
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel = np.absolute(sobel_y)
    
    # Normalize for morphology (0-255)
    grad_norm = cv2.normalize(abs_sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 3. SPAN FILTER: Morphological Opening
    # Create a kernel that is a horizontal line of GRADIENT_FILTER_WIDTH% width.
    # This removes any feature shorter than this width (bubbles, scratches).
    k_w = max(1, int(w * GRADIENT_FILTER_WIDTH))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    
    # Apply Open (Erode then Dilate) -> Removes small/curved features, keeps long lines
    grad_filtered = cv2.morphologyEx(grad_norm, cv2.MORPH_OPEN, kernel)
    
    # 4. Project to 1D Profile (Sum across CENTER REGION)
    # Calculate start/end columns based on GRADIENT_SUM_WIDTH
    center_col = w / 2
    half_width_px = int((w * GRADIENT_SUM_WIDTH) / 2)
    start_col = max(0, int(center_col - half_width_px))
    end_col = min(w, int(center_col + half_width_px))

    if end_col > start_col:
        # Only sum the filtered gradient in the defined center strip
        row_scores = np.sum(grad_filtered[:, start_col:end_col], axis=1)
    else:
        # Fallback to full width if calc fails
        row_scores = np.sum(grad_filtered, axis=1)
        start_col, end_col = 0, w
    
    # Mask Margins
    margin = int(h * 0.1)
    row_scores[:margin] = 0
    row_scores[h-margin:] = 0
    
    # 5. Find Mass of Change (Moving Window Sum)
    window_size = 10 # 10 pixel tall window
    
    # Convolve with a box of 1s to get the moving sum
    mass_scores = np.convolve(row_scores, np.ones(window_size), mode='same')
    
    # The peak of the mass scores is the center of the meniscus region
    best_y = np.argmax(mass_scores)
    max_score = mass_scores[best_y] # Calculate the strength of the detection
    
    # --- Visualization ---
    # Use the FILTERED gradient for display so user sees what the algo sees
    grad_viz = cv2.cvtColor(grad_filtered, cv2.COLOR_GRAY2BGR)
    
    # Draw vertical lines to show the sum strip
    cv2.line(grad_viz, (start_col, 0), (start_col, h), (0, 255, 255), 1)
    cv2.line(grad_viz, (end_col, 0), (end_col, h), (0, 255, 255), 1)

    # Draw the detected line
    cv2.line(grad_viz, (0, best_y), (w, best_y), (0, 0, 255), 2)
    # Draw the window bounds faintly
    cv2.line(grad_viz, (0, best_y - window_size//2), (w, best_y - window_size//2), (0, 100, 255), 1)
    cv2.line(grad_viz, (0, best_y + window_size//2), (w, best_y + window_size//2), (0, 100, 255), 1)
    
    # Add Score Label on Mask Panel
    cv2.putText(grad_viz, f"Grad: {int(max_score)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return int(best_y), grad_viz, int(max_score)

def process_frame(frame, detector, vol_history=None):
    """
    Main processing pipeline.
    Layout: [Raw ROI] [Gradient Map] [Main Feed]
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- 1. AprilTag Detection ---
    corners, ids, rejected = detector.detectMarkers(gray)
    
    detected_tags = []
    roi_defined = False
    
    view_result = frame.copy()
    roi_y1, roi_y2, roi_x1, roi_x2 = 0, 0, 0, 0
    
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(view_result, corners, ids)
        for i, tag_id in enumerate(ids):
            c = corners[i][0]
            points = c.astype(int)
            x, y, tw, th = cv2.boundingRect(points)
            center_y = int(c[:, 1].mean())
            detected_tags.append({'center_y': center_y, 'bbox_x': x, 'bbox_w': tw, 'bbox_top': y})
            
        detected_tags.sort(key=lambda t: t['center_y'])
        top_tag = detected_tags[0]
        bottom_tag = detected_tags[-1]
        
        roi_y1 = top_tag['center_y']
        roi_y2 = bottom_tag['bbox_top']
        roi_x1 = bottom_tag['bbox_x']
        roi_x2 = bottom_tag['bbox_x'] + bottom_tag['bbox_w']
        
        if roi_y2 > roi_y1 + 10 and roi_x2 > roi_x1 + 10:
            roi_defined = True
            cv2.rectangle(view_result, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
            cv2.putText(view_result, "ROI", (roi_x1, roi_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- 3. Water Detection ---
    meniscus_global_y = -1
    debug_text = "Status: Waiting for Tags..."
    
    # Visuals containers
    roi_color_viz = None
    mask_viz_center = None
    gradient_score_val = 0
    
    if roi_defined:
        # Extract ROI
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_color_viz = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        
        if roi_gray.size > 0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi_enhanced = clahe.apply(roi_gray)
            
            # === GRADIENT MASS METHOD ===
            best_y_rel, grad_viz, gradient_score_val = find_meniscus_gradient_mass(roi_enhanced)
            
            # Use Gradient Viz for display
            mask_viz_center = grad_viz
            
            if best_y_rel != -1:
                # Check: Is the gradient strong enough?
                # Adjusted threshold for filtered gradient (values are lower after morphology)
                if gradient_score_val < 5000 * GRADIENT_SUM_WIDTH:
                    # Assume empty
                    meniscus_global_y = roi_y2 
                    debug_text = f"Status: Empty (Grad {int(gradient_score_val)} < {5000 * GRADIENT_SUM_WIDTH})"
                else:
                    meniscus_global_y = roi_y1 + best_y_rel
                    debug_text = "Status: Locked (Gradient)"
            else:
                debug_text = "Status: No Strong Edge"

            # --- Result Drawing & Calc ---
            if meniscus_global_y != -1:
                
                # --- CALC PCT ---
                total_h = roi_y2 - roi_y1
                liquid_h = roi_y2 - meniscus_global_y
                pct = max(0.0, min(1.0, liquid_h / total_h))
                
                # --- JUMP GUARD ---
                current_median = 0.0
                if vol_history:
                    current_median = np.median(vol_history)
                
                # Rule: Ignore sudden jumps from 0 to >50%
                if current_median < 0.05 and pct > 0.50:
                    pct = 0.0
                    meniscus_global_y = roi_y2
                    debug_text = "Status: Ignored Jump (>50%)"

                # Apply Smoothing
                if vol_history is not None:
                    vol_history.append(pct)
                    pct = np.median(vol_history)

                # --- DRAWING ---
                # Draw on Main Feed
                cv2.line(view_result, (roi_x1 - 20, meniscus_global_y), (roi_x2 + 20, meniscus_global_y), (0, 255, 0), 3)
                
                # Draw on Raw ROI Viz (Green Line)
                if roi_color_viz is not None:
                    y_rel = meniscus_global_y - roi_y1
                    cv2.line(roi_color_viz, (0, y_rel), (roi_color_viz.shape[1], y_rel), (0, 255, 0), 3)
                
                # Draw Top and Bottom Limits (Purple)
                cv2.line(view_result, (roi_x1 - 10, roi_y1), (roi_x2 + 10, roi_y1), (255, 0, 255), 2) # Top
                cv2.line(view_result, (roi_x1 - 10, roi_y2), (roi_x2 + 10, roi_y2), (255, 0, 255), 2) # Bottom
                
                vol = calculate_volume_from_height(pct, TUBE_CAPACITY)
                vol_pct = (vol / TUBE_CAPACITY) * 100
                
                label = f"{vol:.1f}ml ({vol_pct:.0f}%)"
                cv2.putText(view_result, label, (roi_x2 + 15, meniscus_global_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Draw Gradient Score on Main View
                cv2.putText(view_result, f"G:{int(gradient_score_val)}", (roi_x2 + 15, meniscus_global_y + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    else:
        if ids is None: debug_text = "Status: No Tags Found"
        else: debug_text = "Status: ROI Error"

    cv2.putText(view_result, debug_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # --- Construct Visualization ---
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

    # 2. Gradient Panel
    if mask_viz_center is not None:
        m_resized = cv2.resize(mask_viz_center, (FIXED_PANEL_W, target_h))
        side_panels.append(m_resized)
    else:
        blank = np.zeros((target_h, FIXED_PANEL_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No Gradient", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        side_panels.append(blank)

    # 3. Main Result
    side_panels.append(view_result)
    
    return np.hstack(side_panels)

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