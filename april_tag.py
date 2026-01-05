import cv2
import numpy as np
import sys
import time
import os

# --- Configuration ---
IMAGE_FOLDER = "./images/april_tag/"
IMAGE_PREFIX = "image_"
IMAGE_EXT = ".jpg"
TOTAL_IMAGES = 14
DISPLAY_SIZE = (1280, 720) # Resize for consistent display

def resize_to_fit(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def main():
    # --- 1. Load Images ---
    print("Looking for images...")
    images = []
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Directory {IMAGE_FOLDER} not found.")
        print("Please ensure your images are in the correct folder.")
        return

    for i in range(1, TOTAL_IMAGES + 1):
        path = os.path.join(IMAGE_FOLDER, f"{IMAGE_PREFIX}{i}{IMAGE_EXT}")
        if os.path.exists(path):
            images.append(cv2.imread(path))
            print(f"Loaded {path}")

    if not images:
        print("No images found.")
        return

    # --- 2. AprilTag Setup (via cv2.aruco) ---
    # Define the dictionary. 36h11 is the most common AprilTag family.
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        print("AprilTag Detector Initialized (36h11).")
    except AttributeError:
        print("Error: cv2.aruco not found.")
        print("Please install opencv-contrib-python: pip install opencv-contrib-python")
        return

    window_name = "AprilTag Detector (Photo Mode)"
    cv2.namedWindow(window_name)

    print("\nControls:")
    print(" [SPACE]: Next Image")
    print(" 'q': Quit")

    idx = 0

    while True:
        # Check window status
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except: pass

        # --- 3. Get Current Image ---
        # Work on a copy so we don't draw over the original in memory
        frame = images[idx].copy() 

        # --- 4. Detect Tags ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # corners: List of detected tag corners
        # ids: List of tag IDs corresponding to the corners
        # rejected: Candidates that were rejected
        corners, ids, rejected = detector.detectMarkers(gray)

        # --- 5. Draw Results ---
        if ids is not None and len(ids) > 0:
            # Draw standard outlines and IDs provided by ArUco
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            detected_tags = []

            # Process individual tags
            for i, tag_id in enumerate(ids):
                c = corners[i][0]
                points = c.astype(int)
                
                # Get Bounding Box
                # cv2.boundingRect returns the straight rectangle (x,y,w,h) 
                # that encloses the tag contour
                x, y, w, h = cv2.boundingRect(points)
                
                # Get Center
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                
                # Draw Green Bounding Box and Red Center for debug
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Store data for ROI calculation
                detected_tags.append({
                    'id': tag_id[0],
                    'center_y': center_y,
                    'bbox_x': x,        # Left wall of bounding box
                    'bbox_w': w,        # Width of bounding box
                    'bbox_top': y       # Top edge of bounding box
                })

            # --- NEW: ROI Calculation ---
            # We need at least 2 tags (Top and Bottom) to form the ROI
            if len(detected_tags) >= 2:
                # Sort tags by Y position (0 is top of image, larger is lower)
                # [0] will be top tag, [-1] will be bottom tag
                detected_tags.sort(key=lambda t: t['center_y'])
                
                top_tag = detected_tags[0]
                bottom_tag = detected_tags[-1]
                
                # Define ROI Coordinates
                # 1. Top: Center line of the Top Tag
                roi_y1 = top_tag['center_y']
                
                # 2. Bottom: Top wall of the Bottom Tag (Bounding Box Top)
                roi_y2 = bottom_tag['bbox_top']
                
                # 3. Sides: Walls of the Bottom Tag's Bounding Box extended upwards
                # We strictly use the bounding box 'x' and 'width' here
                roi_x1 = bottom_tag['bbox_x']
                roi_x2 = bottom_tag['bbox_x'] + bottom_tag['bbox_w']
                
                # Sanity check: Ensure top is actually above bottom
                if roi_y2 > roi_y1:
                    # Draw ROI in YELLOW (BGR: 0, 255, 255)
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
                    
                    # Add Label
                    cv2.putText(frame, "ROI", (roi_x1, roi_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "ROI Error: Tags Inverted/Too Close", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "ROI Info: Need 2+ Tags", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "No Tags Detected", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        # --- 6. Display ---
        final_display = resize_to_fit(frame, DISPLAY_SIZE)
        cv2.imshow(window_name, final_display)

        # --- 7. Input Handling ---
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            idx = (idx + 1) % len(images)
            print(f"Showing Image {idx + 1}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()