import cv2
import time
from picamera2 import Picamera2

print("Starting camera test...")

try:
    # 1. Initialize picamera2
    picam2 = Picamera2()
    if not picam2.camera:
        print("Error: No cameras found by libcamera.")
        print("Check connection and config.txt.")
        exit()

    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    print("picamera2 started.")

    # 2. Create an OpenCV window
    # This sometimes helps initialize the display system
    cv2.namedWindow("Camera Test Window")
    print("OpenCV window created.")

    while True:
        # 3. Capture a frame
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 4. Print to terminal so we know the loop is running
        print(f"LOOP RUNNING: Captured frame with shape {frame_bgr.shape}")

        # 5. Show the frame
        cv2.imshow("Camera Test Window", frame_bgr)

        # 6. CRITICAL: This line processes GUI events and allows the
        #    window to refresh. Without it, the window will not appear.
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("'q' pressed. Exiting.")
            break
        
        # Give the CPU a tiny break
        time.sleep(0.01)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 7. Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    print("Script finished. Window destroyed.")
