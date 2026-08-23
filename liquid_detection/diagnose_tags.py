"""
diagnose_tags.py -- figure out why AprilTags aren't being detected.

Grabs a frame from the camera, tries to detect the tags, prints diagnostics,
and saves an annotated image you can look at (default: ~/tag_debug.png).

  python liquid_detection/diagnose_tags.py

Then open ~/tag_debug.png. Common problems it reveals:
  * blurry image        -> the ArduCam lens is out of focus (turn the lens ring)
  * very dark / washed  -> exposure; add light or the image is over/under exposed
  * tags cut off/small  -> move the camera so both tags are fully in view
  * 'rejected' > 0 but detected 0 -> it SEES tag-like squares but can't decode
    them (usually blur, glare, or too little white border/quiet zone)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import cv2

from hardware import open_camera
from aruco_compat import make_aruco_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pi", action="store_true", help="force USB webcam")
    ap.add_argument("--out", default=os.path.expanduser("~/tag_debug.png"))
    ap.add_argument("--frames", type=int, default=15, help="warm-up frames")
    args = ap.parse_args()

    print(f"OpenCV version: {cv2.__version__}")
    det = make_aruco_detector("DICT_APRILTAG_36h11")
    print(f"aruco API: {'new (ArucoDetector)' if det._new else 'legacy (detectMarkers)'}")

    cam = open_camera(prefer_pi=not args.no_pi)
    frame = None
    for _ in range(max(1, args.frames)):     # let auto-exposure settle
        f = cam.read()
        if f is not None:
            frame = f
    cam.close()

    if frame is None:
        print("ERROR: no frame from the camera.")
        return

    print(f"frame: shape={frame.shape} dtype={frame.dtype}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    print(f"brightness: min={int(gray.min())} mean={int(gray.mean())} max={int(gray.max())}"
          f"  (want a wide spread; mean ~60-180)")
    # A rough focus/sharpness score: variance of the Laplacian (low = blurry).
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"sharpness (Laplacian var): {sharp:.0f}   (blurry < ~100, sharp > ~300)")

    # Detect on the normal grayscale, and on a red/blue-swapped copy, to check
    # whether channel order is affecting detection at all.
    def count(g):
        c, ids, rej = det.detectMarkers(g)
        return (0 if ids is None else len(ids),
                None if ids is None else ids.flatten().tolist(),
                0 if rej is None else len(rej), c, ids)

    n, ids_list, nrej, corners, ids = count(gray)
    if frame.ndim == 3:
        swapped = frame[:, :, ::-1]
        g2 = cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY)
        n2, ids2, _, _, _ = count(g2)
    else:
        n2, ids2 = n, ids_list

    print(f"\nTAGS DETECTED: {n}   ids={ids_list}")
    print(f"rejected tag-like candidates: {nrej}")
    print(f"(sanity) detected with channels swapped: {n2}  ids={ids2}")

    viz = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(viz, corners, ids)
    cv2.imwrite(args.out, viz)
    print(f"\nSaved annotated frame to: {args.out}")
    print("Open that image (or send it) so we can see what the camera sees.")

    if n >= 2:
        print("\n=> Both tags detected. Detection is working.")
    elif n == 1:
        print("\n=> Only ONE tag detected. The other is missing/blocked/out of frame.")
    elif nrej > 0:
        print("\n=> Sees tag-like squares but can't decode them. Usually FOCUS or"
              " glare, or the white border/quiet zone around the tag is too small.")
    else:
        print("\n=> No tag-like squares at all. Check the image: focus, lighting,"
              " and that both tags are actually in view.")


if __name__ == "__main__":
    main()
