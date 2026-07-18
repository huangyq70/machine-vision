"""
calibrate_tube.py -- Build a height->volume calibration curve for your tube.

Why calibrate: a measured curve is the most accurate way to turn a surface
height into a volume. It automatically accounts for the tube's conical/round
bottom, the exact bore, printed-scale offsets, and any optical bias -- for the
specific tube + holder + camera you calibrated. Do it once per tube type.

How it works:
  1. Mount the tube in the holder with both AprilTags visible (see
     guides/test_tube_setup_guide.md).
  2. Add a known amount of liquid (e.g. with a syringe/burette), or fill to a
     printed graduation line.
  3. Press 'c', then type the TRUE volume in mL in the terminal. The tool
     records (measured height mm, your volume mL).
  4. Repeat for 6-12 levels spanning empty -> full (include a few near the
     bottom where the cone makes volume change fastest).
  5. Press 's' to save calibration.csv. Point the monitor at it.

Controls:
  c = capture a calibration point (prompts for volume in the terminal)
  u = undo last point
  s = save CSV
  q = quit (also offers to save)

Usage:
  python calibrate_tube.py --config config.json --out calibration.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import cv2

from liquid_level import AprilTagLocator, MeniscusDetector
from hardware import open_camera, load_configs


def draw_overlay(frame, rect, men, points):
    view = frame.copy()
    h, w = view.shape[:2]
    if rect is not None:
        for c, label in ((rect.top_center, "TOP"), (rect.bottom_center, "BOT")):
            cv2.circle(view, tuple(c.astype(int)), 6, (0, 255, 255), -1)
        if men is not None and men.found:
            # Map the strip row back to an image point along the tube axis.
            a = rect.bottom_center - rect.top_center
            frac = men.row / rect.strip.shape[0]
            pt = rect.top_center + frac * a
            p = tuple(pt.astype(int))
            cv2.line(view, (p[0] - 120, p[1]), (p[0] + 120, p[1]), (0, 255, 0), 3)
            hmm = rect.strip_row_to_height_mm(men.row)
            cv2.putText(view, f"h={hmm:5.1f}mm conf={men.confidence:.2f}",
                        (p[0] + 20, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
    cv2.putText(view, f"points: {len(points)}  [c]apture [u]ndo [s]ave [q]uit",
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i, (hm, vm) in enumerate(points):
        cv2.putText(view, f"{i+1}: {hm:5.1f}mm -> {vm:5.2f}mL",
                    (20, 40 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON config file")
    ap.add_argument("--out", default="calibration.csv")
    ap.add_argument("--no-pi", action="store_true", help="force USB webcam")
    args = ap.parse_args()

    tube_cfg, men_cfg = load_configs(args.config)
    locator = AprilTagLocator(tube_cfg)
    detector = MeniscusDetector(men_cfg)
    cam = open_camera(prefer_pi=not args.no_pi)

    window = "Calibration"
    cv2.namedWindow(window)
    points = []  # (height_mm, volume_ml)
    print(__doc__)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rect = locator.locate(gray)
            men = detector.detect(rect.strip, rect.mm_per_px_strip) if rect else None

            cv2.imshow(window, draw_overlay(frame, rect, men, points))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if rect is None or men is None or not men.found:
                    print("[calibrate] No confident surface right now; adjust and retry.")
                    continue
                hmm = rect.strip_row_to_height_mm(men.row)
                try:
                    raw = input(f"  measured height {hmm:.1f} mm -> enter TRUE volume (mL): ")
                    vol = float(raw)
                except (ValueError, EOFError):
                    print("  invalid; skipped.")
                    continue
                points.append((hmm, vol))
                print(f"  recorded point {len(points)}: {hmm:.1f} mm -> {vol:.2f} mL")
            elif key == ord('u'):
                if points:
                    print(f"  removed {points.pop()}")
            elif key == ord('s'):
                save_points(points, args.out)

        if points and input("Save calibration before quitting? [y/N] ").strip().lower() == 'y':
            save_points(points, args.out)
    finally:
        cam.close()
        cv2.destroyAllWindows()


def save_points(points, path):
    if len(points) < 2:
        print("  need at least 2 points to save.")
        return
    pts = sorted(points)
    with open(path, "w") as fh:
        fh.write("# height_mm,volume_ml  (generated by calibrate_tube.py)\n")
        for hm, vm in pts:
            fh.write(f"{hm:.4f},{vm:.4f}\n")
    print(f"  saved {len(pts)} points to {path}")


if __name__ == "__main__":
    main()
