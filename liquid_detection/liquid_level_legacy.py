"""
liquid_level_legacy.py -- Original gradient-mass liquid detection + streaming
pump communication.

This is the ORIGINAL detection method from the main branch (the "Gradient Mass"
AprilTag monitor you preferred), combined with the pump communication that
actually worked on your hardware: the Pi continuously STREAMS the measured
volume to the pump as an ASCII line "12.3mL\r\n" (no polling handshake needed).

Detection is byte-for-byte the original algorithm:
  * two AprilTags define the ROI,
  * CLAHE contrast enhancement,
  * horizontal blur + vertical Sobel + morphological span filter to find the
    "mass" of vertical change (the meniscus),
  * piece-wise height->volume curve, median smoothing, and the jump guard.

Communication is the streaming method (matches what worked):
  * every --interval seconds, write the current volume in --format
    (default "{volume:.1f}mL\r\n") to the serial port.

Run (headless, streaming to the pump):
  python liquid_detection/liquid_level_legacy.py \
         --port /dev/ttyUSB0 --baud 9600 --capacity 12 --no-window

Add a window (needs a display) by dropping --no-window.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import cv2

from hardware import open_camera
from aruco_compat import make_aruco_detector
from liquid_level import MeniscusDetector, MeniscusConfig

# Mark-rejecting multi-cue detector (built lazily so import stays cheap).
_MULTICUE = None


def _multicue():
    global _MULTICUE
    if _MULTICUE is None:
        _MULTICUE = MeniscusDetector(MeniscusConfig())
    return _MULTICUE

# --- Original detection settings (from the main-branch script) ---
SMOOTHING_WINDOW = 15          # frames for median filtering of volume
GRADIENT_FILTER_WIDTH = 0.80   # a line must span >= this fraction of the width
GRADIENT_SUM_WIDTH = 0.20      # measure the middle this fraction of the tube
CONE_MIN_SCORE = 2.6           # min prominence for a cone-region surface


def calculate_volume_from_height(height_pct, max_capacity):
    """
    Original piece-wise curve: the first 2/15 of the height corresponds to 1 mL,
    the rest scales linearly to max_capacity.
    """
    h_split = 2.0 / 15.0
    v_at_split = 1.0
    if height_pct <= 0:
        return 0.0
    if height_pct <= h_split:
        return (height_pct / h_split) * v_at_split
    h_remaining_range = 1.0 - h_split
    v_remaining_range = max_capacity - v_at_split
    height_above_split = height_pct - h_split
    return v_at_split + (height_above_split / h_remaining_range) * v_remaining_range


def calculate_volume_conical(height_pct, max_capacity, cone_frac):
    """
    Cone + cylinder volume model for a conical-bottom tube.

    The bottom ``cone_frac`` of the measured height is a cone (radius growing
    linearly from the tip), topped by a straight cylinder. The cone is narrow
    near the tip, so its volume grows with the CUBE of height -- far less per mm
    than the cylinder -- which is what makes bottom-of-tube readings accurate.

      * for h <= cone:  V = C/(3D) * h^3 / f^2   (cubic)
      * for h  > cone:  V = C * (h - 2f/3) / D    (linear)
      f = cone_frac, D = 1 - 2f/3, h = height fraction (0..1), C = capacity.

    cone_frac = 0 reduces to a straight linear fill; V(1) == max_capacity.
    """
    f = float(cone_frac)
    if height_pct <= 0:
        return 0.0
    if height_pct >= 1:
        return float(max_capacity)
    if f <= 0:
        return float(height_pct * max_capacity)
    f = min(f, 0.99)
    denom = 1.0 - (2.0 * f / 3.0)
    if height_pct <= f:
        return float(max_capacity / (3.0 * denom) * (height_pct ** 3) / (f * f))
    return float(max_capacity * (height_pct - 2.0 * f / 3.0) / denom)


def find_meniscus_gradient_mass(roi_gray):
    """Original Gradient Mass meniscus finder. Returns (best_y, viz, max_score)."""
    h, w = roi_gray.shape
    if h == 0 or w == 0:
        return -1, None, 0

    blur = cv2.boxFilter(roi_gray, -1, (25, 1))
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel = np.absolute(sobel_y)
    grad_norm = cv2.normalize(abs_sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    k_w = max(1, int(w * GRADIENT_FILTER_WIDTH))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    grad_filtered = cv2.morphologyEx(grad_norm, cv2.MORPH_OPEN, kernel)

    center_col = w / 2
    half_width_px = int((w * GRADIENT_SUM_WIDTH) / 2)
    start_col = max(0, int(center_col - half_width_px))
    end_col = min(w, int(center_col + half_width_px))
    if end_col > start_col:
        row_scores = np.sum(grad_filtered[:, start_col:end_col], axis=1)
    else:
        row_scores = np.sum(grad_filtered, axis=1)
        start_col, end_col = 0, w

    margin = int(h * 0.1)
    row_scores[:margin] = 0
    row_scores[h - margin:] = 0

    window_size = 10
    mass_scores = np.convolve(row_scores, np.ones(window_size), mode='same')
    best_y = int(np.argmax(mass_scores))
    max_score = int(mass_scores[best_y])

    grad_viz = cv2.cvtColor(grad_filtered, cv2.COLOR_GRAY2BGR)
    cv2.line(grad_viz, (start_col, 0), (start_col, h), (0, 255, 255), 1)
    cv2.line(grad_viz, (end_col, 0), (end_col, h), (0, 255, 255), 1)
    cv2.line(grad_viz, (0, best_y), (w, best_y), (0, 0, 255), 2)
    return best_y, grad_viz, max_score


def detect_cone_meniscus(roi_enhanced, cone_zone):
    """
    Find the liquid surface inside the narrow conical bottom.

    The main gradient detector needs a line spanning ~80% of the ROI width, but
    in the cone the tube (and the surface) is much narrower, so that fails. Here
    we look only at a central vertical strip -- the surface crosses the centre
    no matter how narrow the cone gets -- and find the strongest horizontal
    brightness step within the bottom ``cone_zone`` of the ROI.

    Returns (surface_row_in_roi, prominence_score) or (-1, 0.0).
    """
    h, w = roi_enhanced.shape
    ch = max(8, int(h * cone_zone))
    y0 = h - ch
    cone = roi_enhanced[y0:h, :]
    cw = max(4, int(w * 0.34))
    x0 = (w - cw) // 2
    strip = cone[:, x0:x0 + cw].astype(np.float64)
    strip = cv2.GaussianBlur(strip, (5, 1), 0)          # smooth across the strip
    prof = strip.mean(axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()
    grad = np.abs(np.gradient(prof))
    m = max(1, int(ch * 0.08))                          # ignore very top/bottom
    grad[:m] = 0
    grad[len(grad) - m:] = 0
    if grad.max() < 1e-6:
        return -1, 0.0
    best = int(np.argmax(grad))
    base = np.median(grad[grad > 0]) if np.any(grad > 0) else 1.0
    score = float(grad[best] / (base + 1e-6))
    return y0 + best, score


def build_cone_zoom(color_roi, cone_zone, target_h, surface_rel=None):
    """A magnified panel of the bottom (cone) region of the tube for display."""
    h, w = color_roi.shape[:2]
    ch = max(8, int(h * cone_zone))
    y0 = h - ch
    cone = color_roi[y0:h, :].copy()
    if cone.size == 0:
        return None
    scale = target_h / cone.shape[0]
    new_w = int(min(240, max(70, cone.shape[1] * scale)))   # cap the panel width
    panel = cv2.resize(cone, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    if surface_rel is not None and surface_rel >= y0:
        py = int((surface_rel - y0) * scale)
        cv2.line(panel, (0, py), (new_w, py), (0, 255, 0), 2)
    cv2.putText(panel, "CONE", (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return panel


def process_frame(frame, detector, vol_history, tube_capacity,
                  cone_frac=0.0, use_original_curve=False,
                  cone_zone=0.28, cone_detect=True, detector_mode="gradient",
                  empty_pct=0.0, full_pct=1.0, want_viz=True):
    """
    Original processing pipeline.
    Returns (viz, current_volume, status, cone_zoom_panel_or_None, raw_pct).

    empty_pct/full_pct are the ROI fractions (0 = bottom tag, 1 = top tag) that
    correspond to the tube's 0 mL and full marks. They calibrate away the fact
    that the tags aren't at the tube's zero/full lines. Default 0/1 = no cal.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_volume = 0.0
    cone_zoom = None
    raw_pct = 0.0

    corners, ids, _ = detector.detectMarkers(gray)
    detected_tags = []
    roi_defined = False
    view_result = frame.copy() if want_viz else None
    roi_y1 = roi_y2 = roi_x1 = roi_x2 = 0

    if ids is not None and len(ids) >= 2:
        if want_viz:
            cv2.aruco.drawDetectedMarkers(view_result, corners, ids)
        for i in range(len(ids)):
            c = corners[i][0]
            x, y, tw, th = cv2.boundingRect(c.astype(int))
            center_y = int(c[:, 1].mean())
            detected_tags.append({'center_y': center_y, 'bbox_x': x, 'bbox_w': tw, 'bbox_top': y})
        detected_tags.sort(key=lambda t: t['center_y'])
        top_tag, bottom_tag = detected_tags[0], detected_tags[-1]
        roi_y1 = top_tag['center_y']
        roi_y2 = bottom_tag['bbox_top']
        roi_x1 = bottom_tag['bbox_x']
        roi_x2 = bottom_tag['bbox_x'] + bottom_tag['bbox_w']
        if roi_y2 > roi_y1 + 10 and roi_x2 > roi_x1 + 10:
            roi_defined = True
            if want_viz:
                cv2.rectangle(view_result, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

    meniscus_global_y = -1
    status = "Waiting for Tags..."

    if roi_defined:
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi_gray.size > 0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi_enhanced = clahe.apply(roi_gray)

            # Main detection: either the original gradient-mass finder, or the
            # multi-cue finder that REJECTS graduation marks (for graduated /
            # clear-liquid tubes).
            if detector_mode == "multicue":
                Hroi = roi_enhanced.shape[0]
                res = _multicue().detect(roi_enhanced, 100.0 / max(Hroi, 1))
                main_ok = res.found
                best_y_rel = int(round(res.row)) if res.found else -1
                lock_label = "Locked (Multi)"
            else:
                best_y_rel, _, gradient_score_val = find_meniscus_gradient_mass(roi_enhanced)
                main_ok = best_y_rel != -1 and gradient_score_val >= 5000 * GRADIENT_SUM_WIDTH
                lock_label = "Locked (Gradient)"

            meniscus_rel = -1
            if main_ok:
                meniscus_rel = best_y_rel
                status = lock_label
            elif cone_detect:
                # Wide-span found nothing -> the surface is likely low, in the
                # narrow cone. Use the cone-focused (central-strip) detector.
                cone_rel, cone_score = detect_cone_meniscus(roi_enhanced, cone_zone)
                if cone_rel != -1 and cone_score >= CONE_MIN_SCORE:
                    meniscus_rel = cone_rel
                    status = "Locked (Cone)"
                else:
                    status = "Empty"
            else:
                status = ("Empty" if best_y_rel != -1 else "No Strong Edge")

            if meniscus_rel != -1:
                meniscus_global_y = roi_y1 + meniscus_rel
                total_h = roi_y2 - roi_y1
                liquid_h = roi_y2 - meniscus_global_y
                pct = max(0.0, min(1.0, liquid_h / total_h))

                current_median = np.median(vol_history) if vol_history else 0.0
                if current_median < 0.05 and pct > 0.50:
                    pct = 0.0
                    meniscus_global_y = roi_y2
                    status = "Ignored Jump (>50%)"

                vol_history.append(pct)
                pct = float(np.median(vol_history))
                raw_pct = pct
                # Map ROI fraction -> true tube fraction using the calibration
                # anchors, then apply the volume curve.
                span = full_pct - empty_pct
                tube_pct = pct if span <= 1e-6 else max(0.0, min(1.0, (pct - empty_pct) / span))
                if use_original_curve:
                    current_volume = calculate_volume_from_height(tube_pct, tube_capacity)
                else:
                    current_volume = calculate_volume_conical(tube_pct, tube_capacity, cone_frac)

                if want_viz:
                    cv2.line(view_result, (roi_x1 - 20, meniscus_global_y),
                             (roi_x2 + 20, meniscus_global_y), (0, 255, 0), 3)
                    label = f"{current_volume:.1f}ml ({(current_volume/tube_capacity)*100:.0f}%)"
                    cv2.putText(view_result, label, (roi_x2 + 15, meniscus_global_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                meniscus_global_y = roi_y2

            # Zoomed cone panel (shows the liquid in the narrow bottom).
            if want_viz:
                cone_zoom = build_cone_zoom(
                    frame[roi_y1:roi_y2, roi_x1:roi_x2], cone_zone,
                    view_result.shape[0],
                    surface_rel=(meniscus_rel if meniscus_rel != -1 else None))
    else:
        status = "No Tags Found" if ids is None else "ROI Error"

    if want_viz:
        cv2.putText(view_result, status, (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return view_result, current_volume, status, cone_zoom, raw_pct


def main():
    ap = argparse.ArgumentParser(description="Original gradient detection + streaming pump comms")
    ap.add_argument("--port", default=None, help="pump serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--capacity", type=float, default=12.0, help="tube capacity in mL")
    ap.add_argument("--cone-frac", type=float, default=0.15,
                    help="fraction of the measured height taken by the conical "
                         "bottom (cone height / distance between the two tags). "
                         "Makes bottom-of-tube volumes accurate. 0 = linear.")
    ap.add_argument("--original-curve", action="store_true",
                    help="use the old 2/15 piece-wise curve instead of the cone model")
    ap.add_argument("--cone-zone", type=float, default=0.28,
                    help="bottom fraction of the tube treated as the cone: shown "
                         "in the zoom panel and where cone-focused detection runs")
    ap.add_argument("--no-cone-detect", action="store_true",
                    help="disable the cone-focused detector (only wide-span)")
    ap.add_argument("--detector", choices=["gradient", "multicue"], default="gradient",
                    help="meniscus finder: 'gradient' (original) or 'multicue' "
                         "(rejects graduation marks -- use for graduated / clear tubes)")
    ap.add_argument("--empty-pct", type=float, default=0.0,
                    help="ROI fraction (0=bottom tag..1=top tag) that reads 0 mL; "
                         "set live with 'z'. Corrects the tube's zero offset.")
    ap.add_argument("--full-pct", type=float, default=1.0,
                    help="ROI fraction that reads `capacity` mL; set live with 'f'.")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between streamed lines")
    ap.add_argument("--format", dest="fmt", default=r"{volume:.1f}mL\r\n",
                    help=r"streamed ASCII line; use {volume} and \r \n (default '{volume:.1f}mL\r\n')")
    ap.add_argument("--no-window", action="store_true", help="headless (no GUI)")
    ap.add_argument("--no-pi", action="store_true", help="force USB webcam")
    # --- Stabilisation (stops the number flickering when the level is static) ---
    ap.add_argument("--smooth", type=int, default=31,
                    help="median window in frames (bigger = steadier, more lag)")
    ap.add_argument("--ema", type=float, default=0.15,
                    help="EMA factor 0-1 (smaller = smoother, more lag; 1 = off)")
    ap.add_argument("--deadband", type=float, default=0.1,
                    help="mL; the shown/sent value only moves when it changes by "
                         "more than this (holds steady when the level isn't changing)")
    args = ap.parse_args()

    fmt = args.fmt.encode("ascii", "ignore").decode("unicode_escape")

    # Serial (optional): stream the volume to the pump.
    serial_port = None
    if args.port:
        import serial
        try:
            serial_port = serial.Serial(args.port, args.baud, timeout=0)
            print(f"[legacy] streaming volume to {args.port} @ {args.baud} as {fmt!r} every {args.interval}s")
        except serial.SerialException as e:
            print(f"[legacy] serial init failed: {e}")
    else:
        print("[legacy] no --port; running detection only (no pump output).")

    detector = make_aruco_detector("DICT_APRILTAG_36h11")

    cam = open_camera(prefer_pi=not args.no_pi)
    window = "Liquid Level (Gradient) + Pump"
    if not args.no_window:
        cv2.namedWindow(window)

    vol_history = deque(maxlen=max(1, args.smooth))
    last_stream = 0.0
    last_print = 0.0
    ema_vol = None        # smoothed value
    held_vol = None       # deadbanded value that is actually shown/sent
    empty_pct = args.empty_pct   # ROI fraction that is 0 mL (live-calibratable)
    full_pct = args.full_pct     # ROI fraction that is `capacity` mL
    last_pct = 0.0
    if not args.no_window:
        print("\nCALIBRATION (in the window): fill to the tube's 0 mark and press "
              "'z'; fill to the full/capacity mark and press 'f'. 'p' prints values.")
    print("\nSystem ready. Ctrl-C (headless) or 'q' (window) to quit.")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            now = time.time()
            viz, volume, status, cone_zoom, last_pct = process_frame(
                frame, detector, vol_history, args.capacity,
                cone_frac=args.cone_frac, use_original_curve=args.original_curve,
                cone_zone=args.cone_zone, cone_detect=not args.no_cone_detect,
                detector_mode=args.detector, empty_pct=empty_pct, full_pct=full_pct,
                want_viz=not args.no_window)

            # --- Stabilisation pipeline -------------------------------------
            # The median (done in process_frame over --smooth frames) already
            # rejects occasional edge-snap spikes. On top of it:
            #  1) EMA smoothing to iron out sub-frame wobble,
            #  2) a deadband so the shown/sent number only moves on a real
            #     change and stays rock-steady when the level is static.
            if ema_vol is None:
                ema_vol = volume
            else:
                ema_vol += args.ema * (volume - ema_vol)
            if held_vol is None or abs(ema_vol - held_vol) >= args.deadband:
                held_vol = ema_vol
            out_vol = round(held_vol, 1)

            # --- Stream the stabilised volume to the pump -------------------
            if serial_port and (now - last_stream) >= args.interval:
                try:
                    serial_port.write(fmt.format(volume=out_vol).encode("ascii", "ignore"))
                except Exception as e:  # noqa: BLE001
                    print(f"[legacy] serial write error: {e}")
                last_stream = now

            if not args.no_window:
                if viz is not None:
                    cv2.putText(viz, f"{out_vol:.1f} mL (stable)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(viz, f"pos {last_pct*100:4.0f}%  cal[0mL={empty_pct*100:.0f}%"
                                f" full={full_pct*100:.0f}%]  z/f=set p=print",
                                (20, viz.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)
                    # Show the zoomed cone panel alongside the main view.
                    if cone_zoom is not None and cone_zoom.shape[0] == viz.shape[0]:
                        display = np.hstack([viz, cone_zoom])
                    else:
                        display = viz
                    cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('z'):
                    empty_pct = last_pct
                    print(f"[cal] 0 mL set at pos {empty_pct*100:.1f}%")
                elif key == ord('f'):
                    full_pct = last_pct
                    print(f"[cal] full ({args.capacity:g} mL) set at pos {full_pct*100:.1f}%")
                elif key == ord('p'):
                    print(f"[cal] --empty-pct {empty_pct:.4f} --full-pct {full_pct:.4f}")
            else:
                if now - last_print >= 0.5:
                    print(f"[level] {out_vol:5.1f} mL   (raw {volume:5.1f})   {status}",
                          flush=True)
                    last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if serial_port:
            serial_port.close()
        cv2.destroyAllWindows()
        if (empty_pct, full_pct) != (0.0, 1.0):
            print(f"\n[cal] final calibration:  --empty-pct {empty_pct:.4f} "
                  f"--full-pct {full_pct:.4f}  (add to your command / tanda_launch.sh)")


if __name__ == "__main__":
    main()
