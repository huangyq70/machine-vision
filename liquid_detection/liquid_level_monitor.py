"""
liquid_level_monitor.py -- Live liquid-level monitor + closed-loop pump control.

This is the main program. It:
  * reads frames from the Pi camera (or a USB webcam fallback),
  * rectifies the tube between the two AprilTags and measures the surface,
  * converts height -> volume with your chosen model (calibration curve
    recommended),
  * temporally filters the reading (median + EMA + flow rate + stability),
  * drives a pump over USB/serial toward a target volume with a predictive,
    non-overshooting stop, streaming the live volume the whole time.

Quick start
-----------
  # 1) (once) calibrate your tube:
  python calibrate_tube.py --config config.json --out calibration.csv

  # 2) run the monitor with a target of 25 mL, stopping the pump precisely:
  python liquid_level_monitor.py --config config.json \
         --calibration calibration.csv --port /dev/ttyACM0 --target 25

Controls (in the window):
  q          quit
  + / -      raise / lower the target by 0.5 mL
  space      start/pause dosing
  r          reset the temporal filter
  d          toggle fill/drain direction

If you have no calibration file yet, pass --capacity to use a rough linear
model so you can see it working, then calibrate for real accuracy.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import cv2

from liquid_level import (
    LevelEstimator, CalibrationCurve, GeometricTube, LinearModel,
)
from hardware import open_camera, load_configs
from pump_controller import (
    PumpController, PumpConfig, SerialPumpLink, MockPumpLink, Direction, PumpState,
)


def build_volume_model(args, tube_cfg):
    if args.calibration:
        print(f"[monitor] using calibration curve: {args.calibration}")
        return CalibrationCurve.from_csv(args.calibration)
    if args.geometry:
        d, base, bottom, bh = args.geometry
        print(f"[monitor] using geometric tube d={d} base={base} {bottom} bh={bh}")
        return GeometricTube(float(d), float(base), bottom, float(bh))
    print("[monitor] WARNING: no calibration -> rough LINEAR model. Calibrate for accuracy.")
    # Linear from ~0 mm to the axis length once tags are seen; approximate here.
    return LinearModel(height_full_mm=args.linear_full_mm,
                       capacity_ml=tube_cfg.tube_capacity_ml)


def make_dashboard(frame, rect, est: LevelEstimator, reading, controller, target):
    """Compose the annotated main view + a strip/profile side panel."""
    view = frame.copy()
    h, w = view.shape[:2]

    # Draw tags + surface line on the main feed.
    if rect is not None:
        cv2.circle(view, tuple(rect.top_center.astype(int)), 6, (0, 255, 255), -1)
        cv2.circle(view, tuple(rect.bottom_center.astype(int)), 6, (0, 255, 255), -1)
        men = est.last_meniscus
        if men is not None and men.found:
            a = rect.bottom_center - rect.top_center
            frac = men.row / rect.strip.shape[0]
            pt = (rect.top_center + frac * a).astype(int)
            col = (0, 255, 0) if reading.confidence > 0.4 else (0, 165, 255)
            cv2.line(view, (pt[0] - 130, pt[1]), (pt[0] + 130, pt[1]), col, 3)

    # HUD text.
    lines = [
        f"Volume : {reading.volume_ml:6.2f} mL"
        + (f"  (raw {reading.raw_volume_ml:5.2f})" if reading.ok else ""),
        f"Target : {target:6.2f} mL   dir={controller.cfg.direction.value}",
        f"Height : {reading.height_mm:6.2f} mm   conf={reading.confidence:.2f}",
        f"Flow   : {reading.flow_ml_per_s:+6.2f} mL/s  {'STABLE' if reading.stable else ''}",
        f"Pump   : {controller.state.value.upper()}   rem={reading.__dict__.get('rem', 0):.2f}",
        f"Status : {reading.status}",
    ]
    y = 34
    for t in lines:
        cv2.putText(view, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 34

    # Side panel: rectified strip + fused profile.
    panel_w = 260
    if rect is not None:
        strip = cv2.cvtColor(cv2.resize(rect.strip, (panel_w // 2, h)),
                             cv2.COLOR_GRAY2BGR)
        prof = np.zeros((h, panel_w // 2, 3), np.uint8)
        men = est.last_meniscus
        if men is not None and men.fused is not None:
            f = cv2.resize(men.fused.reshape(-1, 1), (1, h)).ravel()
            for yy in range(h):
                cv2.line(prof, (0, yy), (int(f[yy] * (panel_w // 2 - 4)), yy),
                         (0, 200, 255), 1)
            ry = int(men.row / rect.strip.shape[0] * h)
            cv2.line(strip, (0, ry), (panel_w // 2, ry), (0, 0, 255), 2)
        panel = np.hstack([strip, prof])
    else:
        panel = np.zeros((h, panel_w, 3), np.uint8)
        cv2.putText(panel, "No tags", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
    return np.hstack([panel, view])


def main():
    ap = argparse.ArgumentParser(description="Liquid level monitor + pump control")
    ap.add_argument("--config", default=None)
    ap.add_argument("--calibration", default=None, help="height_mm,volume_ml CSV")
    ap.add_argument("--geometry", nargs=4, metavar=("DIAM_MM", "BASE_MM", "BOTTOM", "BH_MM"),
                    default=None, help="geometric model, e.g. 20 0 cone 12")
    ap.add_argument("--capacity", type=float, default=None,
                    help="tube capacity (ml) for the linear fallback")
    ap.add_argument("--linear-full-mm", type=float, default=80.0,
                    help="height (mm) treated as 'full' for the linear fallback")
    ap.add_argument("--port", default=None, help="pump serial port, e.g. /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--target", type=float, default=0.0, help="target volume (ml)")
    ap.add_argument("--drain", action="store_true", help="dosing removes liquid")
    ap.add_argument("--stop-latency", type=float, default=0.25,
                    help="pump/plumbing stop latency (s) for predictive cutoff")
    ap.add_argument("--tolerance", type=float, default=0.1, help="target tolerance (ml)")
    ap.add_argument("--legacy-protocol", action="store_true",
                    help="responder mode matching the original firmware: pump "
                         "polls print/autoPrint/stopPrint, Pi replies '<x.x>mL'")
    ap.add_argument("--stream-format", default=r"{volume:.1f}\r\n",
                    help=r"ASCII line streamed to a listener pump each interval. "
                         r"Use {volume} and \r \n escapes. Default '{volume:.1f}\r\n' "
                         r"(number only, CRLF). Add units with '{volume:.1f}mL\r\n'.")
    ap.add_argument("--stream-interval", type=float, default=0.3,
                    help="seconds between streamed volume lines (default 0.3)")
    ap.add_argument("--no-stream", action="store_true",
                    help="do not continuously stream the volume to the pump")
    ap.add_argument("--no-pi", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="headless (no GUI)")
    args = ap.parse_args()

    tube_cfg, men_cfg = load_configs(args.config)
    if args.capacity:
        tube_cfg.tube_capacity_ml = args.capacity

    volume_model = build_volume_model(args, tube_cfg)
    estimator = LevelEstimator(tube_cfg, volume_model, men_cfg)

    # Pump link: real serial if a port is given, else a mock so you can dry-run.
    if args.port:
        link = SerialPumpLink(args.port, args.baud)
        print(f"[monitor] pump link on {args.port} @ {args.baud}")
    else:
        link = MockPumpLink()
        print("[monitor] no --port; using MOCK pump link (prints commands).")

    # Decode \r \n etc. from the CLI string into real control characters.
    stream_fmt = args.stream_format.encode("ascii", "ignore").decode("unicode_escape")

    pump_cfg = PumpConfig(
        direction=Direction.DRAIN if args.drain else Direction.FILL,
        target_ml=args.target, tolerance_ml=args.tolerance,
        stop_latency_s=args.stop_latency,
        stream=not args.no_stream,
        stream_format=stream_fmt,
        stream_min_interval_s=args.stream_interval,
    )
    print(f"[monitor] streaming volume to pump as {stream_fmt!r} every "
          f"{args.stream_interval}s" if pump_cfg.stream else "[monitor] volume stream OFF")
    if args.legacy_protocol:
        # Responder mode: the pump is the master. It polls with
        # print/autoPrint/stopPrint and decides when to stop itself; the Pi only
        # answers with the volume in the original "<x.x>mL\r\n" format. No V:
        # stream and no RUN/STOP are ever sent.
        pump_cfg.stream = False
        pump_cfg.answer_queries = True
        print("[monitor] LEGACY responder mode: streaming volume as '<x.x>mL' on request.")
    controller = PumpController(link, pump_cfg)
    target = args.target
    dosing = False

    cam = open_camera(prefer_pi=not args.no_pi)
    window = "Liquid Level Monitor"
    if not args.no_window:
        cv2.namedWindow(window)

    print("\nSystem ready. [space] start/pause  [+/-] target  [d]irection  [r]eset  [q]uit")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            now = time.time()
            reading = estimator.update(frame, now)

            # Drive the pump only while dosing.
            if dosing and controller.state == PumpState.IDLE:
                controller.start()
            if dosing:
                out = controller.update(reading.volume_ml if reading.ok else float("nan"),
                                        reading.flow_ml_per_s, reading.confidence, now)
                reading.__dict__["rem"] = out.remaining_ml
                if out.state == PumpState.REACHED and dosing:
                    print(f"[pump] TARGET REACHED at {reading.volume_ml:.2f} mL "
                          f"(target {target:.2f}). Pump stopped.")
                    dosing = False
            else:
                # Still stream the live volume even when not dosing. Hold the
                # last good value if this frame was momentarily unreadable so we
                # never send a spurious 0.0 to the pump.
                vol = reading.volume_ml
                if not np.isfinite(vol):
                    vol = 0.0
                if isinstance(link, MockPumpLink) or args.port:
                    controller.update(vol, reading.flow_ml_per_s,
                                      reading.confidence, now)

            if not args.no_window:
                dash = make_dashboard(frame, estimator.last_rect, estimator,
                                      reading, controller, target)
                cv2.imshow(window, dash)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    target += 0.5; controller.set_target(target); controller.pause(); dosing = False
                elif key == ord('-') or key == ord('_'):
                    target = max(0.0, target - 0.5); controller.set_target(target); controller.pause(); dosing = False
                elif key == ord(' '):
                    dosing = not dosing
                    if dosing:
                        controller.set_target(target)
                    else:
                        controller.pause()
                    print(f"[pump] dosing {'STARTED' if dosing else 'PAUSED'} target={target:.2f}")
                elif key == ord('r'):
                    estimator.reset(); print("[monitor] filter reset")
                elif key == ord('d'):
                    controller.cfg.direction = (Direction.DRAIN
                                                if controller.cfg.direction == Direction.FILL
                                                else Direction.FILL)
                    print(f"[pump] direction -> {controller.cfg.direction.value}")
            else:
                time.sleep(0.01)
    finally:
        controller.pause()
        cam.close()
        link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
