"""
Synthetic, hardware-free tests for the liquid-level pipeline.

Run with:  python -m pytest liquid_detection/tests/ -v
       or:  python liquid_detection/tests/test_liquid_level.py   (self-runs)

The image tests build synthetic tube strips (including one with graduation
marks) with a known ground-truth surface row, then assert the detector lands
within a sub-millimetre tolerance and that graduation ticks do not fool it.
"""

import os
import sys
import math

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquid_level import (  # noqa: E402
    MeniscusDetector, MeniscusConfig,
    CalibrationCurve, GeometricTube, LinearModel,
    AprilTagLocator, TubeConfig, LevelEstimator,
)
from pump_controller import (  # noqa: E402
    PumpController, PumpConfig, MockPumpLink, Direction, PumpState,
)


# ---------------------------------------------------------------------------
# Synthetic strip builders
# ---------------------------------------------------------------------------

def make_strip(h=512, w=160, surface_row=300, liquid_dark=True,
               with_marks=False, noise=3.0, seed=0):
    """
    Build a rectified-strip-like grayscale image.

    Air (above surface): brighter, higher texture (crisp background).
    Liquid (below surface): slightly darker, lower texture (refraction blur).
    Optionally overlay regular graduation tick marks across the FULL width
    (the hard case that fools a plain gradient detector).
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), np.float64)

    air_level = 180.0
    liq_level = 150.0 if liquid_dark else 205.0
    img[:surface_row] = air_level
    img[surface_row:] = liq_level

    # A thin dark meniscus band right at the surface (total internal reflection).
    band = 3
    img[max(0, surface_row - band):surface_row + band] -= 35.0

    # Texture: sharp vertical stripes above (air/background), blurred below.
    stripes = (np.sin(np.arange(w) * 0.7) * 18.0)
    img[:surface_row] += stripes[None, :]
    img[surface_row:] += (stripes[None, :] * 0.25)  # refraction-blurred

    if with_marks:
        # Full-width graduation ticks every ~40 px, above AND below the surface.
        for y in range(30, h - 30, 40):
            img[y - 1:y + 2, 10:w - 10] -= 60.0
            # tick numerals hint (small blob) - localized, not full width
            img[y - 4:y + 5, w // 2 - 6:w // 2 + 6] -= 25.0

    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

def test_detects_plain_surface():
    det = MeniscusDetector()
    for gt in (150, 250, 300, 400):
        strip = make_strip(surface_row=gt, seed=gt)
        res = det.detect(strip, mm_per_px_strip=0.2)
        assert res.found, f"surface not found for gt={gt}"
        err_px = abs(res.row - gt)
        assert err_px <= 6, f"gt={gt} got row={res.row:.1f} (err {err_px:.1f}px)"


def test_rejects_graduation_marks():
    """The meniscus must win over regularly spaced full-width tick marks."""
    det = MeniscusDetector()
    ok = 0
    trials = [140, 220, 305, 410]
    for gt in trials:
        strip = make_strip(surface_row=gt, with_marks=True, seed=gt + 7)
        res = det.detect(strip, mm_per_px_strip=0.2)
        # Accept if within ~1 tick spacing of ground truth (i.e. not locked onto
        # a random tick elsewhere).
        if res.found and abs(res.row - gt) <= 20:
            ok += 1
    assert ok >= 3, f"only {ok}/4 graduated cases detected near truth"


def test_subpixel_precision():
    """Sub-pixel output should track fractional shifts on average."""
    det = MeniscusDetector()
    errs = []
    for gt in range(200, 360, 13):
        strip = make_strip(surface_row=gt, seed=gt)
        res = det.detect(strip, mm_per_px_strip=0.2)
        if res.found:
            errs.append(abs(res.row - gt))
    assert np.mean(errs) < 3.0, f"mean sub-pixel err too high: {np.mean(errs):.2f}px"


# ---------------------------------------------------------------------------
# Volume model tests
# ---------------------------------------------------------------------------

def test_calibration_curve_interp_and_inverse():
    # A conical-bottom-ish curve: little volume per mm near the base.
    heights = [0, 5, 10, 20, 30, 40, 50]
    vols = [0, 1, 3, 10, 20, 33, 50]
    cal = CalibrationCurve(heights, vols)
    assert abs(cal.volume(0) - 0) < 1e-6
    assert abs(cal.volume(25) - 15) < 1e-6      # midpoint of 20->30 : 10->20
    assert abs(cal.volume(50) - 50) < 1e-6
    # Clamping outside range.
    assert cal.volume(-10) == 0
    assert cal.volume(999) == 50
    # Inverse round-trips.
    h = cal.height_for_volume(15)
    assert abs(cal.volume(h) - 15) < 1e-6


def test_calibration_csv_roundtrip(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    path = os.path.join(str(d), "cal.csv")
    cal = CalibrationCurve([0, 10, 20, 30], [0, 5, 20, 45])
    cal.to_csv(path)
    cal2 = CalibrationCurve.from_csv(path)
    for hh in (0, 7, 15, 28, 30):
        assert abs(cal.volume(hh) - cal2.volume(hh)) < 1e-6


def test_geometric_cone_monotonic_and_nonlinear():
    tube = GeometricTube(inner_diameter_mm=20, base_offset_mm=0,
                         bottom="cone", bottom_height_mm=10)
    v_all = [tube.volume(hh) for hh in range(0, 60, 2)]
    assert all(b >= a - 1e-9 for a, b in zip(v_all, v_all[1:])), "not monotonic"
    # Cone region grows slower than the cylinder region (nonlinear bottom).
    dv_bottom = tube.volume(5) - tube.volume(3)
    dv_top = tube.volume(45) - tube.volume(43)
    assert dv_bottom < dv_top, "cone bottom should add less volume per mm"
    # Inverse.
    h = tube.height_for_volume(10)
    assert abs(tube.volume(h) - 10) < 0.05


def test_linear_model():
    m = LinearModel(height_full_mm=100, capacity_ml=50)
    assert abs(m.volume(50) - 25) < 1e-6
    assert m.volume(-5) == 0
    assert abs(m.volume(200) - 50) < 1e-6


# ---------------------------------------------------------------------------
# AprilTag rectification: render tags, detect, check mm scale
# ---------------------------------------------------------------------------

def render_apriltag_scene(surface_frac=0.5):
    """
    Render a scene with two tag36h11 markers (ids 0 top, 1 bottom) and a tube
    column between them, with a surface at the given fraction from the top.
    Returns (frame_bgr, expected_height_mm_of_surface).
    """
    cfg = TubeConfig()
    dictionary = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict())
    H, W = 900, 500
    frame = np.full((H, W, 3), 210, np.uint8)

    tag_px = 120
    quiet = 24                       # white quiet-zone border (required for detection)
    cx = W // 2
    top_y, bot_y = 120, 700  # tag top-left y
    for tid, ty in ((0, top_y), (1, bot_y)):
        marker = cv2.aruco.generateImageMarker(dictionary, tid, tag_px)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        x0 = cx - tag_px // 2
        # White quiet zone.
        frame[ty - quiet:ty + tag_px + quiet, x0 - quiet:x0 + tag_px + quiet] = 255
        frame[ty:ty + tag_px, x0:x0 + tag_px] = marker

    # Tube column drawn ONLY in the gap between the tags (never over them).
    c_top_y = top_y + tag_px / 2
    c_bot_y = bot_y + tag_px / 2
    gap_top = top_y + tag_px          # just below the top tag
    gap_bot = bot_y                   # just above the bottom tag
    col_x0, col_x1 = cx - 40, cx + 40
    surface_y = int(gap_top + surface_frac * (gap_bot - gap_top))
    frame[gap_top:surface_y, col_x0:col_x1] = 185              # air
    frame[surface_y:gap_bot, col_x0:col_x1] = 150             # liquid
    frame[surface_y - 2:surface_y + 2, col_x0:col_x1] = 110    # meniscus band

    # Expected height above the bottom reference (inner edge of bottom tag).
    mm_per_px = cfg.tag_size_mm / tag_px
    height_mm = (gap_bot - surface_y) * mm_per_px
    return frame, height_mm


def test_apriltag_locate_and_scale():
    cfg = TubeConfig()
    loc = AprilTagLocator(cfg)
    frame, _ = render_apriltag_scene(0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = loc.locate(gray)
    assert rect is not None, "tags not located"
    # Tag side ~120 px -> mm scale sane.
    assert 90 < rect.tag_side_px < 150
    assert rect.strip.shape == (cfg.strip_height_px, cfg.strip_width_px)


def test_end_to_end_height_accuracy():
    """Full pipeline: rendered scene -> measured height within ~1.5 mm."""
    cfg = TubeConfig()
    loc = AprilTagLocator(cfg)
    det = MeniscusDetector()
    for frac in (0.35, 0.5, 0.65):
        frame, gt_mm = render_apriltag_scene(frac)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = loc.locate(gray)
        assert rect is not None
        men = det.detect(rect.strip, rect.mm_per_px_strip)
        assert men.found
        meas_mm = rect.strip_row_to_height_mm(men.row)
        err = abs(meas_mm - gt_mm)
        assert err < 2.5, f"frac={frac}: measured {meas_mm:.1f} vs {gt_mm:.1f} (err {err:.1f}mm)"


# ---------------------------------------------------------------------------
# Pump controller tests
# ---------------------------------------------------------------------------

def test_pump_predictive_stop_no_overshoot():
    """Simulate filling; the pump should stop before/at target, not past it."""
    link = MockPumpLink()
    cfg = PumpConfig(direction=Direction.FILL, target_ml=20.0,
                     tolerance_ml=0.1, stop_latency_s=0.3,
                     require_stable_frames=1, min_confidence=0.1,
                     stream=True)
    ctl = PumpController(link, cfg)
    ctl.set_target(20.0)

    vol = 5.0
    flow = 2.0          # ml/s
    dt = 0.1
    t = 0.0
    stopped_at = None
    for _ in range(400):
        out = ctl.update(vol, flow, confidence=1.0, timestamp=t)
        if out.state == PumpState.REACHED:
            stopped_at = vol
            break
        # Pump keeps adding while RUN was the command.
        if out.command == cfg.cmd_run:
            vol += flow * dt
        t += dt
    assert stopped_at is not None, "never reached target"
    # Predictive stop cuts off ~ flow*latency before target; with momentum the
    # true landing is near target. Command threshold must be <= target.
    assert stopped_at <= 20.0 + 0.05, f"overshot: stopped at {stopped_at:.2f}"
    assert stopped_at >= 20.0 - (flow * 0.3) - 0.5, f"stopped too early: {stopped_at:.2f}"


def test_pump_latches_and_streams():
    link = MockPumpLink()
    cfg = PumpConfig(target_ml=10.0, require_stable_frames=1,
                     stream=True, stream_min_interval_s=0.0)
    ctl = PumpController(link, cfg)
    ctl.set_target(10.0)
    # Already over target -> should stop and latch.
    out = ctl.update(11.0, 0.0, confidence=1.0, timestamp=0.0)
    assert out.state == PumpState.REACHED
    # Streaming happened.
    assert any(s.startswith("V:") for s in link.sent)
    # Stays latched even if it dips slightly.
    out2 = ctl.update(9.9, 0.0, confidence=1.0, timestamp=0.2)
    assert out2.state == PumpState.REACHED


def test_pump_legacy_query_protocol():
    link = MockPumpLink()
    ctl = PumpController(link, PumpConfig(stream=False))
    link.feed("print")
    ctl.update(7.53, 0.0, confidence=1.0, timestamp=0.0)
    # Must match the ORIGINAL firmware format exactly: one decimal, 'mL', CRLF.
    assert "7.5mL\r\n" in link.sent, f"bad legacy reply format: {link.sent}"


def test_pump_legacy_autoprint_stream_and_stop():
    link = MockPumpLink()
    ctl = PumpController(link, PumpConfig(stream=False, auto_query_interval_s=0.5))
    # Pump requests continuous streaming (the flag is set; first stream comes
    # one interval later, matching the original rate-limited behaviour).
    link.feed("autoPrint")
    ctl.update(3.0, 0.0, 1.0, timestamp=0.0)
    # First stream after the interval elapses.
    ctl.update(3.1, 0.0, 1.0, timestamp=0.6)
    assert "3.1mL\r\n" in link.sent
    # Within the interval -> no new stream line.
    n = len(link.sent)
    ctl.update(3.15, 0.0, 1.0, timestamp=0.7)
    assert len(link.sent) == n, "auto-stream not rate-limited"
    # After another interval -> streams again.
    ctl.update(3.2, 0.0, 1.0, timestamp=1.2)
    assert "3.2mL\r\n" in link.sent
    # Pump says stop -> streaming ceases.
    link.feed("stopPrint")
    ctl.update(3.4, 0.0, 1.0, timestamp=2.0)
    assert not any("3.4" in s for s in link.sent), "did not stop auto-stream"


# ---------------------------------------------------------------------------
# Self-runner
# ---------------------------------------------------------------------------

def _run_all():
    fns = [g for name, g in sorted(globals().items())
           if name.startswith("test_") and callable(g)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
