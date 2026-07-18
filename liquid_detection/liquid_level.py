"""
liquid_level.py -- Core liquid-level detection library.

This module contains the pure computer-vision + math pipeline for reading the
liquid level in a test tube and converting it to a volume. It has NO camera,
serial, or GUI dependencies (only numpy + opencv), so it can be imported and
unit-tested off the hardware.

Pipeline overview
-----------------
1. AprilTagLocator      -> find the two holder tags and build a rectifying
                           affine transform + a millimetre scale.
2. rectify_strip()      -> warp the tube column into an upright, fixed-size
                           grayscale strip (perspective / rotation invariant).
3. MeniscusDetector     -> locate the liquid surface inside the strip with a
                           fused, multi-cue score that rejects graduation
                           marks, with sub-pixel refinement + a confidence.
4. VolumeModel          -> map the measured surface height (mm above the bottom
                           tag reference) to a volume in millilitres. A measured
                           calibration curve is the recommended, most accurate
                           option and handles conical bottoms + any tube shape.
5. LevelEstimator       -> ties it together and adds temporal filtering
                           (confidence gating, median + EMA, flow-rate estimate,
                           stability detection) to produce a smooth reading for
                           closed-loop pump control.

Why this is more accurate/precise than a plain gradient detector:
  * Affine rectification removes camera tilt and gives a consistent, physical
    (mm) scale from the known 16 mm tag size, instead of measuring in raw,
    perspective-distorted pixels.
  * The detector fuses a *region step* cue (sustained brightness change across
    the surface) and a *texture step* cue (crisp markings/air above vs.
    refraction-blurred liquid below) with the edge cue. Isolated graduation
    ticks produce a big edge spike but almost no region/texture step, so they
    are suppressed -- this is what lets it work on graduated tubes.
  * Sub-pixel parabolic peak refinement + temporal median/EMA cut jitter to a
    fraction of a millimetre.
  * A measured calibration curve linearises the printed scale directly, so the
    conical bottom and the exact tube geometry no longer introduce error.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TubeConfig:
    """Physical + geometric configuration for one setup."""

    # --- AprilTag geometry ---
    tag_family: str = "DICT_APRILTAG_36h11"
    tag_size_mm: float = 16.0          # inner black square size (detector value)
    top_tag_id: int = 0                # optional: which id is the top tag
    bottom_tag_id: int = 1             # optional: which id is the bottom tag
    use_tag_ids: bool = False          # if True, use ids above; else sort by y

    # --- Rectified strip ---
    strip_height_px: int = 512         # rows in the rectified strip
    strip_width_px: int = 160          # columns in the rectified strip
    # Half-width of the sampled strip as a multiple of the tag side length.
    # 0.7 keeps the strip inside the tube walls for a typical holder.
    width_to_tag_ratio: float = 0.75

    # --- Volume ---
    tube_capacity_ml: float = 50.0     # only used by linear/geometric fallback

    def aruco_dict(self):
        return getattr(cv2.aruco, self.tag_family)


# ---------------------------------------------------------------------------
# AprilTag location + rectification
# ---------------------------------------------------------------------------

@dataclass
class RectifyResult:
    """Everything needed to interpret the rectified strip."""
    strip: np.ndarray                  # HxW uint8 grayscale, upright tube column
    affine: np.ndarray                 # 2x3 image->strip transform
    mm_per_px_strip: float             # vertical mm per strip row
    axis_len_mm: float                 # tag-center to tag-center distance in mm
    top_center: np.ndarray             # (x,y) image px of top tag center
    bottom_center: np.ndarray          # (x,y) image px of bottom tag center
    tag_side_px: float                 # mean tag side length in image px

    def strip_row_to_height_mm(self, row: float) -> float:
        """
        Convert a strip row (0 = top reference, H = bottom reference) into a
        height in mm *above the bottom tag reference*. This is the quantity the
        VolumeModel is calibrated against, and it is invariant to strip_height_px.
        """
        h = self.strip.shape[0]
        frac_from_bottom = 1.0 - (row / float(h))
        return frac_from_bottom * self.axis_len_mm


class AprilTagLocator:
    """Detects the two holder tags and builds a rectifying transform."""

    def __init__(self, cfg: TubeConfig):
        self.cfg = cfg
        dictionary = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict())
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel corner refinement dramatically improves the mm scale and the
        # rectification stability -- cheap and worth it.
        try:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, gray: np.ndarray):
        """Return (corners, ids) as given by the ArUco detector."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def locate(self, gray: np.ndarray) -> Optional[RectifyResult]:
        """
        Find the tags and produce a RectifyResult, or None if not enough tags.
        """
        corners, ids = self.detect(gray)
        if ids is None or len(ids) < 2:
            return None

        ids = ids.flatten()
        tags = {}
        for c, i in zip(corners, ids):
            pts = c[0].astype(np.float64)          # 4x2
            center = pts.mean(axis=0)
            # Mean side length from the 4 edges.
            side = np.mean([np.linalg.norm(pts[(k + 1) % 4] - pts[k]) for k in range(4)])
            tags[int(i)] = {"center": center, "side": side, "pts": pts}

        # Pick the two reference tags.
        if self.cfg.use_tag_ids and self.cfg.top_tag_id in tags and self.cfg.bottom_tag_id in tags:
            top = tags[self.cfg.top_tag_id]
            bot = tags[self.cfg.bottom_tag_id]
            # Make sure "top" is actually higher in the image.
            if top["center"][1] > bot["center"][1]:
                top, bot = bot, top
        else:
            ordered = sorted(tags.values(), key=lambda t: t["center"][1])
            top, bot = ordered[0], ordered[-1]

        return self._build_rectify(gray, top, bot)

    def _build_rectify(self, gray, top, bot) -> Optional[RectifyResult]:
        cfg = self.cfg
        c_top = top["center"]
        c_bot = bot["center"]

        axis = c_bot - c_top
        axis_len_px = float(np.linalg.norm(axis))
        if axis_len_px < 5:
            return None
        a_hat = axis / axis_len_px
        # Right direction is exactly perpendicular to the tube axis, so the
        # rectified strip is aligned to the physical centreline.
        right_hat = np.array([a_hat[1], -a_hat[0]])

        tag_side_px = 0.5 * (top["side"] + bot["side"])
        mm_per_px = cfg.tag_size_mm / tag_side_px

        # Reference the INNER edges of the tags (the edge facing the other tag)
        # rather than their centres, so the rectified strip is purely the tube
        # column between the markers and contains no tag pixels to lock onto.
        p_top = c_top + a_hat * (tag_side_px / 2.0)
        p_bot = c_bot - a_hat * (tag_side_px / 2.0)
        axis_len_px = float(np.linalg.norm(p_bot - p_top))
        if axis_len_px < 5:
            return None
        axis_len_mm = axis_len_px * mm_per_px

        half_w_px = cfg.width_to_tag_ratio * tag_side_px

        # Source triangle in the image, dest triangle in the strip.
        src = np.float32([
            p_top - right_hat * half_w_px,   # top-left
            p_top + right_hat * half_w_px,   # top-right
            p_bot - right_hat * half_w_px,   # bottom-left
        ])
        dst = np.float32([
            [0, 0],
            [cfg.strip_width_px, 0],
            [0, cfg.strip_height_px],
        ])
        affine = cv2.getAffineTransform(src, dst)
        strip = cv2.warpAffine(
            gray, affine, (cfg.strip_width_px, cfg.strip_height_px),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        mm_per_px_strip = axis_len_mm / cfg.strip_height_px
        return RectifyResult(
            strip=strip, affine=affine, mm_per_px_strip=mm_per_px_strip,
            axis_len_mm=axis_len_mm, top_center=c_top, bottom_center=c_bot,
            tag_side_px=tag_side_px,
        )


# ---------------------------------------------------------------------------
# Meniscus detection
# ---------------------------------------------------------------------------

@dataclass
class MeniscusConfig:
    # Fraction of the strip width kept for analysis (drop wall reflections).
    center_width_frac: float = 0.72
    # Ignore this fraction of rows at the very top/bottom (tag edge artefacts).
    row_margin_frac: float = 0.06
    # Band half-height (in mm) used by the region/texture step filters. This is
    # set larger than the graduation-tick spacing so isolated ticks average out.
    step_band_mm: float = 6.0
    step_gap_mm: float = 0.8           # dead-zone around the candidate row
    # Cue weights in the fused score (normalised cues, so these are relative).
    w_edge: float = 0.34
    w_region: float = 0.40
    w_texture: float = 0.26
    # Search window (mm) around the coarse region estimate to refine the edge.
    refine_window_mm: float = 4.0
    # Minimum fused prominence for a confident detection.
    min_confidence: float = 0.18
    use_clahe: bool = True


@dataclass
class MeniscusResult:
    row: float                         # sub-pixel strip row of the surface
    confidence: float                  # 0..1
    found: bool
    # Debug profiles (row-indexed), useful for visualisation/tuning.
    edge: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    texture: Optional[np.ndarray] = None
    fused: Optional[np.ndarray] = None


def _norm01(x: np.ndarray) -> np.ndarray:
    """Robust 0..1 normalisation using 5th/95th percentiles."""
    x = x.astype(np.float64)
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _band_step(profile: np.ndarray, band: int, gap: int) -> np.ndarray:
    """
    Matched step filter on a 1-D profile: |stat(above band) - stat(below band)|,
    where ``stat`` is the MEDIAN of each band.

    A median (rather than a mean) makes the filter respond to *sustained* level
    changes -- the air/liquid boundary -- while ignoring thin dark notches such
    as a single graduation tick or the meniscus's own thin reflection line
    (both dark above and below alike). That notch-robustness is what lets the
    detector survive graduated tubes.
    """
    n = len(profile)
    out = np.zeros(n, dtype=np.float64)
    if band < 1:
        return out
    prof = profile.astype(np.float64)

    def med_range(a, b):
        a = max(0, a); b = min(n, b)
        if b <= a:
            return np.nan
        return np.median(prof[a:b])

    for y in range(n):
        above = med_range(y - gap - band, y - gap)
        below = med_range(y + gap, y + gap + band)
        if np.isnan(above) or np.isnan(below):
            out[y] = 0.0
        else:
            out[y] = abs(above - below)
    return out


class MeniscusDetector:
    """Locates the liquid surface inside a rectified grayscale strip."""

    def __init__(self, cfg: Optional[MeniscusConfig] = None):
        self.cfg = cfg or MeniscusConfig()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, strip: np.ndarray, mm_per_px_strip: float) -> MeniscusResult:
        cfg = self.cfg
        h, w = strip.shape[:2]
        if h < 16 or w < 8:
            return MeniscusResult(row=0.0, confidence=0.0, found=False)

        img = strip
        if cfg.use_clahe:
            img = self._clahe.apply(img)

        # Keep the central width band (avoid tube walls / reflections).
        cw = max(4, int(w * cfg.center_width_frac))
        x0 = (w - cw) // 2
        band_img = img[:, x0:x0 + cw].astype(np.float64)

        # --- 1-D profiles over rows ---
        # Brightness profile: mean intensity per row.
        p = band_img.mean(axis=1)
        # Light vertical smoothing to stabilise derivatives.
        p_s = cv2.GaussianBlur(p.reshape(-1, 1), (1, 5), 0).ravel()

        # Edge cue: magnitude of the vertical derivative of brightness.
        edge = np.abs(np.gradient(p_s))

        # Texture profile: mean horizontal-gradient energy per row. Crisp
        # markings / sharp background (air side) read high; refraction-blurred
        # liquid side reads lower -> the *step* between them marks the surface.
        gx = cv2.Sobel(band_img, cv2.CV_64F, 1, 0, ksize=3)
        tex = np.abs(gx).mean(axis=1)
        tex_s = cv2.GaussianBlur(tex.reshape(-1, 1), (1, 7), 0).ravel()

        band = max(2, int(round(cfg.step_band_mm / max(mm_per_px_strip, 1e-6))))
        gap = max(1, int(round(cfg.step_gap_mm / max(mm_per_px_strip, 1e-6))))

        region = _band_step(p_s, band, gap)      # brightness region step
        texture = _band_step(tex_s, band, gap)   # texture region step

        # Mask out top/bottom margins (tag artefacts).
        m = max(1, int(h * cfg.row_margin_frac))
        for arr in (edge, region, texture):
            arr[:m] = 0.0
            arr[h - m:] = 0.0

        e_n = _norm01(edge)
        r_n = _norm01(region)
        t_n = _norm01(texture)

        # Coarse localisation from the region+texture cues (tick-robust). The
        # median step filter yields a flat plateau centred on the true surface,
        # so use the centroid of the near-max plateau rather than argmax (which
        # would bias to the plateau's leading edge).
        coarse_score = cfg.w_region * r_n + cfg.w_texture * t_n
        if coarse_score.max() <= 1e-6:
            return MeniscusResult(row=float(h) / 2, confidence=0.0, found=False,
                                  edge=e_n, region=r_n, texture=t_n,
                                  fused=coarse_score)
        peak_val = coarse_score.max()
        near = np.where(coarse_score >= 0.9 * peak_val)[0]
        coarse_y = int(round(near.mean())) if len(near) else int(np.argmax(coarse_score))

        # Refine the exact surface line with the edge cue in a small window.
        win = max(2, int(round(cfg.refine_window_mm / max(mm_per_px_strip, 1e-6))))
        lo = max(m, coarse_y - win)
        hi = min(h - m, coarse_y + win + 1)
        fused = cfg.w_edge * e_n + cfg.w_region * r_n + cfg.w_texture * t_n
        local = fused[lo:hi]
        if len(local) == 0:
            best = coarse_y
            row_sub = float(coarse_y)
        else:
            best = lo + int(np.argmax(local))
            # Weighted centroid over the window: averages symmetric double-edges
            # (e.g. the two sides of the thin meniscus band) onto the true
            # surface centre and yields a naturally sub-pixel estimate.
            wl = np.clip(local - local.min(), 0, None) ** 2
            if wl.sum() > 1e-9:
                idx = np.arange(lo, hi)
                row_sub = float((idx * wl).sum() / wl.sum())
            else:
                row_sub = self._subpixel(fused, best)

        # Confidence: prominence of the fused peak over its robust baseline,
        # combined with agreement between the region and edge cues.
        peak = fused[best]
        baseline = np.median(fused[fused > 0]) if np.any(fused > 0) else 0.0
        prominence = (peak - baseline)
        conf = float(np.clip(prominence / 0.5, 0.0, 1.0))
        # Down-weight if the region cue (the tick-robust one) is weak here.
        conf *= float(np.clip(r_n[best] * 1.5, 0.2, 1.0))

        found = conf >= cfg.min_confidence
        return MeniscusResult(
            row=row_sub, confidence=conf, found=found,
            edge=e_n, region=r_n, texture=t_n, fused=fused,
        )

    @staticmethod
    def _subpixel(f: np.ndarray, i: int) -> float:
        """Parabolic interpolation of the peak at index i for sub-pixel row."""
        if i <= 0 or i >= len(f) - 1:
            return float(i)
        a, b, c = f[i - 1], f[i], f[i + 1]
        denom = (a - 2 * b + c)
        if abs(denom) < 1e-9:
            return float(i)
        delta = 0.5 * (a - c) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        return float(i) + delta


# ---------------------------------------------------------------------------
# Volume models
# ---------------------------------------------------------------------------

class VolumeModel:
    """Base interface: height (mm above bottom reference) -> volume (ml)."""

    def volume(self, height_mm: float) -> float:
        raise NotImplementedError

    def height_for_volume(self, volume_ml: float) -> float:
        raise NotImplementedError


class CalibrationCurve(VolumeModel):
    """
    Piecewise-linear (monotone) calibration built from measured points.

    This is the recommended, most accurate model: it directly linearises the
    tube's printed scale and absorbs the conical bottom, the meniscus offset,
    and any optical bias -- for whatever tube you calibrated against.

    Points are (height_mm, volume_ml). Height is millimetres of the surface
    above the bottom tag reference, as returned by
    RectifyResult.strip_row_to_height_mm().
    """

    def __init__(self, heights_mm, volumes_ml):
        h = np.asarray(heights_mm, dtype=np.float64)
        v = np.asarray(volumes_ml, dtype=np.float64)
        order = np.argsort(h)
        self.h = h[order]
        self.v = v[order]
        if len(self.h) < 2:
            raise ValueError("CalibrationCurve needs at least 2 points")
        if not np.all(np.diff(self.h) > 0):
            # Merge/deduplicate identical heights by averaging their volumes.
            uniq_h, idx = np.unique(self.h, return_inverse=True)
            uniq_v = np.array([self.v[idx == k].mean() for k in range(len(uniq_h))])
            self.h, self.v = uniq_h, uniq_v

    def volume(self, height_mm: float) -> float:
        # Clamp to the calibrated range (do not extrapolate wildly).
        return float(np.interp(height_mm, self.h, self.v,
                               left=self.v[0], right=self.v[-1]))

    def height_for_volume(self, volume_ml: float) -> float:
        # v is monotone-increasing with h for a sane calibration.
        if np.all(np.diff(self.v) > 0):
            return float(np.interp(volume_ml, self.v, self.h,
                                   left=self.h[0], right=self.h[-1]))
        # Fallback: nearest height.
        idx = int(np.argmin(np.abs(self.v - volume_ml)))
        return float(self.h[idx])

    @classmethod
    def from_csv(cls, path: str) -> "CalibrationCurve":
        heights, vols = [], []
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                try:
                    hh = float(row[0]); vv = float(row[1])
                except (ValueError, IndexError):
                    continue  # skip header/garbage lines
                heights.append(hh); vols.append(vv)
        return cls(heights, vols)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["# height_mm", "volume_ml"])
            for hh, vv in zip(self.h, self.v):
                writer.writerow([f"{hh:.4f}", f"{vv:.4f}"])


class GeometricTube(VolumeModel):
    """
    Physical model: a vertical cylinder of known inner diameter with an
    optional conical or hemispherical bottom, referenced to the bottom tag.

    Parameters
    ----------
    inner_diameter_mm : inner bore of the tube
    base_offset_mm    : height of the tube's *inner base* above the bottom tag
                        reference (measure once with a ruler; can be negative)
    bottom            : 'flat', 'cone', or 'hemisphere'
    bottom_height_mm  : axial height of the conical/round bottom section
    """

    def __init__(self, inner_diameter_mm: float, base_offset_mm: float = 0.0,
                 bottom: str = "flat", bottom_height_mm: float = 0.0):
        self.r = inner_diameter_mm / 2.0
        self.base = base_offset_mm
        self.bottom = bottom
        self.bh = bottom_height_mm

    def _volume_above_base(self, y_mm: float) -> float:
        """Volume (ml) for a fill height y_mm above the tube's inner base."""
        if y_mm <= 0:
            return 0.0
        r = self.r
        mm3_to_ml = 1e-3
        if self.bottom == "flat" or self.bh <= 0:
            return np.pi * r * r * y_mm * mm3_to_ml
        if self.bottom == "cone":
            # Cone tip at base, opening to full radius at bottom_height.
            if y_mm <= self.bh:
                rr = r * (y_mm / self.bh)
                vol = (1.0 / 3.0) * np.pi * rr * rr * y_mm
            else:
                cone = (1.0 / 3.0) * np.pi * r * r * self.bh
                vol = cone + np.pi * r * r * (y_mm - self.bh)
            return vol * mm3_to_ml
        if self.bottom == "hemisphere":
            R = r  # hemisphere radius == tube radius
            if y_mm <= R:
                # spherical cap of height y
                vol = np.pi * y_mm * y_mm * (R - y_mm / 3.0)
            else:
                hemi = (2.0 / 3.0) * np.pi * R ** 3
                vol = hemi + np.pi * r * r * (y_mm - R)
            return vol * mm3_to_ml
        raise ValueError(f"unknown bottom type {self.bottom}")

    def volume(self, height_mm: float) -> float:
        return self._volume_above_base(height_mm - self.base)

    def height_for_volume(self, volume_ml: float) -> float:
        # Numeric inverse via bisection over a generous range.
        lo, hi = 0.0, 1000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.volume(mid + self.base) < volume_ml:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi) + self.base


class LinearModel(VolumeModel):
    """Simplest fallback: volume linear in height between two reference points."""

    def __init__(self, height_full_mm: float, capacity_ml: float,
                 height_empty_mm: float = 0.0):
        self.h0 = height_empty_mm
        self.h1 = height_full_mm
        self.cap = capacity_ml

    def volume(self, height_mm: float) -> float:
        if self.h1 <= self.h0:
            return 0.0
        frac = (height_mm - self.h0) / (self.h1 - self.h0)
        return float(np.clip(frac, 0.0, 1.0) * self.cap)

    def height_for_volume(self, volume_ml: float) -> float:
        frac = np.clip(volume_ml / self.cap, 0.0, 1.0)
        return self.h0 + frac * (self.h1 - self.h0)


# ---------------------------------------------------------------------------
# Temporal estimator
# ---------------------------------------------------------------------------

@dataclass
class LevelReading:
    ok: bool                           # a usable measurement this frame
    volume_ml: float = float("nan")    # filtered volume
    raw_volume_ml: float = float("nan")
    height_mm: float = float("nan")
    confidence: float = 0.0
    flow_ml_per_s: float = 0.0         # signed rate of change (filtered)
    stable: bool = False               # reading has settled
    status: str = ""                   # human-readable state
    strip_row: float = float("nan")


class LevelEstimator:
    """
    Full per-frame estimator with temporal filtering.

    Combines AprilTagLocator + MeniscusDetector + VolumeModel and adds:
      * confidence gating (drop unreliable frames),
      * outlier rejection + median smoothing (jitter/robustness),
      * an EMA for a smooth continuous signal,
      * a flow-rate estimate for predictive pump stopping,
      * a stability flag once the reading settles.
    """

    def __init__(self, cfg: TubeConfig, volume_model: VolumeModel,
                 meniscus_cfg: Optional[MeniscusConfig] = None,
                 smoothing_window: int = 9,
                 ema_alpha: float = 0.35,
                 outlier_ml: float = 4.0,
                 stable_ml: float = 0.15):
        self.cfg = cfg
        self.locator = AprilTagLocator(cfg)
        self.detector = MeniscusDetector(meniscus_cfg)
        self.model = volume_model
        self.vol_hist = deque(maxlen=smoothing_window)
        self.time_hist = deque(maxlen=smoothing_window)
        self.ema_alpha = ema_alpha
        self.outlier_ml = outlier_ml
        self.stable_ml = stable_ml
        self._ema = None
        self.last_rect: Optional[RectifyResult] = None
        self.last_meniscus: Optional[MeniscusResult] = None

    def reset(self):
        self.vol_hist.clear()
        self.time_hist.clear()
        self._ema = None

    def update(self, frame_bgr: np.ndarray, timestamp: float) -> LevelReading:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr

        rect = self.locator.locate(gray)
        self.last_rect = rect
        if rect is None:
            return LevelReading(ok=False, status="No tags (need 2)",
                                volume_ml=self._ema if self._ema is not None else float("nan"))

        men = self.detector.detect(rect.strip, rect.mm_per_px_strip)
        self.last_meniscus = men
        if not men.found:
            return LevelReading(ok=False, status="No surface", confidence=men.confidence,
                                volume_ml=self._ema if self._ema is not None else float("nan"))

        height_mm = rect.strip_row_to_height_mm(men.row)
        raw_vol = self.model.volume(height_mm)

        # Outlier rejection vs. the current smoothed estimate.
        if self._ema is not None and abs(raw_vol - self._ema) > self.outlier_ml \
                and len(self.vol_hist) >= 3 and men.confidence < 0.6:
            return LevelReading(ok=False, status="Rejected outlier",
                                raw_volume_ml=raw_vol, confidence=men.confidence,
                                volume_ml=self._ema, height_mm=height_mm,
                                strip_row=men.row)

        self.vol_hist.append(raw_vol)
        self.time_hist.append(timestamp)
        median_vol = float(np.median(self.vol_hist))

        # EMA on top of the median for a smooth continuous output.
        if self._ema is None:
            self._ema = median_vol
        else:
            self._ema += self.ema_alpha * (median_vol - self._ema)

        flow = self._estimate_flow()
        stable = self._is_stable()

        return LevelReading(
            ok=True, volume_ml=self._ema, raw_volume_ml=raw_vol, height_mm=height_mm,
            confidence=men.confidence, flow_ml_per_s=flow, stable=stable,
            status="Locked", strip_row=men.row,
        )

    def _estimate_flow(self) -> float:
        """Least-squares slope of volume vs. time over the history (ml/s)."""
        if len(self.vol_hist) < 3:
            return 0.0
        t = np.asarray(self.time_hist, dtype=np.float64)
        v = np.asarray(self.vol_hist, dtype=np.float64)
        t = t - t[0]
        if t[-1] - t[0] < 1e-6:
            return 0.0
        A = np.vstack([t, np.ones_like(t)]).T
        slope, _ = np.linalg.lstsq(A, v, rcond=None)[0]
        return float(slope)

    def _is_stable(self) -> bool:
        if len(self.vol_hist) < self.vol_hist.maxlen:
            return False
        return float(np.std(self.vol_hist)) < self.stable_ml
