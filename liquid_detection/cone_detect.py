"""
cone_detect.py -- Automatically find where a tube's conical bottom begins.

Idea: the tube's side walls are vertical (constant width) through the
cylindrical body and start converging in the conical bottom. So if we measure
the inner width of the tube at every row, the profile is a flat plateau (the
cylinder) that bends and shrinks toward the bottom (the cone). The row where it
leaves the plateau is where the cone begins.

The walls are near-vertical edges, so they show up in the horizontal gradient
|d/dx| -- and, conveniently, horizontal graduation marks do NOT (those are
horizontal edges), so printed scales don't interfere with wall finding.

detect_cone_fraction(roi_gray) returns the cone height as a fraction of the ROI
height (0..~0.6), suitable to pass straight to the cone+cylinder volume model,
or None if it cannot measure a reliable profile.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2


def measure_width_profile(roi_gray: np.ndarray) -> Optional[np.ndarray]:
    """Return a per-row tube width (in px), NaN where a wall pair isn't found."""
    h, w = roi_gray.shape[:2]
    if h < 20 or w < 12:
        return None
    gx = np.abs(cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3))
    gx = cv2.GaussianBlur(gx, (1, 9), 0)   # stabilise along the walls
    half = w // 2
    widths = np.full(h, np.nan, dtype=np.float64)
    for y in range(h):
        row = gx[y]
        if row.max() < 1e-6:
            continue
        left = row[:half]
        right = row[half:]
        if left.max() <= 0 or right.max() <= 0:
            continue
        # Strongest vertical edge in each half = that side's wall.
        l = int(np.argmax(left))
        r = int(np.argmax(right)) + half
        if r - l > 2:
            widths[y] = r - l
    return widths


def detect_cone_fraction(roi_gray: np.ndarray, plateau_ratio: float = 0.97,
                         min_valid_rows: int = 25) -> Optional[float]:
    """
    Estimate cone_frac = (cone height) / (ROI height) from the tube silhouette.

    Row 0 is the top of the ROI (cylinder), the last row is the bottom (cone
    tip). We find the cylinder's plateau width, then scan up from the bottom to
    the first row that reaches ``plateau_ratio`` of it -- that row is the top of
    the cone.
    """
    widths = measure_width_profile(roi_gray)
    if widths is None:
        return None
    h = len(widths)
    valid = ~np.isnan(widths)
    if valid.sum() < min_valid_rows:
        return None

    idx = np.arange(h)
    wprof = np.interp(idx, idx[valid], widths[valid])
    wprof = cv2.GaussianBlur(wprof.reshape(-1, 1), (1, 15), 0).ravel()

    # Cylinder (plateau) width: robust upper value over the top ~60% of rows.
    top_region = wprof[: max(5, int(h * 0.6))]
    w_cyl = float(np.percentile(top_region, 75))
    if w_cyl <= 1:
        return None

    thresh = plateau_ratio * w_cyl
    cone_start = None
    for y in range(h - 1, -1, -1):        # scan up from the bottom
        if wprof[y] >= thresh:
            cone_start = y
            break
    if cone_start is None or cone_start >= h - 1:
        return 0.0                         # no narrowing found -> flat bottom

    cone_frac = (h - cone_start) / float(h)
    return float(np.clip(cone_frac, 0.0, 0.6))
