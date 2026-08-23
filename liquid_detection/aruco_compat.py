"""
aruco_compat.py -- AprilTag/ArUco detection that works across OpenCV versions.

OpenCV changed the aruco API in 4.7:
  * new (>=4.7): cv2.aruco.ArucoDetector(dict, params).detectMarkers(gray)
  * old (<4.7):  cv2.aruco.detectMarkers(gray, dict, parameters=params)

This module hides that difference behind one object with a .detectMarkers(gray)
method, so the rest of the code is version-agnostic.
"""

from __future__ import annotations

import cv2


def _get_dictionary(dict_attr: str):
    key = getattr(cv2.aruco, dict_attr)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(key)
    # Very old API
    return cv2.aruco.Dictionary_get(key)


def _make_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        try:
            params = cv2.aruco.DetectorParameters()
        except TypeError:
            params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters_create()
    # Sub-pixel corner refinement, if available (improves the mm scale).
    try:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    except Exception:
        pass

    # Robustness tuning: a wide adaptive-threshold window range copes with
    # uneven lighting / backlit tags, and a low min perimeter lets smaller or
    # farther tags still be found. Each guarded so it is a no-op on odd builds.
    for attr, val in (
        ("adaptiveThreshWinSizeMin", 3),
        ("adaptiveThreshWinSizeMax", 45),
        ("adaptiveThreshWinSizeStep", 6),
        ("adaptiveThreshConstant", 7),
        ("minMarkerPerimeterRate", 0.02),
        ("maxMarkerPerimeterRate", 4.0),
        ("polygonalApproxAccuracyRate", 0.06),
        ("minCornerDistanceRate", 0.03),
    ):
        try:
            setattr(params, attr, val)
        except Exception:
            pass
    return params


class ArucoDetectorCompat:
    """Uniform detector: .detectMarkers(gray) -> (corners, ids, rejected)."""

    def __init__(self, dict_attr: str = "DICT_APRILTAG_36h11"):
        self.dictionary = _get_dictionary(dict_attr)
        self.params = _make_params()
        self._new = hasattr(cv2.aruco, "ArucoDetector")
        if self._new:
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)
        else:
            self._detector = None

    def detectMarkers(self, gray):
        if self._new:
            return self._detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)


def make_aruco_detector(dict_attr: str = "DICT_APRILTAG_36h11") -> ArucoDetectorCompat:
    return ArucoDetectorCompat(dict_attr)
