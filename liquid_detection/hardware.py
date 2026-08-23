"""
hardware.py -- Camera abstraction and JSON config loading.

Keeps all the platform-specific bits (Picamera2 vs. USB webcam) in one place so
the monitor and calibration tools stay clean. Importing this module never
requires a camera; the camera is only opened when you call open_camera().
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import cv2

from liquid_level import TubeConfig, MeniscusConfig


class Camera:
    """Uniform frame source over Picamera2 or cv2.VideoCapture (BGR frames)."""

    def __init__(self, width=1280, height=720, prefer_pi=True):
        self.picam2 = None
        self.cap = None
        if prefer_pi:
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                cfg = self.picam2.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"})
                self.picam2.configure(cfg)
                # A fixed, moderate exposure/gain gives far more repeatable
                # readings than full auto that hunts frame-to-frame.
                self.picam2.start()
            except Exception as e:  # noqa: BLE001
                print(f"[camera] Picamera2 unavailable ({e}); using webcam.")
                self.picam2 = None

        if self.picam2 is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open any camera (Pi or USB).")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        if self.picam2 is not None:
            arr = self.picam2.capture_array()
            # Picamera2's "RGB888" already stores channels in B,G,R order --
            # exactly what OpenCV treats as BGR -- so DO NOT colour-convert.
            # Converting here swaps red/blue (blue-tinted image) and, because the
            # grayscale for AprilTag detection is derived from these channels,
            # also hurts tag detection.
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]     # drop alpha if an XRGB format sneaks in
            return arr
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()


def open_camera(width=1280, height=720, prefer_pi=True) -> Camera:
    return Camera(width, height, prefer_pi)


# ---------------------------------------------------------------------------
# Config file (JSON) <-> dataclasses
# ---------------------------------------------------------------------------

def load_configs(path: Optional[str]):
    """
    Load TubeConfig + MeniscusConfig from a JSON file. Missing keys fall back to
    the dataclass defaults, so a partial file is fine. Returns (tube, meniscus).
    """
    tube = TubeConfig()
    men = MeniscusConfig()
    if not path or not os.path.exists(path):
        return tube, men
    with open(path) as fh:
        data = json.load(fh)
    for k, v in data.get("tube", {}).items():
        if hasattr(tube, k):
            setattr(tube, k, v)
    for k, v in data.get("meniscus", {}).items():
        if hasattr(men, k):
            setattr(men, k, v)
    return tube, men


def save_configs(path: str, tube: TubeConfig, men: MeniscusConfig,
                 extra: Optional[dict] = None):
    data = {"tube": asdict(tube), "meniscus": asdict(men)}
    if extra:
        data.update(extra)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
