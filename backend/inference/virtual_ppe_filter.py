"""Virtual PPE filter — synthetic hardhat / vest overlays driven by MediaPipe Pose.

Used by the live test app so we can validate YOLO PPE detection without owning a
real hardhat. The shapes try to look enough like real PPE for YOLO to fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


HARDHAT_BGR = (0, 200, 255)         # high-vis yellow
HARDHAT_BRIM_BGR = (0, 130, 200)    # darker yellow for brim
VEST_BGR = (0, 100, 255)            # high-vis orange
VEST_STRIPE_BGR = (220, 220, 220)   # silver/grey reflective tape
OUTLINE_BGR = (10, 10, 10)


# MediaPipe Pose landmark indices we use
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class _Pt:
    x: int
    y: int
    visible: bool


class VirtualPPEFilter:
    """Draws a synthetic hardhat / vest on the body detected by MediaPipe Pose."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,
            smooth_landmarks=True,
        )

    def close(self) -> None:
        self.pose.close()

    def __enter__(self) -> "VirtualPPEFilter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------- public API ----------

    def apply_hardhat(self, frame: np.ndarray, enabled: bool = True) -> np.ndarray:
        if not enabled:
            return frame
        lm = self._detect(frame)
        if lm is not None:
            h, w = frame.shape[:2]
            self._draw_hardhat(frame, lm, w, h)
        return frame

    def apply_vest(self, frame: np.ndarray, enabled: bool = True) -> np.ndarray:
        if not enabled:
            return frame
        lm = self._detect(frame)
        if lm is not None:
            h, w = frame.shape[:2]
            self._draw_vest(frame, lm, w, h)
        return frame

    def process(self, frame: np.ndarray, hardhat: bool = True, vest: bool = True) -> np.ndarray:
        """Apply both filters with a single pose detection pass (efficient path)."""
        if not (hardhat or vest):
            return frame
        lm = self._detect(frame)
        if lm is None:
            return frame
        h, w = frame.shape[:2]
        if vest:
            self._draw_vest(frame, lm, w, h)
        if hardhat:
            self._draw_hardhat(frame, lm, w, h)
        return frame

    # ---------- internals ----------

    def _detect(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        return result.pose_landmarks

    def _pt(self, lm, idx: int, w: int, h: int, vis_threshold: float = 0.4) -> _Pt:
        l = lm.landmark[idx]
        return _Pt(int(l.x * w), int(l.y * h), l.visibility >= vis_threshold)

    def _draw_hardhat(self, frame: np.ndarray, lm, w: int, h: int) -> None:
        nose = self._pt(lm, NOSE, w, h)
        le = self._pt(lm, LEFT_EAR, w, h)
        re = self._pt(lm, RIGHT_EAR, w, h)
        if not (nose.visible and le.visible and re.visible):
            return

        cx = (le.x + re.x) // 2
        cy = (le.y + re.y) // 2
        head_width = max(40, int(math.hypot(re.x - le.x, re.y - le.y)))
        ear_to_nose = math.hypot(nose.x - cx, nose.y - cy)

        hat_half_w = int(head_width * 0.85)
        hat_h = int(head_width * 0.55)
        dome_cy = cy - int(ear_to_nose * 0.4)
        angle_deg = math.degrees(math.atan2(re.y - le.y, re.x - le.x))

        # Filled dome (upper half ellipse)
        cv2.ellipse(frame, (cx, dome_cy), (hat_half_w, hat_h), angle_deg, 180, 360, HARDHAT_BGR, -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, dome_cy), (hat_half_w, hat_h), angle_deg, 180, 360, OUTLINE_BGR, 2, cv2.LINE_AA)
        # Brim — a flatter ellipse at the dome's base
        brim_w = int(hat_half_w * 1.15)
        brim_h = max(4, int(hat_h * 0.18))
        cv2.ellipse(frame, (cx, dome_cy), (brim_w, brim_h), angle_deg, 0, 360, HARDHAT_BRIM_BGR, -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, dome_cy), (brim_w, brim_h), angle_deg, 0, 360, OUTLINE_BGR, 2, cv2.LINE_AA)

    def _draw_vest(self, frame: np.ndarray, lm, w: int, h: int) -> None:
        ls = self._pt(lm, LEFT_SHOULDER, w, h)
        rs = self._pt(lm, RIGHT_SHOULDER, w, h)
        lh = self._pt(lm, LEFT_HIP, w, h)
        rh = self._pt(lm, RIGHT_HIP, w, h)
        if not (ls.visible and rs.visible and lh.visible and rh.visible):
            return

        shoulder_w = max(1, int(math.hypot(rs.x - ls.x, rs.y - ls.y)))
        shoulder_inset = int(0.06 * shoulder_w)
        torso = np.array(
            [
                [ls.x - shoulder_inset, ls.y],
                [rs.x + shoulder_inset, rs.y],
                [rh.x, rh.y],
                [lh.x, lh.y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [torso], VEST_BGR, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [torso], isClosed=True, color=OUTLINE_BGR, thickness=2, lineType=cv2.LINE_AA)

        # Two horizontal reflective stripes at ~40% and ~62% torso height
        for fraction in (0.40, 0.62):
            half = 0.05
            l_top = self._lerp_xy(ls, lh, fraction - half)
            l_bot = self._lerp_xy(ls, lh, fraction + half)
            r_top = self._lerp_xy(rs, rh, fraction - half)
            r_bot = self._lerp_xy(rs, rh, fraction + half)
            stripe = np.array([l_top, r_top, r_bot, l_bot], dtype=np.int32)
            cv2.fillPoly(frame, [stripe], VEST_STRIPE_BGR, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [stripe], isClosed=True, color=OUTLINE_BGR, thickness=1, lineType=cv2.LINE_AA)

    @staticmethod
    def _lerp_xy(a: _Pt, b: _Pt, t: float) -> tuple[int, int]:
        return (int(a.x + (b.x - a.x) * t), int(a.y + (b.y - a.y) * t))
