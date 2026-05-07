"""Saves cropped person images as evidence for violation events.

Output structure:

    data/snapshots/violations/
      2026-05-07/
        no_hardhat_W007_143218_412.jpg
        no_vest_W014_143245_001.jpg
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from backend.inference.text_render import put_text_outlined
from backend.inference.violation_engine import VIOLATION_LABEL_TR


_PAD = 60          # pixels of padding around the person crop
_OVERLAY_H = 88    # height of the translucent header band


class SnapshotSaver:
    """Crops + annotates + saves violation evidence images."""

    def __init__(self, base_dir: str | Path = "data/snapshots/violations") -> None:
        self.base_dir = Path(base_dir)

    def save(
        self,
        frame: np.ndarray,
        person_bbox: tuple[int, int, int, int],
        worker_id: int,
        violation_type: str,
        timestamp: float,
    ) -> str:
        """Save a cropped+annotated snapshot. Returns the absolute path."""
        # Use wall clock for the on-image timestamp + filename — that's what
        # operators care about when reviewing evidence.
        dt = datetime.now()
        day_dir = self.base_dir / dt.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox
        x1c = max(0, int(x1) - _PAD)
        y1c = max(0, int(y1) - _PAD)
        x2c = min(w, int(x2) + _PAD)
        y2c = min(h, int(y2) + _PAD)
        crop = frame[y1c:y2c, x1c:x2c].copy()

        # Translucent header band
        if crop.shape[0] > _OVERLAY_H:
            band = crop[:_OVERLAY_H, :]
            overlay = np.zeros_like(band)
            cv2.addWeighted(overlay, 0.6, band, 0.4, 0, band)

        label_tr = VIOLATION_LABEL_TR.get(violation_type, violation_type)
        time_str = dt.strftime("%H:%M:%S")
        put_text_outlined(crop, label_tr, (12, 28), scale=0.9, fg=(80, 80, 255), thickness=2)
        put_text_outlined(crop, f"İşçi: W{worker_id:03d}", (12, 54), scale=0.6, fg=(255, 255, 255))
        put_text_outlined(crop, f"Zaman: {time_str}", (12, 76), scale=0.6, fg=(255, 255, 255))

        ms = int((timestamp - int(timestamp)) * 1000)
        ts_str = dt.strftime("%H%M%S") + f"_{ms:03d}"
        path = day_dir / f"{violation_type}_W{worker_id:03d}_{ts_str}.jpg"
        cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return str(path.resolve())
