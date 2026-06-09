"""core/postprocess.py - Layer 1: spatial detection cleanup (3 classes).

A STATELESS scrub of one frame's detections before they reach the violation
engine. Rewritten (not imported) from reference_old/post_processor.py for the
Person / Hardhat / Safety Vest scope. Temporal smoothing lives in Layer 2
(violations.WorkerStateBuffer), so this layer is purely spatial.

Filters, applied in order:
  1. class-specific confidence floors,
  2. Person-box geometric validation (area %, aspect ratio),
  3. same-class IoU de-duplication (keep the higher-confidence box).
"""
from __future__ import annotations

from typing import Iterable

# Confidence floors per class. Starting values taken from
# reference_old/post_processor.py (Person / Hardhat / Safety Vest = 0.50 each).
CONFIDENCE_FLOORS: dict[str, float] = {
    "Person": 0.50,
    "Hardhat": 0.50,
    "Safety Vest": 0.50,
}
DEFAULT_FLOOR = 0.50

# Person geometry, relative to frame area (from reference_old).
PERSON_MIN_AREA_RATIO = 0.005   # >= 0.5% of frame
PERSON_MAX_AREA_RATIO = 0.85    # <= 85% of frame
PERSON_MIN_ASPECT = 0.20        # h / w
PERSON_MAX_ASPECT = 5.00        # h / w

# Same-class duplicate removal (reference HARDHAT_DEDUP_IOU = 0.5).
DEDUP_IOU = 0.50


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def process(detections: list[dict], frame_w: int, frame_h: int) -> list[dict]:
    """Return a cleaned copy of `detections` for a single frame."""
    if not detections:
        return []
    dets = _apply_floors(detections)
    dets = _validate_persons(dets, frame_w, frame_h)
    dets = _dedupe_same_class(dets)
    return dets


def _apply_floors(detections: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for d in detections:
        floor = CONFIDENCE_FLOORS.get(d["class_name"], DEFAULT_FLOOR)
        if float(d["confidence"]) >= floor:
            out.append(d)
    return out


def _validate_persons(detections: Iterable[dict], frame_w: int, frame_h: int) -> list[dict]:
    frame_area = max(1, int(frame_w) * int(frame_h))
    out: list[dict] = []
    for d in detections:
        if d["class_name"] != "Person":
            out.append(d)
            continue
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        area_ratio = (w * h) / frame_area
        if area_ratio < PERSON_MIN_AREA_RATIO or area_ratio > PERSON_MAX_AREA_RATIO:
            continue
        aspect = h / w
        if aspect < PERSON_MIN_ASPECT or aspect > PERSON_MAX_ASPECT:
            continue
        out.append(d)
    return out


def _dedupe_same_class(detections: list[dict]) -> list[dict]:
    keep = [True] * len(detections)
    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue
            if detections[i]["class_name"] != detections[j]["class_name"]:
                continue
            if _iou(detections[i]["bbox"], detections[j]["bbox"]) < DEDUP_IOU:
                continue
            if detections[i]["confidence"] >= detections[j]["confidence"]:
                keep[j] = False
            else:
                keep[i] = False
                break
    return [d for d, k in zip(detections, keep) if k]


__all__ = [
    "process", "CONFIDENCE_FLOORS", "DEDUP_IOU",
    "PERSON_MIN_AREA_RATIO", "PERSON_MAX_AREA_RATIO",
    "PERSON_MIN_ASPECT", "PERSON_MAX_ASPECT",
]
