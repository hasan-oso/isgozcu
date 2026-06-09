# REFERENCE ONLY — DO NOT IMPORT
# Copied verbatim from old project (backend/inference/post_processor.py) for guidance during the clean rewrite.

"""Post-processing layer that scrubs raw v3 detections before they reach the
violation engine.

Four filters, applied in order:

    1. Class-specific confidence floors (stricter than the model's global
       conf threshold).
    2. Geometric validation for `Person` boxes — rejects tiny / huge /
       weirdly-shaped boxes that are typically shadow / chair / wall
       false positives.
    3. Hardhat / NO-Hardhat de-duplication — when both classes fire on the
       same head with high IoU, keep only the higher-confidence one.
    4. Per-tracker temporal smoothing for Safety Vest / NO-Safety Vest —
       majority vote over a sliding window suppresses single-frame flicker.

All filters operate on the project's `list[StandardDetection]` format (the
dicts produced by `DetectorAdapter.detect_and_track`). track_id, when
present, is used by the vest smoother. Detections without a track_id pass
through smoothing unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from backend.inference.detector_adapter import StandardDetection


class DetectionPostProcessor:
    """Filter / smooth raw model detections to suppress v3's known failure modes."""

    # Class-specific confidence floors — stricter than the model's global
    # threshold so only confident predictions reach the violation engine.
    CONFIDENCE_FLOORS: dict[str, float] = {
        "Hardhat": 0.50,
        "NO-Hardhat": 0.55,
        "Safety Vest": 0.50,        # higher: vest is the flicker problem
        "NO-Safety Vest": 0.55,     # highest: most prone to false positives
        "Mask": 0.60,
        "NO-Mask": 0.65,
        "Person": 0.50,             # higher: shadow / chair false positives
        "Safety Cone": 0.50,
        "machinery": 0.55,
        "vehicle": 0.55,
    }

    # Geometric validation for `Person` boxes (relative to frame area).
    PERSON_MIN_AREA_RATIO = 0.005   # at least 0.5% of frame
    PERSON_MAX_AREA_RATIO = 0.85    # not more than 85%
    PERSON_MIN_ASPECT = 0.20        # h/w > 0.2 (not too wide)
    PERSON_MAX_ASPECT = 5.00        # h/w < 5.0 (not too thin)

    # Vest temporal smoothing window (frames).
    VEST_SMOOTHING_WINDOW = 30      # ~1 second at 30 fps
    VEST_SMOOTHING_THRESHOLD = 0.65 # 65% of recent frames must agree

    # Hardhat / NO-Hardhat IoU threshold for deduplication.
    HARDHAT_DEDUP_IOU = 0.5

    def __init__(
        self,
        class_names: Iterable[str],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> None:
        self.class_names = list(class_names)
        self.name_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.update_frame_size(frame_width, frame_height)
        # Per-tracker rolling history of vest verdicts ("vest" / "no_vest" / None)
        self._vest_history: dict[int, deque[str]] = {}

    def update_frame_size(self, width: int, height: int) -> None:
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.frame_area = max(1, self.frame_width * self.frame_height)

    # ---------- public entry point ----------

    def process(
        self, detections: list[StandardDetection]
    ) -> list[StandardDetection]:
        """Return a filtered + smoothed copy of `detections`."""
        if not detections:
            return detections

        # Step 1: class-specific confidence floors
        detections = self._apply_confidence_floors(detections)
        if not detections:
            return detections

        # Step 2: Person geometric validation
        detections = self._validate_person_boxes(detections)
        if not detections:
            return detections

        # Step 3: dedupe overlapping Hardhat / NO-Hardhat predictions
        detections = self._dedupe_hardhat_pairs(detections)
        if not detections:
            return detections

        # Step 4: per-tracker vest smoothing (uses track_id when available)
        detections = self._smooth_vest_temporal(detections)

        return detections

    # ---------- step 1 ----------

    def _apply_confidence_floors(
        self, detections: list[StandardDetection]
    ) -> list[StandardDetection]:
        out: list[StandardDetection] = []
        for d in detections:
            floor = self.CONFIDENCE_FLOORS.get(d["class_name"], 0.50)
            if float(d["confidence"]) >= floor:
                out.append(d)
        return out

    # ---------- step 2 ----------

    def _validate_person_boxes(
        self, detections: list[StandardDetection]
    ) -> list[StandardDetection]:
        out: list[StandardDetection] = []
        for d in detections:
            if d["class_name"] != "Person":
                out.append(d)
                continue
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            area_ratio = (w * h) / self.frame_area
            if area_ratio < self.PERSON_MIN_AREA_RATIO:
                continue
            if area_ratio > self.PERSON_MAX_AREA_RATIO:
                continue
            aspect = h / w
            if aspect < self.PERSON_MIN_ASPECT or aspect > self.PERSON_MAX_ASPECT:
                continue
            out.append(d)
        return out

    # ---------- step 3 ----------

    def _dedupe_hardhat_pairs(
        self, detections: list[StandardDetection]
    ) -> list[StandardDetection]:
        """When Hardhat and NO-Hardhat overlap on the same head, keep the
        higher-confidence one only."""
        keep = [True] * len(detections)
        for i, di in enumerate(detections):
            if not keep[i]:
                continue
            ci = di["class_name"]
            if ci not in ("Hardhat", "NO-Hardhat"):
                continue
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                dj = detections[j]
                cj = dj["class_name"]
                if cj not in ("Hardhat", "NO-Hardhat"):
                    continue
                if cj == ci:
                    continue  # same class — leave both, NMS isn't our job here
                iou = _iou(di["bbox"], dj["bbox"])
                if iou < self.HARDHAT_DEDUP_IOU:
                    continue
                if di["confidence"] >= dj["confidence"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is gone, no need to compare it to anything else
        return [d for d, k in zip(detections, keep) if k]

    # ---------- step 4 ----------

    def _smooth_vest_temporal(
        self, detections: list[StandardDetection]
    ) -> list[StandardDetection]:
        """Per-tracker majority vote on Vest / NO-Safety-Vest classification.

        Builds the current frame's vest verdict per track_id, appends to the
        rolling history, then retroactively suppresses contradictory
        detections in the current frame using the smoothed verdict.

        Detections with no track_id are passed through unchanged — we have no
        history to smooth against.
        """
        VEST_NAME = "Safety Vest"
        NO_VEST_NAME = "NO-Safety Vest"

        # 1. record what each tracker showed in this frame.
        per_tracker_verdict: dict[int, str] = {}
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            cn = d["class_name"]
            if cn == VEST_NAME:
                per_tracker_verdict[int(tid)] = "vest"
            elif cn == NO_VEST_NAME:
                per_tracker_verdict[int(tid)] = "no_vest"

        # 2. update the rolling histories.
        for tid, verdict in per_tracker_verdict.items():
            buf = self._vest_history.get(tid)
            if buf is None:
                buf = deque(maxlen=self.VEST_SMOOTHING_WINDOW)
                self._vest_history[tid] = buf
            buf.append(verdict)

        # 3. compute a smoothed verdict per tracker (or None when ambiguous).
        smoothed: dict[int, Optional[str]] = {}
        for tid, hist in self._vest_history.items():
            if len(hist) < 5:
                # Not enough history — accept whatever the current frame says.
                continue
            vest_n = sum(1 for s in hist if s == "vest")
            no_vest_n = sum(1 for s in hist if s == "no_vest")
            total = vest_n + no_vest_n
            if total == 0:
                continue
            vest_ratio = vest_n / total
            if vest_ratio >= self.VEST_SMOOTHING_THRESHOLD:
                smoothed[tid] = "vest"
            elif (1.0 - vest_ratio) >= self.VEST_SMOOTHING_THRESHOLD:
                smoothed[tid] = "no_vest"
            else:
                smoothed[tid] = None  # truly ambiguous — drop both this frame

        # 4. apply: drop contradicting / ambiguous vest detections this frame.
        out: list[StandardDetection] = []
        for d in detections:
            cn = d["class_name"]
            tid = d.get("track_id")
            if cn not in (VEST_NAME, NO_VEST_NAME) or tid is None:
                out.append(d)
                continue
            verdict = smoothed.get(int(tid))
            if verdict is None and int(tid) in self._vest_history and len(self._vest_history[int(tid)]) >= 5:
                # history present but inconclusive — drop both vest classes
                continue
            if verdict == "vest" and cn == NO_VEST_NAME:
                continue
            if verdict == "no_vest" and cn == VEST_NAME:
                continue
            out.append(d)
        return out

    # ---------- session helpers ----------

    def reset_tracker(self, tracker_id: int) -> None:
        """Drop a single tracker's vest history (e.g. when ByteTrack drops the track)."""
        self._vest_history.pop(int(tracker_id), None)

    def reset_all(self) -> None:
        """Drop all vest histories (e.g. on video loop boundary or session reset)."""
        self._vest_history.clear()


# ---------- helpers ----------


def _iou(box1, box2) -> float:
    """Standard IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / max(union, 1e-6)
