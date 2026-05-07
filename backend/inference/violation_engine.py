"""Violation engine: turns per-frame detections into debounced violation events.

Architecture
------------
1. IoUMatcher.match_frame: associates PPE detections (Hardhat / NO-Hardhat /
   Safety Vest / NO-Safety Vest / Mask / NO-Mask) with each tracked Person and
   produces a PPEMatch per (person, frame).
2. WorkerStateBuffer: per-worker deque of recent PPEMatch observations within a
   sliding time window.
3. ViolationDetector: state machine that fires a ViolationEvent when a worker's
   non-compliance ratio exceeds `violation_threshold` for at least
   `min_violation_duration` seconds, and ends the event when the ratio falls
   below `recovery_threshold`.

Time semantics
--------------
All `timestamp` arguments are floats in *video seconds* (advances at 1/fps per
frame regardless of playback speed). ViolationEvent's `started_at`/`ended_at`
are wall-clock datetimes (so reports show when in the real session it
happened); `duration_seconds` is computed in the same clock as `started_at`
for consistency.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


# ---------- public types ----------


VIOLATION_TYPES: tuple[str, ...] = ("no_hardhat", "no_vest", "no_mask")
PPE_TO_VIOLATION_TYPE: dict[str, str] = {
    "hardhat": "no_hardhat",
    "vest": "no_vest",
    "mask": "no_mask",
}
VIOLATION_LABEL_TR: dict[str, str] = {
    "no_hardhat": "Baret yok",
    "no_vest": "Yelek yok",
    "no_mask": "Maske yok",
}


@dataclass
class PPEMatch:
    """Result of matching a person with their PPE items in one frame."""

    worker_id: int
    person_bbox: tuple[int, int, int, int]
    has_hardhat: Optional[bool] = None
    has_vest: Optional[bool] = None
    has_mask: Optional[bool] = None
    hardhat_confidence: float = 0.0
    vest_confidence: float = 0.0
    mask_confidence: float = 0.0
    frame_number: int = 0
    timestamp: float = 0.0


@dataclass
class ViolationEvent:
    """A confirmed violation lasting >= min_violation_duration."""

    event_id: str
    worker_id: int
    violation_type: str
    camera_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: float
    snapshot_path: Optional[str]
    confidence_avg: float
    is_active: bool

    # Internal accumulators (not serialized to dict by default)
    _conf_sum: float = field(default=0.0, repr=False)
    _conf_count: int = field(default=0, repr=False)
    _start_video_ts: float = field(default=0.0, repr=False)
    _last_video_ts: float = field(default=0.0, repr=False)

    def update_confidence(self, conf: float) -> None:
        if conf > 0:
            self._conf_sum += conf
            self._conf_count += 1
            self.confidence_avg = self._conf_sum / self._conf_count

    @property
    def violation_label_tr(self) -> str:
        return VIOLATION_LABEL_TR.get(self.violation_type, self.violation_type)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "worker_id": self.worker_id,
            "violation_type": self.violation_type,
            "violation_label_tr": self.violation_label_tr,
            "camera_id": self.camera_id,
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "ended_at": self.ended_at.isoformat(timespec="milliseconds") if self.ended_at else None,
            "duration_seconds": round(self.duration_seconds, 3),
            "snapshot_path": self.snapshot_path,
            "confidence_avg": round(self.confidence_avg, 3),
            "is_active": self.is_active,
        }


# ---------- IoU matcher ----------


_HEAD_FRAC = 0.30   # upper 30% of person bbox = head region
_TORSO_FRAC = (0.20, 0.80)  # rows 20% .. 80% = torso region
_FACE_FRAC = 0.20   # upper 20% = face region (mask)
_MIN_IOU_OR_CENTER = 0.05  # IoU threshold; we also accept "center inside region"


class IoUMatcher:
    """Spatial matching of persons to PPE items."""

    @staticmethod
    def calculate_iou(box1, box2) -> float:
        """Standard IoU. Both boxes as (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
        area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _person_region(person_bbox, region: str) -> tuple[float, float, float, float]:
        px1, py1, px2, py2 = person_bbox
        h = py2 - py1
        if region == "head":
            return (px1, py1, px2, py1 + _HEAD_FRAC * h)
        if region == "face":
            return (px1, py1, px2, py1 + _FACE_FRAC * h)
        if region == "torso":
            return (px1, py1 + _TORSO_FRAC[0] * h, px2, py1 + _TORSO_FRAC[1] * h)
        raise ValueError(f"Unknown region: {region}")

    @classmethod
    def is_inside_person_region(cls, ppe_bbox, person_bbox, region: str) -> bool:
        """Match if the PPE box's center lies inside the region OR IoU >= 0.05."""
        rx1, ry1, rx2, ry2 = cls._person_region(person_bbox, region)
        cx = (ppe_bbox[0] + ppe_bbox[2]) * 0.5
        cy = (ppe_bbox[1] + ppe_bbox[3]) * 0.5
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
        return cls.calculate_iou(ppe_bbox, (rx1, ry1, rx2, ry2)) >= _MIN_IOU_OR_CENTER

    def match_frame(
        self,
        persons: list[dict],
        hardhats: list[dict],
        no_hardhats: list[dict],
        vests: list[dict],
        no_vests: list[dict],
        masks: list[dict],
        no_masks: list[dict],
        frame_number: int,
        timestamp: float,
    ) -> list[PPEMatch]:
        """Return one PPEMatch per tracked person."""
        out: list[PPEMatch] = []
        for p in persons:
            tid = p.get("track_id")
            if tid is None:
                continue
            pbox = p["bbox"]
            has_h, conf_h = self._best_match(hardhats, pbox, region="head")
            no_h, no_conf_h = self._best_match(no_hardhats, pbox, region="head")
            has_v, conf_v = self._best_match(vests, pbox, region="torso")
            no_v, no_conf_v = self._best_match(no_vests, pbox, region="torso")
            has_m, conf_m = self._best_match(masks, pbox, region="face")
            no_m, no_conf_m = self._best_match(no_masks, pbox, region="face")

            hardhat_state, hardhat_conf = self._resolve(has_h, conf_h, no_h, no_conf_h)
            vest_state, vest_conf = self._resolve(has_v, conf_v, no_v, no_conf_v)
            mask_state, mask_conf = self._resolve(has_m, conf_m, no_m, no_conf_m)

            out.append(
                PPEMatch(
                    worker_id=int(tid),
                    person_bbox=tuple(pbox),
                    has_hardhat=hardhat_state,
                    has_vest=vest_state,
                    has_mask=mask_state,
                    hardhat_confidence=hardhat_conf,
                    vest_confidence=vest_conf,
                    mask_confidence=mask_conf,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
            )
        return out

    # ---- helpers ----

    def _best_match(self, candidates: list[dict], person_bbox, region: str) -> tuple[bool, float]:
        """Return (any_match, best_conf) for candidates overlapping person's region."""
        best_conf = 0.0
        any_match = False
        for c in candidates:
            if self.is_inside_person_region(c["bbox"], person_bbox, region):
                any_match = True
                if c["confidence"] > best_conf:
                    best_conf = float(c["confidence"])
        return any_match, best_conf

    @staticmethod
    def _resolve(
        positive_match: bool,
        positive_conf: float,
        negative_match: bool,
        negative_conf: float,
    ) -> tuple[Optional[bool], float]:
        """Decide compliance state when both Hardhat and NO-Hardhat may overlap.

        Precedence: positive (compliant) wins if confidences are within 0.1 — we
        prefer to give the worker the benefit of the doubt. Otherwise the higher
        confidence wins.
        """
        if positive_match and not negative_match:
            return True, positive_conf
        if negative_match and not positive_match:
            return False, negative_conf
        if not positive_match and not negative_match:
            return None, 0.0
        # Both matched (e.g., model double-detects a vest area):
        if positive_conf + 0.1 >= negative_conf:
            return True, positive_conf
        return False, negative_conf


# ---------- buffer ----------


class WorkerStateBuffer:
    """Sliding window of compliance observations per worker."""

    def __init__(self, window_seconds: float = 5.0, fps: float = 8.0) -> None:
        self.window_seconds = window_seconds
        self._fps = max(1.0, fps)
        # +1 so 20 observations at 10 fps actually span 2.0 s (0.0..2.0), not 1.9 s.
        self.window_size = max(2, int(round(window_seconds * fps)) + 1)
        self._buffers: dict[int, deque[PPEMatch]] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, match: PPEMatch) -> None:
        wid = match.worker_id
        if wid not in self._buffers:
            self._buffers[wid] = deque(maxlen=self.window_size)
        self._buffers[wid].append(match)
        self._last_seen[wid] = match.timestamp

    def _recent(self, worker_id: int) -> list[PPEMatch]:
        buf = self._buffers.get(worker_id)
        if not buf:
            return []
        now_ts = buf[-1].timestamp
        return [m for m in buf if now_ts - m.timestamp <= self.window_seconds]

    def get_violation_ratio(self, worker_id: int, ppe_type: str) -> float:
        recent = self._recent(worker_id)
        if not recent:
            return 0.0
        attr = {"hardhat": "has_hardhat", "vest": "has_vest", "mask": "has_mask"}[ppe_type]
        false_count = sum(1 for m in recent if getattr(m, attr) is False)
        return false_count / len(recent)

    def get_avg_violation_confidence(self, worker_id: int, ppe_type: str) -> float:
        recent = self._recent(worker_id)
        if not recent:
            return 0.0
        attr_state = {"hardhat": "has_hardhat", "vest": "has_vest", "mask": "has_mask"}[ppe_type]
        attr_conf = {"hardhat": "hardhat_confidence", "vest": "vest_confidence", "mask": "mask_confidence"}[ppe_type]
        confs = [getattr(m, attr_conf) for m in recent if getattr(m, attr_state) is False]
        return float(np.mean(confs)) if confs else 0.0

    def get_buffer_duration(self, worker_id: int) -> float:
        """Seconds covered by the buffer's observations.

        We add `1/fps` to the timestamp span so that N observations spaced at
        the nominal frame interval represent N/fps seconds (not (N-1)/fps).
        """
        buf = self._buffers.get(worker_id)
        if not buf:
            return 0.0
        if len(buf) < 2:
            return 1.0 / self._fps
        return buf[-1].timestamp - buf[0].timestamp + 1.0 / self._fps

    def cleanup_inactive(self, current_timestamp: float, max_age_seconds: float = 30.0) -> None:
        stale = [
            wid for wid, ts in self._last_seen.items()
            if current_timestamp - ts > max_age_seconds
        ]
        for wid in stale:
            self._buffers.pop(wid, None)
            self._last_seen.pop(wid, None)


# ---------- detector ----------


class ViolationDetector:
    """State machine that fires/ends ViolationEvents based on debounced ratios."""

    def __init__(
        self,
        min_violation_duration: float = 5.0,
        violation_threshold: float = 0.7,
        recovery_threshold: float = 0.3,
        fps: float = 8.0,
        camera_id: str = "CAM_01",
    ) -> None:
        self.min_violation_duration = min_violation_duration
        self.violation_threshold = violation_threshold
        # Hysteresis: recovery must be strictly lower than firing threshold,
        # otherwise events flicker on/off in the same frame.
        if recovery_threshold >= violation_threshold:
            adjusted = max(0.05, violation_threshold * 0.5)
            print(
                f"[uyarı] recovery_threshold ({recovery_threshold}) >= "
                f"violation_threshold ({violation_threshold}); auto-set to {adjusted:.2f}."
            )
            recovery_threshold = adjusted
        self.recovery_threshold = recovery_threshold
        self.camera_id = camera_id
        self._buffer = WorkerStateBuffer(window_seconds=min_violation_duration, fps=fps)
        self._active_violations: dict[tuple[int, str], ViolationEvent] = {}
        self._completed: list[ViolationEvent] = []

    @property
    def buffer(self) -> WorkerStateBuffer:
        return self._buffer

    def process_matches(
        self,
        matches: list[PPEMatch],
        frame_number: int,
        timestamp: float,
        frame_image: Optional[np.ndarray] = None,
        snapshot_saver=None,
    ) -> dict[str, list[ViolationEvent]]:
        """Process this frame's PPEMatches. Returns dict with 'new', 'ongoing', 'resolved'."""
        new_violations: list[ViolationEvent] = []
        ongoing: list[ViolationEvent] = []
        resolved: list[ViolationEvent] = []

        # 1) Add observations to the buffer
        for m in matches:
            self._buffer.update(m)

        # 2) For each worker observed this frame, evaluate each PPE type
        observed_workers: dict[int, PPEMatch] = {m.worker_id: m for m in matches}

        for wid, m in observed_workers.items():
            buf_dur = self._buffer.get_buffer_duration(wid)

            for ppe_type, has_attr, conf_attr in (
                ("hardhat", "has_hardhat", "hardhat_confidence"),
                ("vest", "has_vest", "vest_confidence"),
                ("mask", "has_mask", "mask_confidence"),
            ):
                violation_type = PPE_TO_VIOLATION_TYPE[ppe_type]
                key = (wid, violation_type)
                ratio = self._buffer.get_violation_ratio(wid, ppe_type)
                ev = self._active_violations.get(key)

                if ev is not None:
                    # Track ongoing event
                    ev._last_video_ts = timestamp
                    ev.duration_seconds = (datetime.now() - ev.started_at).total_seconds()
                    inst_conf = getattr(m, conf_attr) or 0.0
                    if getattr(m, has_attr) is False and inst_conf > 0:
                        ev.update_confidence(inst_conf)
                    if ratio < self.recovery_threshold:
                        ev.is_active = False
                        ev.ended_at = datetime.now()
                        ev.duration_seconds = (ev.ended_at - ev.started_at).total_seconds()
                        del self._active_violations[key]
                        self._completed.append(ev)
                        resolved.append(ev)
                    else:
                        ongoing.append(ev)
                else:
                    if (
                        buf_dur >= self.min_violation_duration
                        and ratio >= self.violation_threshold
                    ):
                        avg_conf = self._buffer.get_avg_violation_confidence(wid, ppe_type)
                        ev = ViolationEvent(
                            event_id=str(uuid.uuid4()),
                            worker_id=int(wid),
                            violation_type=violation_type,
                            camera_id=self.camera_id,
                            started_at=datetime.now(),
                            ended_at=None,
                            duration_seconds=0.0,
                            snapshot_path=None,
                            confidence_avg=avg_conf,
                            is_active=True,
                        )
                        ev._start_video_ts = timestamp
                        ev._last_video_ts = timestamp
                        if avg_conf > 0:
                            ev._conf_sum = avg_conf
                            ev._conf_count = 1
                        # Save evidence snapshot
                        if snapshot_saver is not None and frame_image is not None:
                            try:
                                ev.snapshot_path = snapshot_saver.save(
                                    frame=frame_image,
                                    person_bbox=m.person_bbox,
                                    worker_id=int(wid),
                                    violation_type=violation_type,
                                    timestamp=timestamp,
                                )
                            except Exception as e:
                                print(f"[uyarı] Snapshot kaydedilemedi: {e}")
                        self._active_violations[key] = ev
                        new_violations.append(ev)

        # 3) Periodic cleanup of stale workers (out of view > 30s)
        self._buffer.cleanup_inactive(timestamp)

        return {"new": new_violations, "ongoing": ongoing, "resolved": resolved}

    def get_active_violations(self) -> list[ViolationEvent]:
        return list(self._active_violations.values())

    def get_completed_violations(self) -> list[ViolationEvent]:
        return list(self._completed)

    def all_violations(self) -> list[ViolationEvent]:
        return self.get_completed_violations() + self.get_active_violations()

    def finalize(self, end_timestamp: float) -> None:
        """Close any still-active violations at end of session."""
        end_dt = datetime.now()
        for key, ev in list(self._active_violations.items()):
            ev.is_active = False
            ev.ended_at = end_dt
            ev.duration_seconds = (ev.ended_at - ev.started_at).total_seconds()
            self._completed.append(ev)
        self._active_violations.clear()

    # convenience for HUD
    def active_worker_violations(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for ev in self._active_violations.values():
            out.setdefault(ev.worker_id, []).append(ev.violation_type)
        return out
