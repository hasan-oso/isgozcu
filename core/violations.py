"""core/violations.py - Layers 2+3: association, temporal smoothing, state machine.

Rewritten (not imported) from reference_old/violation_engine.py for the 3-class
scope. Because the engine only sees POSITIVE classes (Person / Hardhat /
Safety Vest), a violation is inferred from the ABSENCE of associated PPE:

    person with no hardhat over the head  -> helmet_missing
    person with no vest over the torso    -> vest_missing

Layer 2 (WorkerStateBuffer): per-track sliding window; a majority vote over the
window (~min_violation_duration seconds of frames) suppresses single-frame flicker.
Layer 3 (ViolationDetector): debounce (must persist >= min_violation_duration) +
hysteresis (fire at >= violation_threshold, resolve only below recovery_threshold).
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.labels_tr import tr

# ppe key -> violation key (the violation key is also a labels_tr key)
PPE_TO_VIOLATION = {"hardhat": "helmet_missing", "vest": "vest_missing"}

# person-box sub-regions as fractions of height (from reference_old).
_HEAD_FRAC = 0.30            # top 30% = head region (hardhat)
_TORSO_FRAC = (0.20, 0.80)   # rows 20%..80% = torso region (vest)
_MIN_IOU_OR_CENTER = 0.05    # accept "center inside region" OR IoU >= this


@dataclass
class PPEMatch:
    """Per-person PPE association result for one frame."""
    worker_id: int
    person_bbox: tuple[int, int, int, int]
    person_confidence: float = 0.0
    has_hardhat: bool = False
    has_vest: bool = False
    hardhat_confidence: float = 0.0
    vest_confidence: float = 0.0
    frame_number: int = 0
    timestamp: float = 0.0


@dataclass
class ViolationEvent:
    """A confirmed violation (persisted >= min_violation_duration)."""
    event_id: str
    worker_id: int
    violation_type: str               # helmet_missing | vest_missing
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: float
    confidence_avg: float
    is_active: bool
    _conf_sum: float = field(default=0.0, repr=False)
    _conf_n: int = field(default=0, repr=False)

    @property
    def label_tr(self) -> str:
        return tr(self.violation_type)

    def add_conf(self, c: float) -> None:
        if c > 0:
            self._conf_sum += c
            self._conf_n += 1
            self.confidence_avg = self._conf_sum / self._conf_n

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "worker_id": self.worker_id,
            "violation_type": self.violation_type,
            "label_tr": self.label_tr,
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "ended_at": self.ended_at.isoformat(timespec="milliseconds") if self.ended_at else None,
            "duration_seconds": round(self.duration_seconds, 3),
            "confidence_avg": round(self.confidence_avg, 3),
            "is_active": self.is_active,
        }


class IoUMatcher:
    """Associate hardhats with heads and vests with torsos."""

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _region(person_bbox, region: str):
        px1, py1, px2, py2 = person_bbox
        h = py2 - py1
        if region == "head":
            return (px1, py1, px2, py1 + _HEAD_FRAC * h)
        if region == "torso":
            return (px1, py1 + _TORSO_FRAC[0] * h, px2, py1 + _TORSO_FRAC[1] * h)
        raise ValueError(f"unknown region: {region}")

    @classmethod
    def associated(cls, ppe_bbox, person_bbox, region: str) -> bool:
        rx1, ry1, rx2, ry2 = cls._region(person_bbox, region)
        cx = (ppe_bbox[0] + ppe_bbox[2]) * 0.5
        cy = (ppe_bbox[1] + ppe_bbox[3]) * 0.5
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
        return cls.iou(ppe_bbox, (rx1, ry1, rx2, ry2)) >= _MIN_IOU_OR_CENTER

    @classmethod
    def _best(cls, ppe_list, person_bbox, region) -> tuple[bool, float]:
        found, best = False, 0.0
        for d in ppe_list:
            if cls.associated(d["bbox"], person_bbox, region):
                found = True
                if d["confidence"] > best:
                    best = float(d["confidence"])
        return found, best

    def match_frame(self, persons, hardhats, vests, frame_number: int, timestamp: float) -> list[PPEMatch]:
        """One PPEMatch per tracked person. Absence of PPE -> has_* = False."""
        out: list[PPEMatch] = []
        for p in persons:
            tid = p.get("track_id")
            if tid is None:
                continue
            pbox = tuple(p["bbox"])
            has_h, conf_h = self._best(hardhats, pbox, "head")
            has_v, conf_v = self._best(vests, pbox, "torso")
            out.append(PPEMatch(
                worker_id=int(tid), person_bbox=pbox,
                person_confidence=float(p.get("confidence", 0.0)),
                has_hardhat=has_h, has_vest=has_v,
                hardhat_confidence=conf_h, vest_confidence=conf_v,
                frame_number=frame_number, timestamp=timestamp,
            ))
        return out


class WorkerStateBuffer:
    """Layer 2: per-track sliding window + majority vote over the window."""

    def __init__(self, window_seconds: float = 8.0, fps: float = 8.0) -> None:
        self.window_seconds = window_seconds
        self._fps = max(1.0, fps)
        # +1 so N observations spaced at 1/fps actually span N/fps seconds.
        self.window_size = max(2, int(round(window_seconds * fps)) + 1)
        self._buf: dict[int, deque] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, m: PPEMatch) -> None:
        self._buf.setdefault(m.worker_id, deque(maxlen=self.window_size)).append(m)
        self._last_seen[m.worker_id] = m.timestamp

    def _recent(self, wid: int) -> list[PPEMatch]:
        buf = self._buf.get(wid)
        if not buf:
            return []
        now = buf[-1].timestamp
        return [m for m in buf if now - m.timestamp <= self.window_seconds]

    def missing_ratio(self, wid: int, ppe: str) -> float:
        """Fraction of recent frames where the PPE was NOT associated."""
        recent = self._recent(wid)
        if not recent:
            return 0.0
        attr = {"hardhat": "has_hardhat", "vest": "has_vest"}[ppe]
        missing = sum(1 for m in recent if getattr(m, attr) is False)
        return missing / len(recent)

    def avg_person_conf_when_missing(self, wid: int, ppe: str) -> float:
        recent = self._recent(wid)
        attr = {"hardhat": "has_hardhat", "vest": "has_vest"}[ppe]
        confs = [m.person_confidence for m in recent if getattr(m, attr) is False]
        return sum(confs) / len(confs) if confs else 0.0

    def buffer_duration(self, wid: int) -> float:
        buf = self._buf.get(wid)
        if not buf:
            return 0.0
        if len(buf) < 2:
            return 1.0 / self._fps
        return buf[-1].timestamp - buf[0].timestamp + 1.0 / self._fps

    def cleanup(self, now_ts: float, max_age: float = 30.0) -> None:
        """Drop workers unseen for > max_age seconds (default 30s)."""
        stale = [wid for wid, ts in self._last_seen.items() if now_ts - ts > max_age]
        for wid in stale:
            self._buf.pop(wid, None)
            self._last_seen.pop(wid, None)


class ViolationDetector:
    """Layer 3: debounce + hysteresis state machine."""

    def __init__(
        self,
        min_violation_duration: float = 8.0,
        violation_threshold: float = 0.85,
        recovery_threshold: float = 0.3,
        fps: float = 8.0,
    ) -> None:
        self.min_violation_duration = min_violation_duration
        self.violation_threshold = violation_threshold
        # Hysteresis: recovery must be strictly below firing, else events flicker.
        if recovery_threshold >= violation_threshold:
            recovery_threshold = max(0.05, violation_threshold * 0.5)
        self.recovery_threshold = recovery_threshold
        self.buffer = WorkerStateBuffer(window_seconds=min_violation_duration, fps=fps)
        self._active: dict[tuple[int, str], ViolationEvent] = {}
        self._completed: list[ViolationEvent] = []

    def process_matches(self, matches: list[PPEMatch], frame_number: int, timestamp: float) -> dict:
        """Returns {'new': [...], 'ongoing': [...], 'resolved': [...]} for this frame."""
        new: list[ViolationEvent] = []
        ongoing: list[ViolationEvent] = []
        resolved: list[ViolationEvent] = []

        for m in matches:
            self.buffer.update(m)

        observed = {m.worker_id: m for m in matches}
        for wid in observed:
            buf_dur = self.buffer.buffer_duration(wid)
            for ppe, vtype in PPE_TO_VIOLATION.items():
                key = (wid, vtype)
                ratio = self.buffer.missing_ratio(wid, ppe)
                ev = self._active.get(key)
                if ev is not None:
                    ev.duration_seconds = (datetime.now() - ev.started_at).total_seconds()
                    conf = self.buffer.avg_person_conf_when_missing(wid, ppe)
                    if conf > 0:
                        ev.add_conf(conf)
                    if ratio < self.recovery_threshold:
                        ev.is_active = False
                        ev.ended_at = datetime.now()
                        ev.duration_seconds = (ev.ended_at - ev.started_at).total_seconds()
                        del self._active[key]
                        self._completed.append(ev)
                        resolved.append(ev)
                    else:
                        ongoing.append(ev)
                elif buf_dur >= self.min_violation_duration and ratio >= self.violation_threshold:
                    conf = self.buffer.avg_person_conf_when_missing(wid, ppe)
                    ev = ViolationEvent(
                        event_id=str(uuid.uuid4()), worker_id=int(wid), violation_type=vtype,
                        started_at=datetime.now(), ended_at=None, duration_seconds=0.0,
                        confidence_avg=conf, is_active=True,
                    )
                    if conf > 0:
                        ev._conf_sum, ev._conf_n = conf, 1
                    self._active[key] = ev
                    new.append(ev)

        self.buffer.cleanup(timestamp)
        return {"new": new, "ongoing": ongoing, "resolved": resolved}

    def active(self) -> list[ViolationEvent]:
        return list(self._active.values())

    def completed(self) -> list[ViolationEvent]:
        return list(self._completed)

    def all_events(self) -> list[ViolationEvent]:
        return self.completed() + self.active()


__all__ = [
    "PPEMatch", "ViolationEvent", "IoUMatcher",
    "WorkerStateBuffer", "ViolationDetector", "PPE_TO_VIOLATION",
]
