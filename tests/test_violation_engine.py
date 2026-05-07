"""End-to-end logic tests for the violation engine using synthetic PPEMatches."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.violation_engine import (  # noqa: E402
    PPEMatch,
    ViolationDetector,
    WorkerStateBuffer,
)


def _make_match(worker_id, *, hardhat=None, vest=None, mask=None, frame=0, ts=0.0):
    return PPEMatch(
        worker_id=worker_id,
        person_bbox=(0, 0, 100, 200),
        has_hardhat=hardhat,
        has_vest=vest,
        has_mask=mask,
        hardhat_confidence=0.7 if hardhat is not None else 0.0,
        vest_confidence=0.7 if vest is not None else 0.0,
        mask_confidence=0.7 if mask is not None else 0.0,
        frame_number=frame,
        timestamp=ts,
    )


def test_buffer_violation_ratio():
    buf = WorkerStateBuffer(window_seconds=2.0, fps=10.0)
    # window_size = round(2.0*10) + 1 = 21
    assert buf.window_size == 21
    for i in range(21):
        buf.update(_make_match(1, hardhat=False, frame=i, ts=i * 0.1))
    # All False -> ratio 1.0
    assert abs(buf.get_violation_ratio(1, "hardhat") - 1.0) < 1e-6
    # Now add 5 Trues; deque drops oldest 5 → 16 Falses + 5 Trues = 21
    for i in range(5):
        buf.update(_make_match(1, hardhat=True, frame=21 + i, ts=2.1 + i * 0.1))
    r = buf.get_violation_ratio(1, "hardhat")
    expected = 16 / 21
    assert abs(r - expected) < 1e-6, f"expected {expected} got {r}"


def test_violation_fires_after_min_duration():
    det = ViolationDetector(
        min_violation_duration=2.0,
        violation_threshold=0.7,
        recovery_threshold=0.3,
        fps=10.0,
    )
    fired = None
    for i in range(40):
        m = _make_match(1, hardhat=False, frame=i, ts=i * 0.1)
        result = det.process_matches([m], i, i * 0.1)
        if result["new"]:
            fired = i
            break
    assert fired is not None, "violation should fire when worker is non-compliant for >= min_duration"
    assert 18 <= fired <= 25, f"fired too early/late at frame {fired}"


def test_violation_does_not_fire_below_threshold():
    """Worker who is mostly compliant (90% True, 10% False) should not trigger."""
    det = ViolationDetector(min_violation_duration=2.0, violation_threshold=0.7, fps=10.0)
    fired_any = False
    for i in range(50):
        # 1 in 10 frames is non-compliant
        nh = (i % 10 == 0)
        m = _make_match(1, hardhat=not nh, frame=i, ts=i * 0.1)
        result = det.process_matches([m], i, i * 0.1)
        if result["new"]:
            fired_any = True
            break
    assert not fired_any, "should not fire for mostly-compliant worker"


def test_violation_resolves_when_worker_recovers():
    det = ViolationDetector(
        min_violation_duration=2.0,
        violation_threshold=0.7,
        recovery_threshold=0.3,
        fps=10.0,
    )
    # First 30 frames non-compliant -> fires
    for i in range(30):
        det.process_matches([_make_match(1, hardhat=False, frame=i, ts=i * 0.1)], i, i * 0.1)
    assert len(det.get_active_violations()) == 1

    # Then 30 frames compliant -> resolves
    resolved = 0
    for i in range(30, 60):
        result = det.process_matches([_make_match(1, hardhat=True, frame=i, ts=i * 0.1)], i, i * 0.1)
        resolved += len(result["resolved"])
    assert resolved >= 1, "violation should resolve when worker recovers"
    assert len(det.get_active_violations()) == 0


def test_finalize_closes_active():
    det = ViolationDetector(min_violation_duration=2.0, violation_threshold=0.7, fps=10.0)
    for i in range(30):
        det.process_matches([_make_match(1, hardhat=False, frame=i, ts=i * 0.1)], i, i * 0.1)
    assert len(det.get_active_violations()) == 1
    det.finalize(end_timestamp=3.0)
    assert len(det.get_active_violations()) == 0
    assert len(det.get_completed_violations()) == 1
    ev = det.get_completed_violations()[0]
    assert ev.is_active is False
    assert ev.ended_at is not None


def test_event_to_dict_has_required_fields():
    det = ViolationDetector(min_violation_duration=1.0, violation_threshold=0.7, fps=10.0)
    for i in range(20):
        det.process_matches([_make_match(1, hardhat=False, frame=i, ts=i * 0.1)], i, i * 0.1)
    det.finalize(2.0)
    ev = det.get_completed_violations()[0]
    d = ev.to_dict()
    for k in (
        "event_id", "worker_id", "violation_type", "violation_label_tr",
        "camera_id", "started_at", "ended_at", "duration_seconds",
        "snapshot_path", "confidence_avg", "is_active",
    ):
        assert k in d, f"missing {k}"
    assert d["violation_type"] == "no_hardhat"
    assert d["violation_label_tr"] == "Baret yok"


def main() -> int:
    tests = [
        test_buffer_violation_ratio,
        test_violation_fires_after_min_duration,
        test_violation_does_not_fire_below_threshold,
        test_violation_resolves_when_worker_recovers,
        test_finalize_closes_active,
        test_event_to_dict_has_required_fields,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures} / {len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
