"""Unit tests for DetectionPostProcessor.

Same assertion-based style as `tests/test_iou_matcher.py` and
`tests/test_violation_engine.py` so they run as a plain script *and* are
discoverable by pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.post_processor import DetectionPostProcessor  # noqa: E402


CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def _det(name: str, bbox, conf: float, track_id=None):
    """Build a single StandardDetection-shaped dict."""
    return {
        "bbox": tuple(int(v) for v in bbox),
        "class_id": NAME_TO_ID[name],
        "class_name": name,
        "confidence": float(conf),
        "track_id": track_id,
    }


def _pp(width=1280, height=720) -> DetectionPostProcessor:
    return DetectionPostProcessor(class_names=CLASS_NAMES, frame_width=width, frame_height=height)


# ---------- step 1: confidence floors ----------


def test_confidence_floor_drops_low_conf_person():
    pp = _pp()
    keep = _det("Person", (200, 200, 600, 700), conf=0.80)
    drop = _det("Person", (700, 200, 1100, 700), conf=0.60)  # < 0.75 floor
    out = pp.process([keep, drop])
    assert len(out) == 1, out
    assert out[0]["confidence"] == 0.80


def test_confidence_floor_drops_low_conf_no_vest():
    pp = _pp()
    drop = _det("NO-Safety Vest", (300, 300, 500, 600), conf=0.65)  # < 0.70
    keep = _det("NO-Safety Vest", (600, 300, 800, 600), conf=0.85)
    out = pp.process([drop, keep])
    assert len(out) == 1
    assert out[0]["confidence"] == 0.85


def test_confidence_floor_uses_default_for_unknown_class():
    pp = _pp()
    # Confidence 0.45 — under the 0.5 default
    drop = _det("machinery", (10, 10, 100, 100), conf=0.45)
    out = pp.process([drop])
    assert out == []


# ---------- step 2: person geometric validation ----------


def test_person_too_small_rejected():
    pp = _pp(1280, 720)
    # 50x50 = 2500 px² → 2500/(1280*720) = 0.0027 < 0.005 floor
    tiny_person = _det("Person", (100, 100, 150, 150), conf=0.90)
    out = pp.process([tiny_person])
    assert out == [], f"expected empty, got {out}"


def test_person_too_huge_rejected():
    pp = _pp(1280, 720)
    # Covers 90% of frame → > 0.85 cap
    huge_person = _det("Person", (10, 10, 1270, 700), conf=0.90)
    out = pp.process([huge_person])
    assert out == []


def test_person_too_thin_rejected():
    pp = _pp(1280, 720)
    # h/w = 1000/100 = 10 > 5 max aspect
    thin_person = _det("Person", (500, 50, 600, 700), conf=0.90)
    # Make it big enough to pass area check (100 * 650 = 65000 px² → 0.07 OK)
    out = pp.process([thin_person])
    assert out == [], f"expected empty (aspect too thin), got {out}"


def test_person_too_wide_rejected():
    pp = _pp(1280, 720)
    # h/w = 100/600 = 0.16 < 0.20 min aspect (looks like a shadow strip)
    wide = _det("Person", (300, 300, 900, 400), conf=0.90)
    out = pp.process([wide])
    assert out == []


def test_person_normal_kept():
    pp = _pp(1280, 720)
    # 200x500 = 100,000 px² → 0.108 of frame; aspect 2.5
    person = _det("Person", (300, 100, 500, 600), conf=0.90)
    out = pp.process([person])
    assert len(out) == 1


def test_geometric_filter_does_not_touch_non_person():
    pp = _pp(1280, 720)
    # A tiny Hardhat box should NOT be rejected by Person geometry checks
    # (geometry only applies to Person class)
    tiny_hardhat = _det("Hardhat", (100, 100, 130, 130), conf=0.80)
    out = pp.process([tiny_hardhat])
    assert len(out) == 1


# ---------- step 3: hardhat / NO-Hardhat dedup ----------


def test_overlapping_hardhat_no_hardhat_keeps_higher_conf():
    pp = _pp()
    # Two boxes covering the same head, IoU >> 0.5
    hardhat = _det("Hardhat", (500, 100, 700, 250), conf=0.90)
    no_hard = _det("NO-Hardhat", (510, 105, 695, 248), conf=0.60)
    out = pp.process([hardhat, no_hard])
    # Hardhat should win (higher conf)
    assert len(out) == 1
    assert out[0]["class_name"] == "Hardhat"


def test_overlapping_no_hardhat_higher_wins():
    pp = _pp()
    hardhat = _det("Hardhat", (500, 100, 700, 250), conf=0.55)
    no_hard = _det("NO-Hardhat", (510, 105, 695, 248), conf=0.85)
    out = pp.process([hardhat, no_hard])
    assert len(out) == 1
    assert out[0]["class_name"] == "NO-Hardhat"


def test_non_overlapping_hardhat_pair_both_kept():
    pp = _pp()
    # Two different people / heads, no IoU between them
    a = _det("Hardhat", (100, 100, 250, 250), conf=0.80)
    b = _det("NO-Hardhat", (900, 100, 1050, 250), conf=0.80)
    out = pp.process([a, b])
    assert len(out) == 2


# ---------- step 4: vest temporal smoothing ----------


def test_vest_smoothing_rejects_single_no_vest_blip():
    pp = _pp()
    PERSON_BOX = (300, 100, 600, 700)  # large valid person
    VEST_BOX = (320, 220, 580, 480)
    # Run 30 frames of clean Vest detections under the same track_id.
    for _ in range(30):
        pp.process(
            [
                _det("Person", PERSON_BOX, 0.90, track_id=42),
                _det("Safety Vest", VEST_BOX, 0.85, track_id=42),
            ]
        )
    # Frame 31 has a single NO-Safety Vest blip — should be dropped.
    out = pp.process(
        [
            _det("Person", PERSON_BOX, 0.90, track_id=42),
            _det("NO-Safety Vest", VEST_BOX, 0.80, track_id=42),
        ]
    )
    classes = sorted(d["class_name"] for d in out)
    # Expect: Person stays, NO-Safety Vest blip is suppressed.
    assert "NO-Safety Vest" not in classes, f"blip should have been dropped, got {classes}"
    assert "Person" in classes


def test_vest_smoothing_lets_no_vest_through_when_consistent():
    pp = _pp()
    PERSON = (300, 100, 600, 700)
    VEST = (320, 220, 580, 480)
    # 30 frames of NO-Safety Vest under track_id 7
    for _ in range(30):
        pp.process(
            [
                _det("Person", PERSON, 0.90, track_id=7),
                _det("NO-Safety Vest", VEST, 0.80, track_id=7),
            ]
        )
    # Frame 31: still NO-Safety Vest — must stay
    out = pp.process(
        [
            _det("Person", PERSON, 0.90, track_id=7),
            _det("NO-Safety Vest", VEST, 0.80, track_id=7),
        ]
    )
    classes = sorted(d["class_name"] for d in out)
    assert "NO-Safety Vest" in classes, classes


def test_vest_smoothing_handles_unknown_track_id():
    pp = _pp()
    # Without a track_id, smoother passes through (no history to compare).
    out = pp.process([_det("Safety Vest", (100, 100, 300, 400), 0.80, track_id=None)])
    assert len(out) == 1


def test_reset_clears_vest_history():
    pp = _pp()
    PERSON = (300, 100, 600, 700)
    VEST = (320, 220, 580, 480)
    for _ in range(30):
        pp.process(
            [
                _det("Person", PERSON, 0.90, track_id=99),
                _det("Safety Vest", VEST, 0.85, track_id=99),
            ]
        )
    pp.reset_all()
    # After reset, a single NO-Safety Vest should pass (no history to override).
    out = pp.process(
        [
            _det("Person", PERSON, 0.90, track_id=99),
            _det("NO-Safety Vest", VEST, 0.80, track_id=99),
        ]
    )
    assert any(d["class_name"] == "NO-Safety Vest" for d in out)


# ---------- empty / no-op edge cases ----------


def test_empty_detections_pass_through():
    pp = _pp()
    assert pp.process([]) == []


def test_all_filters_chained_realistic_frame():
    pp = _pp()
    # Real frame mix:
    #   - one good Person + Hardhat + Safety Vest
    #   - one shadow Person (huge box)
    #   - one low-conf NO-Safety Vest blip
    #   - one duplicate NO-Hardhat overlapping the Hardhat
    out = pp.process(
        [
            _det("Person", (300, 100, 600, 700), 0.90, track_id=1),
            _det("Hardhat", (380, 110, 510, 220), 0.85, track_id=1),
            _det("Safety Vest", (340, 220, 580, 480), 0.80, track_id=1),
            _det("Person", (10, 10, 1270, 700), 0.90, track_id=2),  # huge shadow
            _det("NO-Safety Vest", (340, 220, 580, 480), 0.55, track_id=1),  # below 0.70 floor
            _det("NO-Hardhat", (385, 115, 505, 215), 0.50, track_id=1),  # overlaps real Hardhat
        ]
    )
    classes = sorted(d["class_name"] for d in out)
    # Expect: Person (good one), Hardhat, Safety Vest. The huge Person is geom-rejected;
    # NO-Safety Vest is conf-floored; NO-Hardhat is dedup-suppressed by Hardhat.
    assert "Hardhat" in classes
    assert "Safety Vest" in classes
    assert "Person" in classes
    # Exactly one Person (the good one)
    assert sum(1 for c in classes if c == "Person") == 1, f"got {classes}"
    assert "NO-Safety Vest" not in classes, f"got {classes}"
    assert "NO-Hardhat" not in classes, f"got {classes}"


def main() -> int:
    tests = [
        test_confidence_floor_drops_low_conf_person,
        test_confidence_floor_drops_low_conf_no_vest,
        test_confidence_floor_uses_default_for_unknown_class,
        test_person_too_small_rejected,
        test_person_too_huge_rejected,
        test_person_too_thin_rejected,
        test_person_too_wide_rejected,
        test_person_normal_kept,
        test_geometric_filter_does_not_touch_non_person,
        test_overlapping_hardhat_no_hardhat_keeps_higher_conf,
        test_overlapping_no_hardhat_higher_wins,
        test_non_overlapping_hardhat_pair_both_kept,
        test_vest_smoothing_rejects_single_no_vest_blip,
        test_vest_smoothing_lets_no_vest_through_when_consistent,
        test_vest_smoothing_handles_unknown_track_id,
        test_reset_clears_vest_history,
        test_empty_detections_pass_through,
        test_all_filters_chained_realistic_frame,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails} / {len(tests)} passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
