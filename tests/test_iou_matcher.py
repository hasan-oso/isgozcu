"""Unit tests for IoUMatcher."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.violation_engine import IoUMatcher  # noqa: E402


def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return math.isclose(a, b, abs_tol=eps)


def test_iou_identical_boxes():
    a = (0, 0, 100, 100)
    assert _approx(IoUMatcher.calculate_iou(a, a), 1.0)


def test_iou_disjoint_boxes():
    a = (0, 0, 50, 50)
    b = (60, 60, 100, 100)
    assert _approx(IoUMatcher.calculate_iou(a, b), 0.0)


def test_iou_touching_edges_no_overlap():
    a = (0, 0, 50, 50)
    b = (50, 0, 100, 50)
    assert _approx(IoUMatcher.calculate_iou(a, b), 0.0)


def test_iou_half_overlap():
    a = (0, 0, 100, 100)            # area 10000
    b = (50, 0, 150, 100)           # area 10000, intersection 5000, union 15000
    iou = IoUMatcher.calculate_iou(a, b)
    assert _approx(iou, 5000 / 15000)


def test_iou_b_inside_a():
    a = (0, 0, 100, 100)            # area 10000
    b = (25, 25, 75, 75)            # area 2500, intersection 2500
    iou = IoUMatcher.calculate_iou(a, b)
    assert _approx(iou, 2500 / 10000)


def test_iou_zero_area_box():
    a = (0, 0, 0, 0)
    b = (0, 0, 10, 10)
    assert _approx(IoUMatcher.calculate_iou(a, b), 0.0)


def test_iou_negative_coords():
    a = (-50, -50, 50, 50)
    b = (0, 0, 100, 100)
    iou = IoUMatcher.calculate_iou(a, b)
    # intersection 50x50 = 2500; areas 10000 each; union 17500
    assert _approx(iou, 2500 / 17500)


def test_head_region_center_inside():
    person = (100, 100, 200, 400)   # 100 wide, 300 tall; head = upper 30% = y in [100, 190]
    hat = (130, 110, 170, 150)      # center at (150, 130) -> inside head region
    assert IoUMatcher.is_inside_person_region(hat, person, "head")


def test_head_region_center_outside_uses_iou():
    person = (100, 100, 200, 400)   # head ends at y=190
    hat = (130, 250, 170, 290)      # center far below head; no overlap with head region
    assert not IoUMatcher.is_inside_person_region(hat, person, "head")


def test_torso_region_center_inside():
    person = (100, 100, 200, 400)   # torso y in [160, 340]
    vest = (110, 200, 190, 300)     # center at (150, 250) -> inside torso
    assert IoUMatcher.is_inside_person_region(vest, person, "torso")


def test_face_region_smaller_than_head():
    person = (0, 0, 100, 1000)      # head = upper 30% (y < 300), face = upper 20% (y < 200)
    item = (10, 250, 90, 290)       # center y=270 -> inside head, NOT inside face
    assert IoUMatcher.is_inside_person_region(item, person, "head")
    assert not IoUMatcher.is_inside_person_region(item, person, "face")


def main() -> int:
    tests = [
        test_iou_identical_boxes,
        test_iou_disjoint_boxes,
        test_iou_touching_edges_no_overlap,
        test_iou_half_overlap,
        test_iou_b_inside_a,
        test_iou_zero_area_box,
        test_iou_negative_coords,
        test_head_region_center_inside,
        test_head_region_center_outside_uses_iou,
        test_torso_region_center_inside,
        test_face_region_smaller_than_head,
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
