"""tests_new/test_pipeline.py - unit + integration tests for the clean core engine.

Runs under pytest (`pytest tests_new/`) OR standalone (`python tests_new/test_pipeline.py`).
Unit tests use no heavy deps; the integration test loads the v3 model and is
skipped automatically if torch / model / images are unavailable.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:  # allow standalone execution without pytest installed
    pytest = None

from core import postprocess
from core.labels_tr import WHITELIST, is_allowed, tr
from core.violations import IoUMatcher, PPEMatch, ViolationDetector


def _skip(reason: str) -> None:
    if pytest is not None:
        pytest.skip(reason)
    print(f"[SKIP] {reason}")


# ---------------------------------------------------------------- WHITELIST
def test_whitelist_drops_non_three_classes():
    assert is_allowed("Person") and is_allowed("Hardhat") and is_allowed("Safety Vest")
    for bad in ("Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "vehicle", "machinery", "Safety Cone"):
        assert not is_allowed(bad), f"{bad} should be excluded"
    raw = [{"class_name": n} for n in ("Person", "Mask", "Hardhat", "vehicle", "Safety Vest")]
    kept = [d["class_name"] for d in raw if is_allowed(d["class_name"])]
    assert kept == ["Person", "Hardhat", "Safety Vest"]


# ------------------------------------------------------------- Turkish dict
def test_tr_strings():
    assert tr("helmet_missing") == "Baret takılmamış"
    assert tr("vest_missing") == "Güvenlik yeleği yok"
    assert tr("worker_ok") == "Kurallara uygun"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tr("nonexistent_key") == "nonexistent_key"  # falls back to key


# ------------------------------------------------------------- postprocess
def test_postprocess_floor_and_geometry():
    frame_w, frame_h = 1280, 720
    dets = [
        {"bbox": (100, 100, 200, 400), "class_name": "Person", "confidence": 0.90, "track_id": 1},
        {"bbox": (100, 100, 200, 400), "class_name": "Person", "confidence": 0.40, "track_id": 2},  # below floor
        {"bbox": (0, 0, 5, 5), "class_name": "Person", "confidence": 0.95, "track_id": 3},          # too small
    ]
    out = postprocess.process(dets, frame_w, frame_h)
    assert len(out) == 1 and out[0]["track_id"] == 1


# ------------------------------------------------------------- IoUMatcher
def test_iou_matcher_associates_helmet_over_head():
    person = {"bbox": (100, 100, 200, 400), "track_id": 7, "confidence": 0.9}  # w=100 h=300
    hardhat_on = {"bbox": (140, 115, 165, 150), "confidence": 0.8}  # center in head band (y 100..190)
    vest_on = {"bbox": (110, 250, 190, 330), "confidence": 0.7}     # center in torso band (y 160..340)
    m = IoUMatcher().match_frame([person], [hardhat_on], [vest_on], frame_number=0, timestamp=0.0)
    assert len(m) == 1 and m[0].has_hardhat is True and m[0].has_vest is True


def test_iou_matcher_rejects_far_ppe():
    person = {"bbox": (100, 100, 200, 400), "track_id": 7, "confidence": 0.9}
    hardhat_far = {"bbox": (500, 500, 540, 540), "confidence": 0.8}
    m = IoUMatcher().match_frame([person], [hardhat_far], [], frame_number=0, timestamp=0.0)
    assert len(m) == 1 and m[0].has_hardhat is False and m[0].has_vest is False


# -------------------------------------------------- debounce + hysteresis
def _feed(vd, t0, t1, step, has_hardhat, has_vest, wid=7):
    """Feed single-worker matches across [t0, t1). Returns first-fire timestamp or None."""
    fired_ts = None
    t, fn = t0, 0
    while t < t1 - 1e-9:
        m = PPEMatch(worker_id=wid, person_bbox=(100, 100, 200, 400), person_confidence=0.9,
                     has_hardhat=has_hardhat, has_vest=has_vest, frame_number=fn, timestamp=round(t, 4))
        res = vd.process_matches([m], frame_number=fn, timestamp=round(t, 4))
        if res["new"] and fired_ts is None:
            fired_ts = round(t, 4)
        t += step
        fn += 1
    return fired_ts


def test_no_fire_before_min_duration_and_hysteresis():
    vd = ViolationDetector(min_violation_duration=8.0, violation_threshold=0.85,
                           recovery_threshold=0.3, fps=10.0)
    # Phase A: helmet missing, but only ~7s of evidence -> must NOT fire (debounce).
    fired_early = _feed(vd, 0.0, 7.0, 0.1, has_hardhat=False, has_vest=True)
    assert fired_early is None, "fired before reaching min_violation_duration"
    assert len(vd.active()) == 0

    # Phase B: cross 8s -> fires helmet_missing only (vest is present).
    fired_ts = _feed(vd, 7.0, 8.6, 0.1, has_hardhat=False, has_vest=True)
    assert fired_ts is not None and fired_ts >= 7.5, f"fired too early at {fired_ts}"
    active = vd.active()
    assert len(active) == 1 and active[0].violation_type == "helmet_missing"
    assert active[0].label_tr == "Baret takılmamış"

    # Phase C: ONE compliant frame must NOT resolve immediately (hysteresis).
    res = vd.process_matches(
        [PPEMatch(worker_id=7, person_bbox=(100, 100, 200, 400), person_confidence=0.9,
                  has_hardhat=True, has_vest=True, frame_number=999, timestamp=8.7)],
        frame_number=999, timestamp=8.7)
    assert res["resolved"] == [] and len(vd.active()) == 1, "single good frame flipped state"

    # Phase D: sustained compliance -> ratio < recovery -> resolves.
    _feed(vd, 8.8, 18.0, 0.1, has_hardhat=True, has_vest=True)
    assert len(vd.active()) == 0, "violation never resolved"
    assert any(e.violation_type == "helmet_missing" for e in vd.completed())


# ------------------------------------------ integration on real images (opt)
def test_integration_training_images():
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        from core.detector import DetectorAdapter
    except Exception as e:
        _skip(f"detector deps unavailable: {e}")
        return
    model_weights = PROJECT_ROOT / "model" / "weights" / "v3" / "model.safetensors"
    if not model_weights.exists():
        _skip("v3 weights not present")
        return
    img_root = PROJECT_ROOT / "data" / "training_images"
    images = sorted(img_root.rglob("*.jpg"))[:4] if img_root.exists() else []
    if not images:
        _skip("no training_images found")
        return

    import cv2
    det = DetectorAdapter(detect_every_n=1, conf_threshold=0.25)
    matcher = IoUMatcher()
    vd = ViolationDetector(fps=8.0)
    total = 0
    for k, p in enumerate(images):
        frame = cv2.imread(str(p))
        assert frame is not None, f"could not read {p}"
        h, w = frame.shape[:2]
        dets = det.detect_and_track(frame)
        assert all(d["class_name"] in WHITELIST for d in dets)
        cleaned = postprocess.process(dets, w, h)
        assert all(d["class_name"] in WHITELIST for d in cleaned)
        persons = [d for d in cleaned if d["class_name"] == "Person"]
        hardhats = [d for d in cleaned if d["class_name"] == "Hardhat"]
        vests = [d for d in cleaned if d["class_name"] == "Safety Vest"]
        matches = matcher.match_frame(persons, hardhats, vests, frame_number=k, timestamp=k / 8.0)
        vd.process_matches(matches, frame_number=k, timestamp=k / 8.0)
        total += len(cleaned)
    print(f"[integration] {len(images)} images, {total} detections, stats={det.stats()}")


# ------------------------------------------------------- standalone runner
def _run_standalone() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
