"""Side-by-side: RT-DETR v1 vs v2 on the same 200 frames.

Loads both checkpoints, runs each on identical frames, compares per-class
detection counts and inference speed, and writes an annotated 2× width video
to videos/outputs/v1_vs_v2.mp4 (left = v1, right = v2).
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout (Windows TR/AR shells default to a non-UTF8 codepage that
# can't encode the Δ symbol or Turkish diacritics).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.inference.rtdetr_detector import RTDetrDetector  # noqa: E402

VIDEO = PROJECT_ROOT / "videos" / "samples" / "test_video_01.mp4"
V1_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v1"
V2_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v2"
OUT_VIDEO = PROJECT_ROOT / "videos" / "outputs" / "v1_vs_v2.mp4"
N_FRAMES = 200
CONF = 0.25

NO_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Mask"}


def _color_for(name: str):
    if name in NO_CLASSES:
        return (60, 60, 220)
    if name in COMPLIANT_CLASSES:
        return (60, 200, 60)
    if name == "Person":
        return (220, 130, 50)
    return (200, 200, 200)


def _draw_box(img, x1, y1, x2, y2, color, label):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y1 - 6, th + 6)
    cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _banner(img, text):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)


def run_one(name: str, model_path: Path, frames: list[np.ndarray]) -> dict:
    print(f"Loading {name} from {model_path.name} ...")
    det = RTDetrDetector(model_path=model_path, conf_threshold=CONF)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Warm up (do not count)
    _ = det.detect(frames[0])

    counts: dict[str, int] = defaultdict(int)
    confs_per_class: dict[str, list[float]] = defaultdict(list)
    inf_times: list[float] = []
    annotated: list[np.ndarray] = []

    t0 = time.perf_counter()
    for frame in frames:
        ti = time.perf_counter()
        d = det.detect(frame)
        inf_times.append(time.perf_counter() - ti)
        canvas = frame.copy()
        for i in range(len(d)):
            cid = int(d.class_id[i])
            cname = det.get_class_name(cid)
            conf = float(d.confidence[i])
            counts[cname] += 1
            confs_per_class[cname].append(conf)
            x1, y1, x2, y2 = d.xyxy[i].astype(int)
            _draw_box(canvas, x1, y1, x2, y2, _color_for(cname), f"{cname} {conf:.2f}")
        annotated.append(canvas)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "counts": dict(counts),
        "confs_per_class": dict(confs_per_class),
        "inf_times_ms": [t * 1000 for t in inf_times],
        "fps": len(frames) / elapsed,
        "elapsed_s": elapsed,
        "vram_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0,
        "frames": annotated,
    }


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _print_table(v1: dict, v2: dict) -> None:
    classes = sorted(set(v1["counts"]) | set(v2["counts"]))
    if not classes:
        # ensure stable column even if empty
        classes = sorted(
            ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
             "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"]
        )
    print()
    print(f"{'Class':<18}{'v1':>9}{'v2':>9}{'Δ':>10}{'v1 avg':>10}{'v2 avg':>10}")
    print("-" * 66)
    for c in classes:
        c1 = v1["counts"].get(c, 0)
        c2 = v2["counts"].get(c, 0)
        delta = "n/a"
        if c1 > 0:
            delta = f"{(c2 - c1) / c1 * 100:+.0f}%"
        elif c2 > 0:
            delta = "+inf"
        a1 = _avg(v1["confs_per_class"].get(c, []))
        a2 = _avg(v2["confs_per_class"].get(c, []))
        print(f"{c:<18}{c1:>9}{c2:>9}{delta:>10}{a1:>10.3f}{a2:>10.3f}")
    print("-" * 66)
    print(
        f"{'TOTAL':<18}{sum(v1['counts'].values()):>9}{sum(v2['counts'].values()):>9}"
    )
    print()
    print(f"FPS:        v1 {v1['fps']:.2f}   v2 {v2['fps']:.2f}")
    print(f"Avg ms/frame: v1 {_avg(v1['inf_times_ms']):.1f}   v2 {_avg(v2['inf_times_ms']):.1f}")
    print(f"Peak VRAM:    v1 {v1['vram_mb']:.0f} MB   v2 {v2['vram_mb']:.0f} MB")


def _print_verdict(v1: dict, v2: dict) -> None:
    print("\n=== VERDICT ===")
    improved, degraded, unchanged = [], [], []
    for c in sorted(set(v1["counts"]) | set(v2["counts"])):
        c1 = v1["counts"].get(c, 0)
        c2 = v2["counts"].get(c, 0)
        if c1 == c2:
            unchanged.append(c)
        elif c2 > c1:
            improved.append((c, c1, c2))
        else:
            degraded.append((c, c1, c2))
    print("Improved (more detections in v2):")
    for c, a, b in improved:
        print(f"  {c}: {a} -> {b}  (+{b - a})")
    print("Degraded (fewer detections in v2):")
    for c, a, b in degraded:
        print(f"  {c}: {a} -> {b}  ({b - a})")
    print("Unchanged:", ", ".join(unchanged) if unchanged else "(none)")


def main() -> int:
    if not V1_PATH.exists():
        print(f"[HATA] v1 yok: {V1_PATH}")
        return 1
    if not V2_PATH.exists():
        print(f"[HATA] v2 yok: {V2_PATH}")
        return 1

    print(f"Loading {N_FRAMES} frames from {VIDEO.name} ...")
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {VIDEO}")
        return 1
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop if video ends
            ok, f = cap.read()
            if not ok:
                break
        frames.append(f)
    cap.release()
    print(f"  loaded {len(frames)} frames at {w}x{h} @ {fps_src:.1f} fps")

    v1 = run_one("v1", V1_PATH, frames)
    v2 = run_one("v2", V2_PATH, frames)

    _print_table(v1, v2)
    _print_verdict(v1, v2)

    # Side-by-side video
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps_src, (w * 2, h))
    if not writer.isOpened():
        print(f"[HATA] Çıkış videosu yazılamadı: {OUT_VIDEO}")
        return 2
    for f1, f2 in zip(v1["frames"], v2["frames"]):
        l, r = f1.copy(), f2.copy()
        _banner(l, "RT-DETR v1")
        _banner(r, "RT-DETR v2")
        writer.write(np.concatenate([l, r], axis=1))
    writer.release()
    out_size_mb = OUT_VIDEO.stat().st_size / (1024 * 1024)
    print(f"\nComparison video: {OUT_VIDEO}  ({out_size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
