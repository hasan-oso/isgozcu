"""Threshold sweep for RT-DETR PPE detector.

Runs the model once per frame and post-processes the same logits at multiple
confidence thresholds, so the comparison is apples-to-apples and ~5× faster
than re-running inference per threshold.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: E402


VIDEO = PROJECT_ROOT / "videos" / "samples" / "test_video_01.mp4"
MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v1"
SNAP_DIR = PROJECT_ROOT / "data" / "snapshots" / "threshold_test"
N_FRAMES = 100
THRESHOLDS = [0.40, 0.30, 0.25, 0.20, 0.15]
SAMPLE_FRAMES = [10, 30, 50, 70, 90]
BEST_THRESHOLD = 0.25  # frames at this threshold get saved for visual review

NO_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Mask"}


def load_model():
    proc = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_DIR).cuda().eval().half()
    return proc, model


def run_inference(proc, model, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    h, w = frame_bgr.shape[:2]
    inputs = proc(images=image, return_tensors="pt").to("cuda")
    inputs = {
        k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs, (h, w)


def post_process(proc, outputs, hw, threshold):
    h, w = hw
    target_sizes = torch.tensor([[h, w]], device="cuda")
    return proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]


def _draw_box(img, x1, y1, x2, y2, color, label):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y1 - 6, th + 6)
    cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _color_for(name: str):
    if name in NO_CLASSES:
        return (60, 60, 220)  # red
    if name in COMPLIANT_CLASSES:
        return (60, 200, 60)  # green
    if name == "Person":
        return (220, 130, 50)  # blue
    return (180, 180, 180)


def main() -> int:
    print(f"Loading model from {MODEL_DIR}")
    t0 = time.perf_counter()
    proc, model = load_model()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    print(f"Loading {N_FRAMES} frames from {VIDEO.name}")
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {VIDEO}")
        return 1
    frames: list[np.ndarray] = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"  loaded {len(frames)} frames")

    print("Warming up...")
    _ = run_inference(proc, model, frames[0])

    counts_per_thr: dict[float, dict[str, int]] = {thr: defaultdict(int) for thr in THRESHOLDS}
    cached: dict[int, tuple] = {}  # frame_idx -> (outputs, hw) for sample drawing

    print(f"Running inference x {len(frames)} frames (one fwd pass, post-process at {len(THRESHOLDS)} thresholds)...")
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        outputs, hw = run_inference(proc, model, frame)
        if i in SAMPLE_FRAMES:
            cached[i] = (outputs, hw)
        for thr in THRESHOLDS:
            r = post_process(proc, outputs, hw, thr)
            for cid in r["labels"].cpu().numpy().astype(int):
                name = id2label.get(int(cid), str(cid))
                counts_per_thr[thr][name] += 1
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.2f}s ({len(frames) / elapsed:.2f} fps inference-only)")

    # ---------- table ----------
    classes = sorted(set().union(*[set(c.keys()) for c in counts_per_thr.values()]))
    print()
    print("Detection counts per threshold (across {} frames)".format(len(frames)))
    header = f"{'class':<18}" + "".join(f"{thr:>10.2f}" for thr in THRESHOLDS)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for cls in classes:
        line = f"{cls:<18}"
        for thr in THRESHOLDS:
            line += f"{counts_per_thr[thr].get(cls, 0):>10}"
        print(line)
    # totals + group sums
    print("-" * len(header))
    line = f"{'TOTAL':<18}"
    for thr in THRESHOLDS:
        line += f"{sum(counts_per_thr[thr].values()):>10}"
    print(line)
    line = f"{'  NO-* (violations)':<18}"
    for thr in THRESHOLDS:
        line += f"{sum(counts_per_thr[thr].get(c, 0) for c in NO_CLASSES):>10}"
    print(line)
    line = f"{'  Compliant':<18}"
    for thr in THRESHOLDS:
        line += f"{sum(counts_per_thr[thr].get(c, 0) for c in COMPLIANT_CLASSES):>10}"
    print(line)

    # ---------- sample frames ----------
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sample frames at threshold={BEST_THRESHOLD} -> {SNAP_DIR}")
    for i in SAMPLE_FRAMES:
        if i not in cached:
            continue
        outputs, hw = cached[i]
        results = post_process(proc, outputs, hw, BEST_THRESHOLD)
        canvas = frames[i].copy()
        for box, cid, score in zip(
            results["boxes"].cpu().numpy().astype(int),
            results["labels"].cpu().numpy().astype(int),
            results["scores"].cpu().numpy(),
        ):
            x1, y1, x2, y2 = box.tolist()
            name = id2label.get(int(cid), str(cid))
            _draw_box(canvas, x1, y1, x2, y2, _color_for(name), f"{name} {score:.2f}")
        cv2.putText(
            canvas, f"frame {i}  thr={BEST_THRESHOLD}",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"frame {i}  thr={BEST_THRESHOLD}",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        out_path = SNAP_DIR / f"thr_{BEST_THRESHOLD:.2f}_frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"  {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
