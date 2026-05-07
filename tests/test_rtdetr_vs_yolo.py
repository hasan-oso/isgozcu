"""Side-by-side comparison: YOLOv8 vs RT-DETR on the same video.

Runs both detectors on N frames of test_video_01.mp4, prints per-class detection
counts + FPS + VRAM, and writes an annotated side-by-side video to
videos/outputs/comparison.mp4 (left = YOLO, right = RT-DETR).
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")  # supervision deprecation noise

from ultralytics import YOLO  # noqa: E402

from backend.inference.rtdetr_detector import RTDetrDetector  # noqa: E402


VIDEO = PROJECT_ROOT / "videos" / "samples" / "test_video_01.mp4"
YOLO_MODEL = PROJECT_ROOT / "backend" / "models" / "ppe_yolov8n.pt"
RTDETR_MODEL = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v1"
OUT_VIDEO = PROJECT_ROOT / "videos" / "outputs" / "comparison.mp4"
N_FRAMES = 100
CONF = 0.4

COMPLIANT = {"Hardhat", "Safety Vest", "Mask"}
VIOLATION = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
COLOR_OK = (60, 200, 60)
COLOR_BAD = (60, 60, 220)
COLOR_PERSON = (220, 130, 50)
COLOR_OTHER = (180, 180, 180)


def _color(name: str):
    if name in COMPLIANT:
        return COLOR_OK
    if name in VIOLATION:
        return COLOR_BAD
    if name == "Person":
        return COLOR_PERSON
    return COLOR_OTHER


def _draw_box(img, x1, y1, x2, y2, color, label):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y1 - 6, th + 6)
    cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _banner(img, text: str):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)


def run_yolo(frames: list[np.ndarray]) -> dict:
    model = YOLO(str(YOLO_MODEL))
    class_names = model.names
    counts: dict[str, int] = defaultdict(int)
    inf_times: list[float] = []
    annotated_frames: list[np.ndarray] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warm up
    _ = model.track(frames[0], conf=CONF, persist=True, tracker="bytetrack.yaml", verbose=False, half=True, device="cuda", imgsz=640)

    t0 = time.perf_counter()
    for frame in frames:
        ti = time.perf_counter()
        results = model.track(frame, conf=CONF, persist=True, tracker="bytetrack.yaml", verbose=False, half=True, device="cuda", imgsz=640)
        inf_times.append(time.perf_counter() - ti)
        r = results[0]
        canvas = frame.copy()
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            cls_idx = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, conf in zip(xyxy, cls_idx, confs):
                name = class_names.get(int(c), str(c))
                counts[name] += 1
                _draw_box(canvas, x1, y1, x2, y2, _color(name), f"{name} {conf:.2f}")
        annotated_frames.append(canvas)
    elapsed = time.perf_counter() - t0
    return {
        "name": "YOLOv8n",
        "counts": dict(counts),
        "elapsed": elapsed,
        "inf_times_ms": [t * 1000 for t in inf_times],
        "fps": len(frames) / elapsed,
        "vram_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0,
        "frames": annotated_frames,
    }


def run_rtdetr(frames: list[np.ndarray]) -> dict:
    det = RTDetrDetector(model_path=RTDETR_MODEL, conf_threshold=CONF)
    counts: dict[str, int] = defaultdict(int)
    inf_times: list[float] = []
    annotated_frames: list[np.ndarray] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warm up
    _ = det.detect_and_track(frames[0])
    det.reset_tracker()

    t0 = time.perf_counter()
    for frame in frames:
        ti = time.perf_counter()
        d = det.detect_and_track(frame)
        inf_times.append(time.perf_counter() - ti)
        canvas = frame.copy()
        if len(d) > 0:
            for i in range(len(d)):
                x1, y1, x2, y2 = d.xyxy[i].astype(int)
                cid = int(d.class_id[i])
                conf = float(d.confidence[i])
                name = det.get_class_name(cid)
                counts[name] += 1
                _draw_box(canvas, x1, y1, x2, y2, _color(name), f"{name} {conf:.2f}")
        annotated_frames.append(canvas)
    elapsed = time.perf_counter() - t0
    return {
        "name": "RT-DETR",
        "counts": dict(counts),
        "elapsed": elapsed,
        "inf_times_ms": [t * 1000 for t in inf_times],
        "fps": len(frames) / elapsed,
        "vram_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0,
        "frames": annotated_frames,
    }


def _compose_side_by_side(left: np.ndarray, right: np.ndarray, name_l: str, name_r: str) -> np.ndarray:
    _banner(left, name_l)
    _banner(right, name_r)
    return np.concatenate([left, right], axis=1)


def _summary(name: str, r: dict) -> None:
    inf_times = r["inf_times_ms"]
    print(f"\n=== {name} ===")
    print(f"  Frames:           {len(inf_times)}")
    print(f"  Total elapsed:    {r['elapsed']:.2f} s")
    print(f"  Avg FPS:          {r['fps']:.2f}")
    print(f"  Avg inf time:     {sum(inf_times) / len(inf_times):.1f} ms")
    print(f"  Median inf time:  {sorted(inf_times)[len(inf_times) // 2]:.1f} ms")
    print(f"  Peak VRAM:        {r['vram_mb']:.1f} MB")
    print(f"  Total dets:       {sum(r['counts'].values())}")
    print(f"  Per-class detections:")
    for k, v in sorted(r["counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18} {v}")


def _verdict(yolo: dict, rt: dict) -> str:
    yolo_total = sum(yolo["counts"].values())
    rt_total = sum(rt["counts"].values())
    parts = [
        f"YOLOv8n: {yolo['fps']:.1f} fps, {yolo_total} dets, {yolo['vram_mb']:.0f} MB",
        f"RT-DETR: {rt['fps']:.1f} fps, {rt_total} dets, {rt['vram_mb']:.0f} MB",
    ]
    faster = "YOLO" if yolo["fps"] > rt["fps"] else "RT-DETR"
    more_dets = "YOLO" if yolo_total > rt_total else "RT-DETR"
    parts.append(f"Faster: {faster}.  More detections: {more_dets}.")
    return "  |  ".join(parts)


def main() -> int:
    print(f"Loading {N_FRAMES} frames from {VIDEO} ...")
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {VIDEO}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"  loaded {len(frames)} frames at {w}x{h} @ {fps:.1f} fps")

    print("\nRunning YOLOv8...")
    yolo_r = run_yolo(frames)
    print("Running RT-DETR...")
    rt_r = run_rtdetr(frames)

    _summary("YOLOv8", yolo_r)
    _summary("RT-DETR", rt_r)
    print("\n=== VERDICT ===")
    print(_verdict(yolo_r, rt_r))

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (w * 2, h))
    if not writer.isOpened():
        print(f"[HATA] cıkış videosu yazılamadı: {OUT_VIDEO}")
        return 2
    for fy, fr in zip(yolo_r["frames"], rt_r["frames"]):
        writer.write(_compose_side_by_side(fy, fr, yolo_r["name"], rt_r["name"]))
    writer.release()

    out_size_mb = OUT_VIDEO.stat().st_size / (1024 * 1024)
    print(f"\nComparison video: {OUT_VIDEO}  ({out_size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
