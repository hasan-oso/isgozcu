"""PPE detection on a single still image.

Usage:
    python backend/inference/test_image.py --input videos/samples/test_image_01.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Optional

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# BGR colors per the Phase 1.6 spec.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Hardhat": (0, 255, 0),
    "Safety Vest": (0, 255, 0),
    "NO-Hardhat": (0, 0, 255),
    "NO-Safety Vest": (0, 0, 255),
    "Person": (255, 100, 0),
    "Mask": (255, 255, 0),
    "Safety Cone": (0, 165, 255),
    "machinery": (255, 0, 255),
    "vehicle": (0, 200, 255),
    "NO-Mask": (0, 0, 200),
}
DEFAULT_COLOR = (200, 200, 200)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _ensure_env() -> None:
    env = PROJECT_ROOT / ".env"
    example = PROJECT_ROOT / ".env.example"
    if not env.exists() and example.exists():
        env.write_bytes(example.read_bytes())
    load_dotenv(env)


def _draw_detections(
    img: np.ndarray,
    boxes: np.ndarray,
    cls_idx: np.ndarray,
    confs: np.ndarray,
    class_names: dict[int, str],
) -> None:
    for (x1, y1, x2, y2), c, conf in zip(boxes, cls_idx, confs):
        name = class_names.get(int(c), str(c))
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1 - 6, th + 6)
        cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 4, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def _print_detection_table(
    boxes: np.ndarray,
    cls_idx: np.ndarray,
    confs: np.ndarray,
    class_names: dict[int, str],
) -> None:
    print()
    print("Per-detection table")
    print("-" * 78)
    print(f"{'#':>3}  {'class':<18}{'conf':>7}{'x1':>6}{'y1':>6}{'x2':>6}{'y2':>6}{'w':>6}{'h':>6}")
    print("-" * 78)
    for i, ((x1, y1, x2, y2), c, conf) in enumerate(
        sorted(
            zip(boxes, cls_idx, confs),
            key=lambda kv: (class_names.get(int(kv[1]), str(kv[1])), -float(kv[2])),
        ),
        1,
    ):
        name = class_names.get(int(c), str(c))
        w = x2 - x1
        h = y2 - y1
        print(f"{i:>3}  {name:<18}{conf:>7.3f}{x1:>6}{y1:>6}{x2:>6}{y2:>6}{w:>6}{h:>6}")


def _summarize(counts: dict[str, list[float]]) -> None:
    persons = len(counts.get("Person", []))
    hardhat = len(counts.get("Hardhat", []))
    no_hardhat = len(counts.get("NO-Hardhat", []))
    vest = len(counts.get("Safety Vest", []))
    no_vest = len(counts.get("NO-Safety Vest", []))

    print()
    print("Class summary")
    print("-" * 60)
    print(f"  Persons:                  {persons}")
    print(f"  Hardhat detections:       {hardhat}")
    print(f"  NO-Hardhat detections:    {no_hardhat}")
    print(f"  Safety Vest detections:   {vest}")
    print(f"  NO-Safety Vest dets:      {no_vest}")
    other_classes = sorted(
        k for k in counts if k not in {"Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"}
    )
    for k in other_classes:
        print(f"  {k}: {len(counts[k])}")

    print()
    print("Confidence per class (min / avg / max)")
    print("-" * 60)
    for k in sorted(counts):
        vals = counts[k]
        print(f"  {k:<18}  min {min(vals):.3f}   avg {mean(vals):.3f}   max {max(vals):.3f}")

    print()
    print("Compliance rates")
    print("-" * 60)
    if hardhat + no_hardhat > 0:
        rate = hardhat / (hardhat + no_hardhat)
        print(f"  Hardhat:  {hardhat}/{hardhat + no_hardhat} = {rate * 100:.1f}%")
    else:
        print("  Hardhat:  N/A (no helmet/no-helmet detections)")
    if vest + no_vest > 0:
        rate = vest / (vest + no_vest)
        print(f"  Vest:     {vest}/{vest + no_vest} = {rate * 100:.1f}%")
    else:
        print("  Vest:     N/A (no vest/no-vest detections)")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-image PPE detection.")
    p.add_argument("--input", required=True, help="Path to input image.")
    p.add_argument("--output", default=None, help="Path to annotated output image.")
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default 0.3).")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_env()
    args = _parse_args(argv)

    in_path = _resolve(args.input)
    if not in_path.exists():
        print(f"[HATA] Görsel bulunamadı / image missing: {in_path}")
        return 1

    if args.output:
        out_path = _resolve(args.output)
    else:
        out_path = PROJECT_ROOT / "data" / "snapshots" / f"{in_path.stem}_detected.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(os.getenv("YOLO_MODEL_PATH", "backend/models/ppe_yolov8n.pt"))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print(f"[HATA] Model yok / model missing: {model_path}")
        return 1

    print(f"Görsel / image: {in_path}")
    print(f"Model:          {model_path.name}")
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"[HATA] cv2.imread başarısız / failed to read.")
        return 1
    h, w = img.shape[:2]
    print(f"Resolution:     {w} x {h}")

    yolo = YOLO(str(model_path))
    class_names = yolo.names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:         {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = yolo.predict(
        source=img,
        conf=args.conf,
        imgsz=args.imgsz,
        half=(device == "cuda"),
        device=device,
        verbose=False,
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        print("Tespit yok / no detections.")
        cv2.imwrite(str(out_path), img)
        print(f"Output (unchanged copy): {out_path}")
        return 0

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    cls_idx = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    counts: dict[str, list[float]] = {}
    for c, conf in zip(cls_idx, confs):
        name = class_names.get(int(c), str(c))
        counts.setdefault(name, []).append(float(conf))

    _draw_detections(img, boxes, cls_idx, confs, class_names)
    cv2.imwrite(str(out_path), img)
    print(f"Output:         {out_path}")
    print(f"Total detections: {len(boxes)}")

    _print_detection_table(boxes, cls_idx, confs, class_names)
    _summarize(counts)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\nPeak VRAM:        {peak:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
