"""Headless end-to-end smoke test for the live pipeline.

Reads N webcam frames, applies the virtual PPE filter, runs YOLO on the result,
and reports per-frame detections. No window is shown.

Run from project root:
    python tests/test_live_pipeline_headless.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

from backend.inference.virtual_ppe_filter import VirtualPPEFilter  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "ppe_yolov8n.pt"
N_FRAMES = 12
SNAP_DIR = PROJECT_ROOT / "data" / "snapshots"


def main() -> int:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model: {MODEL_PATH}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Webcam open failed.")
        return 1
    print(
        f"Webcam: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
        f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    filt = VirtualPPEFilter(model_complexity=1)
    yolo = YOLO(str(MODEL_PATH))

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    saved = SNAP_DIR / "live_pipeline_smoke.jpg"

    t0 = time.perf_counter()
    pose_hits = 0
    yolo_total_dets = 0
    classes_seen: dict[str, int] = {}
    last_filtered = None

    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            print(f"[frame {i}] read failed")
            continue

        filtered = filt.process(frame.copy(), hardhat=True, vest=True)
        # crude check: did we modify the frame? (hard to tell perfectly, but mean intensity should change if we drew anything)
        delta = float((filtered.astype("int16") - frame.astype("int16")).__abs__().mean())
        pose_drew = delta > 1.0
        if pose_drew:
            pose_hits += 1

        results = yolo.track(
            source=filtered,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            imgsz=640,
            half=True,
            device="cuda",
            verbose=False,
        )
        r = results[0]
        n = 0 if r.boxes is None else len(r.boxes)
        yolo_total_dets += n
        if n:
            cls_idx = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for c, cf in zip(cls_idx, confs):
                name = yolo.names.get(int(c), str(c))
                classes_seen[name] = classes_seen.get(name, 0) + 1
            print(
                f"[frame {i}] filter_drew={pose_drew} delta={delta:.2f} | "
                f"yolo={[(yolo.names[int(c)], float(cf)) for c, cf in zip(cls_idx, confs)]}"
            )
        else:
            print(f"[frame {i}] filter_drew={pose_drew} delta={delta:.2f} | yolo=[]")
        last_filtered = filtered

    elapsed = time.perf_counter() - t0
    if last_filtered is not None:
        cv2.imwrite(str(saved), last_filtered)

    cap.release()
    filt.close()

    print()
    print(f"Frames processed:     {N_FRAMES}")
    print(f"Pose drew on:         {pose_hits} / {N_FRAMES}")
    print(f"YOLO total detections:{yolo_total_dets}")
    print(f"Classes seen:         {classes_seen}")
    print(f"Elapsed:              {elapsed:.2f}s ({N_FRAMES/elapsed:.2f} FPS)")
    print(f"Last filtered frame:  {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
