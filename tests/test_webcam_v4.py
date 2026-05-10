"""Webcam test for the v4 RT-DETR (6-class Construction-PPE) model.

Same layout as `tests/test_webcam_with_ppe.py` but pinned to v4 — different
class set (Person, helmet, vest, gloves, no_helmet, no_gloves) and a
diagnostic-friendly palette that picks distinct colors for the gloves /
no_gloves pair so the user can tell them apart at a glance.

Run from the project root with the venv activated.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# UTF-8 stdout/stderr for Turkish text on Windows consoles.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.rtdetr_detector import RTDetrDetector  # noqa: E402


# Diagnostic palette per Phase 2A.5.B spec — gloves get cyan and no_gloves get
# orange so they're distinguishable from the helmet/vest greens and no_helmet
# red. Matches the user's spec exactly.
CLASS_COLORS = {
    "Person": (255, 100, 0),     # blue
    "helmet": (0, 200, 0),       # green
    "vest": (0, 200, 0),         # green
    "gloves": (255, 200, 0),     # cyan
    "no_helmet": (0, 0, 255),    # red
    "no_gloves": (0, 100, 255),  # orange
}

CLASS_THICKNESS = {
    "no_helmet": 3,
    "no_gloves": 3,
}

WINDOW_TITLE = "İşGözcü v4 - Webcam Test"


def main():
    print("=" * 60)
    print("İşGözcü v4 - Webcam PPE Detection Test")
    print("=" * 60)

    print("\nLoading RT-DETR v4 model...")
    detector = RTDetrDetector(
        model_version="v4",  # opt-in: existing v3 default unchanged
        device="cuda",
        conf_threshold=0.25,
        use_fp16=True,
    )
    print(f"Model loaded with classes: {detector.class_names}")

    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Try: cv2.VideoCapture(0) without CAP_DSHOW, or check camera index")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam ready: {actual_w}x{actual_h}")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    snapshots_dir = PROJECT_ROOT / "data" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    detection_counts: dict[str, int] = defaultdict(int)
    max_confidences: dict[str, float] = defaultdict(float)
    fps_smoothed = 0.0
    last_time = time.time()
    start_time = time.time()

    print("\nStarting detection loop. Press Q to exit, S to save snapshot.")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame_count += 1

            # Run detection (no tracker for this test — purer signal).
            detections = detector.detect(frame)

            current_time = time.time()
            instant_fps = 1.0 / max(current_time - last_time, 0.001)
            last_time = current_time
            fps_smoothed = (
                0.9 * fps_smoothed + 0.1 * instant_fps if frame_count > 1 else instant_fps
            )

            # Draw detections
            if len(detections) > 0:
                frame_summary = []
                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    class_id = int(detections.class_id[i])
                    conf = float(detections.confidence[i])
                    class_name = detector.class_names[class_id]

                    detection_counts[class_name] += 1
                    max_confidences[class_name] = max(max_confidences[class_name], conf)

                    color = CLASS_COLORS.get(class_name, (128, 128, 128))
                    thickness = CLASS_THICKNESS.get(class_name, 2)

                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    label = f"{class_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    frame_summary.append(f"{class_name}({conf:.2f})")

                if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f"Frame {frame_count}: {', '.join(frame_summary)}")

            # Top-left overlay: title + FPS
            cv2.rectangle(frame, (10, 10), (380, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"IsGozcu v4 - Webcam Test  FPS: {fps_smoothed:.1f}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Top-right overlay: timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.rectangle(frame, (actual_w - 130, 10), (actual_w - 10, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                timestamp,
                (actual_w - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Bottom strip: controls
            cv2.rectangle(frame, (0, actual_h - 40), (actual_w, actual_h), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "[Q] Cikis  [S] Anlik gorunti kaydet",
                (20, actual_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            elif key == ord("s") or key == ord("S"):
                snapshot_path = snapshots_dir / (
                    f"webcam_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(str(snapshot_path), frame)
                print(f"Snapshot saved: {snapshot_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Session Summary (v4)")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Elapsed time: {elapsed:.1f}s")
        if elapsed > 0:
            print(f"Average FPS: {frame_count / elapsed:.2f}")

        print("\nDetections by class:")
        print(f"{'Class':<20} {'Count':>10} {'Max Conf':>12}")
        print("-" * 44)
        for cls in sorted(detection_counts.keys()):
            print(f"{cls:<20} {detection_counts[cls]:>10} {max_confidences[cls]:>12.3f}")

        if not detection_counts:
            print("(No detections)")

        print("=" * 60)


if __name__ == "__main__":
    main()
