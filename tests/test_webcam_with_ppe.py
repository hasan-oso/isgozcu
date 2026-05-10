import cv2
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import os
import sys

# Force UTF-8 console output for Turkish text
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Make `backend.*` importable regardless of CWD (consistent with other tests).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from backend.inference.rtdetr_detector import RTDetrDetector

CLASS_COLORS = {
    'Hardhat': (0, 200, 0),
    'Safety Vest': (0, 200, 0),
    'NO-Hardhat': (0, 0, 255),
    'NO-Safety Vest': (0, 0, 255),
    'NO-Mask': (0, 0, 255),
    'Mask': (0, 200, 0),
    'Person': (255, 100, 0),
    'Safety Cone': (0, 165, 255),
    'machinery': (200, 0, 200),
    'vehicle': (0, 200, 255),
}

CLASS_THICKNESS = {
    'NO-Hardhat': 3,
    'NO-Safety Vest': 3,
    'NO-Mask': 3,
}


def main():
    print("=" * 60)
    print("İşGözcü - Webcam PPE Detection Test")
    print("=" * 60)

    # Phase 2A.6: pinned to v5 to validate the Construction-PPE fine-tune.
    # NOTE: v5's training showed overfitting signs (best val loss at epoch
    # 0.99, regressed afterwards). This test is the practical gate for
    # whether v5 should replace v3 as the production default. If v5 looks
    # better than the v3 webcam baseline (Hardhat 1484, Vest 1862:1698
    # flicker, shadow false positives) → we flip the default. If worse,
    # we change `rtdetr_isgozcu_ppe_v5` below back to `_v3` and roll back.
    print("\nLoading RT-DETR v5 model...")
    detector = RTDetrDetector(
        model_path=str(PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v5"),
        device="cuda",
        conf_threshold=0.25,
        use_fp16=True,
    )
    print(f"Model loaded with classes: {detector.class_names}")

    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DSHOW is more reliable on Windows
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

    cv2.namedWindow("İşGözcü - Webcam Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("İşGözcü - Webcam Test", 1280, 720)

    snapshots_dir = PROJECT_ROOT / "data" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    detection_counts = defaultdict(int)
    max_confidences = defaultdict(float)
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

            # Run detection
            detections = detector.detect(frame)

            # Calculate FPS
            current_time = time.time()
            instant_fps = 1.0 / max(current_time - last_time, 0.001)
            last_time = current_time
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if frame_count > 1 else instant_fps

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
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    frame_summary.append(f"{class_name}({conf:.2f})")

                if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f"Frame {frame_count}: {', '.join(frame_summary)}")

            # Top-left overlay
            cv2.rectangle(frame, (10, 10), (380, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"IsGozcu - Webcam Test  FPS: {fps_smoothed:.1f}",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Top-right overlay
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.rectangle(frame, (actual_w - 130, 10), (actual_w - 10, 50), (0, 0, 0), -1)
            cv2.putText(frame, timestamp, (actual_w - 120, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Bottom strip
            cv2.rectangle(frame, (0, actual_h - 40), (actual_w, actual_h), (0, 0, 0), -1)
            cv2.putText(frame, "[Q] Cikis  [S] Anlik gorunti kaydet",
                        (20, actual_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("İşGözcü - Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                snapshot_path = snapshots_dir / f"webcam_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                print(f"Snapshot saved: {snapshot_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Session Summary")
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
