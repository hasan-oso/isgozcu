"""Otomatik Goruntu Yakalama - IsGozcu Sunum"""
import argparse, time, sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.rtdetr_detector import RTDetrDetector


def draw_detections(frame, detections, class_names):
    COLORS = {
        'Hardhat': (0, 200, 0), 'Safety Vest': (0, 200, 0),
        'NO-Hardhat': (0, 0, 255), 'NO-Safety Vest': (0, 0, 255),
        'Mask': (0, 200, 0), 'NO-Mask': (0, 100, 255),
        'Person': (255, 100, 0),
    }
    CONF_FLOORS = {
        'Hardhat': 0.50, 'Safety Vest': 0.55, 'NO-Hardhat': 0.55,
        'NO-Safety Vest': 0.60, 'Person': 0.55, 'Mask': 0.50, 'NO-Mask': 0.60,
    }

    if isinstance(detections, dict):
        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        labels = detections.get('labels', [])
        iterator = zip(boxes, scores, labels)
    elif isinstance(detections, list):
        items = []
        for d in detections:
            if isinstance(d, dict):
                items.append((d.get('bbox') or d.get('box'),
                              d.get('confidence') or d.get('score'),
                              d.get('class_id') or d.get('label')))
            elif isinstance(d, (tuple, list)) and len(d) >= 3:
                items.append((d[0], d[1], d[2]))
        iterator = items
    else:
        iterator = []

    for bbox, conf, cls_id in iterator:
        if bbox is None or conf is None or cls_id is None:
            continue
        cls_name = class_names[int(cls_id)]
        floor = CONF_FLOORS.get(cls_name, 0.50)
        if float(conf) < floor:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        color = COLORS.get(cls_name, (200, 200, 200))
        thickness = 3 if 'NO-' in cls_name else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{cls_name} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def add_overlay(frame, title, countdown):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (26, 22, 13), -1)
    cv2.putText(frame, "ISGOZCU - CCTV_01 LIVE", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (w - 110, 25), 6, (68, 68, 239), -1)
    cv2.putText(frame, "CANLI", (w - 95, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (68, 68, 239), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 60), (w, h), (26, 22, 13), -1)
    cv2.putText(frame, title, (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if countdown is not None:
        text = str(countdown) if countdown > 0 else "YAKALA!"
        fs = 4 if countdown > 0 else 2
        th_t = 8 if countdown > 0 else 5
        color = (0, 0, 255) if countdown > 0 else (0, 255, 0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th_t)
        x = (w - tw) // 2
        y = (h + th) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 40, y - th - 30), (x + tw + 40, y + 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, th_t, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="captures")
    parser.add_argument("--prep-time", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Opening camera: {args.input}")
    try:
        cam_input = int(args.input)
    except ValueError:
        cam_input = args.input

    cap = cv2.VideoCapture(cam_input)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera: {args.input}")
        return 1

    print("[Info] Loading RT-DETR model...")
    detector = RTDetrDetector()
    class_names = detector.class_names

    print(f"[Info] Detection returns type test...")
    ret, test_frame = cap.read()
    if ret:
        test_det = detector.detect(test_frame)
        print(f"[Info] Detection type: {type(test_det).__name__}")
        if isinstance(test_det, list) and len(test_det) > 0:
            print(f"[Info] First item type: {type(test_det[0]).__name__}")
            print(f"[Info] First item: {test_det[0]}")

    captures = [
        ("violation", "1/2: BARET'i CIKAR - IHLAL"),
        ("compliant", "2/2: TAM DONANIM - UYUM"),
    ]

    cv2.namedWindow("Yakalama", cv2.WINDOW_NORMAL)

    for cap_name, instruction in captures:
        print(f"\n{'='*60}\n  YAKALAMA: {cap_name.upper()}\n  {instruction}\n{'='*60}")
        prep_start = time.time()
        while True:
            elapsed = time.time() - prep_start
            remaining = max(0, args.prep_time - int(elapsed))
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                detections = detector.detect(frame)
            except Exception:
                detections = []
            display = frame.copy()
            display = draw_detections(display, detections, class_names)
            if remaining > 3:
                display = add_overlay(display, f"{instruction} - Hazirlik: {remaining}s", None)
            elif remaining > 0:
                display = add_overlay(display, instruction, remaining)
            else:
                display = add_overlay(display, instruction, 0)
                cv2.imshow("Yakalama", display)
                cv2.waitKey(1)
                final = frame.copy()
                final = draw_detections(final, detections, class_names)
                output_path = output_dir / f"{cap_name}.jpg"
                cv2.imwrite(str(output_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"\n[OK] Saved: {output_path}")
                flash = frame.copy()
                flash[:] = (255, 255, 255)
                cv2.imshow("Yakalama", flash)
                cv2.waitKey(200)
                time.sleep(1)
                break
            cv2.imshow("Yakalama", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return 1
            if key == ord('s'):
                prep_start = time.time() - args.prep_time + 3

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*60}\n  HAZIR!\n{'='*60}")
    for cap_name, _ in captures:
        p = output_dir / f"{cap_name}.jpg"
        print(f"  [{'OK' if p.exists() else 'MISSING'}] {p.absolute()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
