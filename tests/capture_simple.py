"""Basit kamera + tespit + S ile kayit"""
import sys, argparse
from pathlib import Path
from datetime import datetime
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.rtdetr_detector import RTDetrDetector


def draw_detections(frame, detections, class_names):
    COLORS = {
        'Hardhat': (0, 200, 0), 'Safety Vest': (0, 200, 0),
        'NO-Hardhat': (0, 0, 255), 'NO-Safety Vest': (0, 0, 255),
        'Mask': (0, 200, 0), 'NO-Mask': (0, 100, 255),
        'Person': (255, 100, 0), 'Safety Cone': (0, 165, 255),
        'machinery': (180, 180, 180), 'vehicle': (180, 180, 180),
    }
    CONF_FLOORS = {
        'Hardhat': 0.50, 'Safety Vest': 0.55, 'NO-Hardhat': 0.55,
        'NO-Safety Vest': 0.60, 'Person': 0.55,
        'Mask': 0.50, 'NO-Mask': 0.60,
    }

    items = []
    if isinstance(detections, dict):
        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        labels = detections.get('labels', [])
        for b, s, l in zip(boxes, scores, labels):
            items.append((b, s, l))
    elif isinstance(detections, list):
        for d in detections:
            if isinstance(d, dict):
                items.append((d.get('bbox') or d.get('box'),
                              d.get('confidence') or d.get('score'),
                              d.get('class_id') or d.get('label')))
            elif isinstance(d, (tuple, list)) and len(d) >= 3:
                items.append((d[0], d[1], d[2]))

    for bbox, conf, cls_id in items:
        if bbox is None or conf is None or cls_id is None:
            continue
        cls_name = class_names[int(cls_id)]
        if float(conf) < CONF_FLOORS.get(cls_name, 0.50):
            continue
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            continue
        color = COLORS.get(cls_name, (200, 200, 200))
        thickness = 3 if 'NO-' in cls_name else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{cls_name} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def add_hud(frame, save_count):
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 45), (26, 22, 13), -1)
    cv2.putText(frame, "ISGOZCU - CCTV_01 LIVE", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (w - 100, 22), 6, (68, 68, 239), -1)
    cv2.putText(frame, "CANLI", (w - 85, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (68, 68, 239), 2, cv2.LINE_AA)
    # Bottom bar
    cv2.rectangle(frame, (0, h - 50), (w, h), (26, 22, 13), -1)
    cv2.putText(frame, f"[S] Goruntu kaydet  |  [Q] Cikis  |  Kayitlar: {save_count}",
                (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="captures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Kayit klasoru: {output_dir.absolute()}")

    print(f"[Info] Kamera baglaniyor: {args.input}")
    try:
        cam_input = int(args.input)
    except ValueError:
        cam_input = args.input

    cap = cv2.VideoCapture(cam_input)
    if not cap.isOpened():
        print(f"[HATA] Kamera acilamadi: {args.input}")
        return 1

    print("[Info] RT-DETR model yukleniyor...")
    detector = RTDetrDetector()
    class_names = detector.class_names
    print("[Info] Model hazir. Kamera penceresi aciliyor...")
    print("\n  >> S tusuna basarak goruntu kaydet")
    print("  >> Q tusuna basarak cikis\n")

    cv2.namedWindow("Yakalama", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yakalama", 1280, 720)

    save_count = 0
    flash_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            detections = detector.detect(frame)
        except Exception:
            detections = []

        display = frame.copy()
        display = draw_detections(display, detections, class_names)
        display = add_hud(display, save_count)

        # White flash effect after save
        import time as _t
        if _t.time() < flash_until:
            overlay = display.copy()
            overlay[:] = (255, 255, 255)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        cv2.imshow("Yakalama", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

        if key == ord('s') or key == ord('S'):
            save_count += 1
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"capture_{save_count:02d}_{timestamp}.jpg"
            output_path = output_dir / filename

            # Save frame WITH detections drawn
            final = frame.copy()
            final = draw_detections(final, detections, class_names)
            cv2.imwrite(str(output_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print(f"[KAYIT] {filename}  ({len(detections) if detections else 0} tespit)")
            flash_until = _t.time() + 0.3

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI - {save_count} goruntu kaydedildi")
    print(f"  Klasor: {output_dir.absolute()}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
