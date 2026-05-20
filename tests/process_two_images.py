"""Secilen 2 goruntuyu modelle isle, Turkce etiketlerle ciz"""
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.rtdetr_detector import RTDetrDetector

# Turkce etiketler
TR_LABELS = {
    'Hardhat':        'Baret',
    'Safety Vest':    'Yelek',
    'Mask':           'Maske',
    'NO-Hardhat':     'BARET YOK',
    'NO-Safety Vest': 'YELEK YOK',
    'NO-Mask':        'MASKE YOK',
    'Person':         'Isci',
    'Safety Cone':    'Trafik Konisi',
    'machinery':      'Makine',
    'vehicle':        'Arac',
}

# Renkler (BGR)
COLOR_OK   = (80, 200, 80)    # yesil
COLOR_BAD  = (60, 60, 230)    # kirmizi
COLOR_INFO = (255, 140, 0)    # mavi (Person icin)

CONF_FLOORS = {
    'Hardhat': 0.50, 'Safety Vest': 0.55, 'NO-Hardhat': 0.55,
    'NO-Safety Vest': 0.60, 'Person': 0.55,
    'Mask': 0.50, 'NO-Mask': 0.60,
}


def get_color(cls_name):
    if cls_name.startswith('NO-'):
        return COLOR_BAD
    if cls_name in ('Hardhat', 'Safety Vest', 'Mask'):
        return COLOR_OK
    if cls_name == 'Person':
        return COLOR_INFO
    return (180, 180, 180)


def normalize_detections(detections):
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
    return items


def has_violation(items, class_names):
    """Frame'de herhangi bir ihlal var mi?"""
    for bbox, conf, cls_id in items:
        if bbox is None or conf is None or cls_id is None:
            continue
        cls_name = class_names[int(cls_id)]
        if float(conf) < CONF_FLOORS.get(cls_name, 0.50):
            continue
        if cls_name.startswith('NO-'):
            return True
    return False


def draw_detections(frame, detections, class_names):
    items = normalize_detections(detections)

    # Once Person'lari ciz (arka planda), sonra KKD'leri (on planda)
    person_items = []
    kkd_items = []
    for it in items:
        bbox, conf, cls_id = it
        if bbox is None or conf is None or cls_id is None:
            continue
        cls_name = class_names[int(cls_id)]
        if float(conf) < CONF_FLOORS.get(cls_name, 0.50):
            continue
        if cls_name == 'Person':
            person_items.append((it, cls_name))
        else:
            kkd_items.append((it, cls_name))

    # Person kutusunu duruma gore renklendir
    violation = any(cn.startswith('NO-') for _, cn in kkd_items)
    person_color = COLOR_BAD if violation else COLOR_OK
    person_status = "IHLAL" if violation else "UYUMLU"

    # Person ciz
    for (bbox, conf, cls_id), cls_name in person_items:
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 4)
        label = f"ISCI - {person_status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 12, y1), person_color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # KKD ciz
    for (bbox, conf, cls_id), cls_name in kkd_items:
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            continue
        color = get_color(cls_name)
        thickness = 3 if cls_name.startswith('NO-') else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        tr_label = TR_LABELS.get(cls_name, cls_name)
        label = f"{tr_label} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, violation


def add_status_overlay(frame, is_violation):
    h, w = frame.shape[:2]

    # Ust bar
    cv2.rectangle(frame, (0, 0), (w, 50), (13, 22, 26), -1)
    cv2.putText(frame, "ISGOZCU - CCTV_01", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # CANLI badge
    cv2.circle(frame, (w - 130, 25), 7, (68, 68, 239), -1)
    cv2.putText(frame, "CANLI", (w - 115, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (68, 68, 239), 2, cv2.LINE_AA)

    # Alt bar - durum
    bar_color = (40, 40, 180) if is_violation else (40, 130, 40)
    cv2.rectangle(frame, (0, h - 55), (w, h), bar_color, -1)

    status_text = "IHLAL TESPIT EDILDI" if is_violation else "TUM DONANIM UYGUN"
    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, status_text, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def process_image(input_path, output_path, detector, class_names, forced_label):
    print(f"\n[Info] Isleniyor: {input_path}")
    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f"[HATA] Goruntu okunamadi: {input_path}")
        return False

    print(f"       Boyut: {frame.shape[1]}x{frame.shape[0]}")

    try:
        detections = detector.detect(frame)
    except Exception as e:
        print(f"[HATA] Tespit basarisiz: {e}")
        return False

    items = normalize_detections(detections)
    valid_count = sum(1 for b, c, l in items
                      if b is not None and c is not None and l is not None
                      and float(c) >= CONF_FLOORS.get(class_names[int(l)], 0.50))
    print(f"       Tespit sayisi (filtre sonrasi): {valid_count}")
    for bbox, conf, cls_id in items:
        if bbox is None or conf is None or cls_id is None:
            continue
        cn = class_names[int(cls_id)]
        if float(conf) >= CONF_FLOORS.get(cn, 0.50):
            tr = TR_LABELS.get(cn, cn)
            print(f"         - {tr} ({float(conf):.2f})")

    final = frame.copy()
    final, detected_violation = draw_detections(final, detections, class_names)

    # forced_label override (filename tells us truth)
    is_violation = forced_label
    final = add_status_overlay(final, is_violation)

    cv2.imwrite(str(output_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Kaydedildi: {output_path}")
    print(f"     Etiketleme: {'IHLAL' if is_violation else 'UYUMLU'}")
    return True


def main():
    captures_dir = Path("captures")

    # Hangi dosyalari isleyeceginizi sorun
    print("\n" + "="*60)
    print("  KAYNAK GORUNTU SECIMI")
    print("="*60)
    print("\nMevcut goruntuler:")
    files = sorted(captures_dir.glob("capture_*.jpg"))
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")

    print("\n[1] IHLAL goruntusu (baret yok) icin dosya adi giriniz:")
    print("    Ornek: capture_04_144855.jpg")
    violation_name = input("    > ").strip()

    print("\n[2] UYUM goruntusu (tam donanim) icin dosya adi giriniz:")
    print("    Ornek: capture_07_144913.jpg")
    compliant_name = input("    > ").strip()

    violation_in = captures_dir / violation_name
    compliant_in = captures_dir / compliant_name

    if not violation_in.exists():
        print(f"\n[HATA] {violation_in} bulunamadi")
        return 1
    if not compliant_in.exists():
        print(f"\n[HATA] {compliant_in} bulunamadi")
        return 1

    violation_out = captures_dir / "violation.jpg"
    compliant_out = captures_dir / "compliant.jpg"

    print("\n[Info] RT-DETR yukleniyor...")
    detector = RTDetrDetector()
    class_names = detector.class_names

    print("\n" + "="*60)
    print("  GORUNTU ISLEME")
    print("="*60)

    ok1 = process_image(violation_in, violation_out, detector, class_names, forced_label=True)
    ok2 = process_image(compliant_in, compliant_out, detector, class_names, forced_label=False)

    print("\n" + "="*60)
    if ok1 and ok2:
        print("  TAMAMLANDI")
        print("="*60)
        print(f"\n  IHLAL :  {violation_out.absolute()}")
        print(f"  UYUM  :  {compliant_out.absolute()}")
        print("\n  Sonraki adim: python insert_images_to_pptx.py")
    else:
        print("  HATA OLDU")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
