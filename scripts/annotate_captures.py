"""Run RT-DETR + post-processor on a list of still images and write
Turkish-annotated copies next to them.

Output convention:
    Person box → blue (UYUMLU) if hardhat + vest both detected, red
    (UYUMSUZ) otherwise. Each PPE detection gets its own box:
        Baret VAR    (green)   Hardhat
        Baret YOK    (red)     NO-Hardhat
        Yelek VAR    (green)   Safety Vest
        Yelek YOK    (red)     NO-Safety Vest

Usage:
    python scripts/annotate_captures.py img1.jpg img2.jpg ...
    python scripts/annotate_captures.py  # default: the 2 captures in the brief
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows console may default to cp1256 — force UTF-8 so Turkish chars print.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.detector_adapter import DetectorAdapter  # noqa: E402
from backend.inference.post_processor import DetectionPostProcessor  # noqa: E402
from backend.inference.text_render import put_text_outlined  # noqa: E402


# ---------- visual config ----------

COLOR_COMPLIANT = (255, 140, 30)   # blue-ish (BGR) — user asked for BLUE
COLOR_VIOLATION = (40, 40, 220)    # red (BGR)
COLOR_OK = (60, 200, 60)           # green (BGR) — for "VAR" PPE boxes
COLOR_NEUTRAL = (140, 140, 140)    # grey for misc

# Turkish labels per class
LABEL_TR = {
    "Hardhat":        ("Baret VAR",   COLOR_OK),
    "NO-Hardhat":     ("Baret YOK",   COLOR_VIOLATION),
    "Safety Vest":    ("Yelek VAR",   COLOR_OK),
    "NO-Safety Vest": ("Yelek YOK",   COLOR_VIOLATION),
    "Mask":           ("Maske VAR",   COLOR_OK),
    "NO-Mask":        ("Maske YOK",   COLOR_VIOLATION),
}


# ---------- compliance logic ----------


def _center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in(box, px, py) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def assess_compliance(person_bbox, detections):
    """Return (has_hardhat, has_vest, missing_hardhat, missing_vest).

    A PPE class "belongs" to a person if its center falls inside the
    person bbox. Simpler than IoU on a head/torso ROI but robust enough
    for the still-frame annotation case.
    """
    has_h = has_v = miss_h = miss_v = False
    for d in detections:
        cn = d["class_name"]
        if cn not in LABEL_TR:
            continue
        cx, cy = _center(d["bbox"])
        if not _point_in(person_bbox, cx, cy):
            continue
        if cn == "Hardhat":
            has_h = True
        elif cn == "NO-Hardhat":
            miss_h = True
        elif cn == "Safety Vest":
            has_v = True
        elif cn == "NO-Safety Vest":
            miss_v = True
    return has_h, has_v, miss_h, miss_v


def compliance_label(has_h, has_v, miss_h, miss_v) -> tuple[str, tuple[int, int, int]]:
    """Return (label_tr, color_bgr) for the person box."""
    # A person is "uyumlu" only when both hardhat AND vest are positively
    # detected, and neither NO-* class fired on them.
    if has_h and has_v and not miss_h and not miss_v:
        return "UYUMLU", COLOR_COMPLIANT
    # Build a precise violation reason in Turkish so the operator knows
    # what's missing.
    missing = []
    if miss_h or not has_h:
        missing.append("Baret")
    if miss_v or not has_v:
        missing.append("Yelek")
    reason = " + ".join(missing) if missing else "İhlal"
    return f"UYUMSUZ ({reason} yok)", COLOR_VIOLATION


# ---------- drawing ----------


def draw_filled_label(canvas, text, anchor_xy, color, scale=0.7, thickness=2,
                      fg=(255, 255, 255)):
    """Outlined Turkish text on top of a translucent filled rect."""
    x, y = anchor_xy
    # Rough text size — Pillow path inside put_text_outlined will redraw
    # at the requested px size. Use cv2 measure just for the background box.
    size_px = max(11, int(round(scale * 22)))
    (tw, th), _ = cv2.getTextSize("Mq", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    approx_w = max(60, int(len(text) * size_px * 0.55) + 12)
    bg_x0, bg_y0 = x - 2, y - size_px - 6
    bg_x1, bg_y1 = x + approx_w, y + 4
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bg_x0, bg_y0), (bg_x1, bg_y1), color, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    put_text_outlined(canvas, text, (x + 4, y - 2), scale=scale,
                      fg=fg, bg=(0, 0, 0), thickness=thickness)


def annotate(canvas: np.ndarray, detections: list) -> dict:
    """Draw everything on `canvas` in place. Returns a summary dict."""
    persons = [d for d in detections if d["class_name"] == "Person"]
    other = [d for d in detections if d["class_name"] in LABEL_TR]

    # 1) PPE boxes first so person box draws on top.
    for d in other:
        cn = d["class_name"]
        label_tr, color = LABEL_TR[cn]
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_filled_label(
            canvas, f"{label_tr} {d['confidence']:.2f}",
            (x1, max(y1 - 4, 18)), color, scale=0.55, thickness=2,
        )

    # 2) Person boxes — blue if compliant, red otherwise.
    summary = {"persons": 0, "compliant": 0, "violations": 0, "details": []}
    for p in persons:
        summary["persons"] += 1
        x1, y1, x2, y2 = p["bbox"]
        has_h, has_v, miss_h, miss_v = assess_compliance(p["bbox"], other)
        label, color = compliance_label(has_h, has_v, miss_h, miss_v)
        if color == COLOR_COMPLIANT:
            summary["compliant"] += 1
        else:
            summary["violations"] += 1
        summary["details"].append({
            "person_bbox": p["bbox"],
            "confidence": p["confidence"],
            "has_hardhat": has_h, "has_vest": has_v,
            "missing_hardhat": miss_h, "missing_vest": miss_v,
            "label_tr": label,
        })
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        # Big "İşçi - UYUMLU/UYUMSUZ" tag above the person box.
        tag = f"İşçi: {label}"
        draw_filled_label(
            canvas, tag, (x1, max(y1 - 14, 30)), color,
            scale=0.75, thickness=2,
        )

    return summary


# ---------- main ----------


def _raw_detect_as_dicts(detector: DetectorAdapter, frame) -> list[dict]:
    """Run RT-DETR detection WITHOUT the tracker (which needs ≥2 consecutive
    frames before emitting any IDs, and would swallow everything on a single
    still image). Convert sv.Detections → list[StandardDetection]."""
    sv_dets = detector.model.detect(frame)  # underlying RTDetrDetector.detect
    out: list[dict] = []
    if len(sv_dets) == 0:
        return out
    for i in range(len(sv_dets)):
        x1, y1, x2, y2 = sv_dets.xyxy[i].astype(int).tolist()
        cid = int(sv_dets.class_id[i])
        out.append({
            "bbox": (x1, y1, x2, y2),
            "class_id": cid,
            "class_name": detector.model.get_class_name(cid),
            "confidence": float(sv_dets.confidence[i]),
            "track_id": None,
        })
    return out


def process_one(img_path: Path, detector: DetectorAdapter,
                post: DetectionPostProcessor, out_dir: Path,
                use_post: bool = True) -> dict:
    print(f"\n=== {img_path.name} ===")
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[HATA] Görsel okunamadı: {img_path}")
        return {}

    h, w = frame.shape[:2]
    post.update_frame_size(w, h)
    dets = _raw_detect_as_dicts(detector, frame)
    print(f"  ham algılama (post-process öncesi): {len(dets)}")
    for d in sorted(dets, key=lambda x: -x["confidence"])[:8]:
        print(f"    · {d['class_name']:<16} {d['confidence']:.3f}")
    if use_post and post is not None:
        dets = post.process(dets)
        print(f"  post-process sonrası: {len(dets)}")

    canvas = frame.copy()
    summary = annotate(canvas, dets)

    # Header banner: filename + tally.
    banner = (
        f"{img_path.name}   |   İşçi: {summary['persons']}   "
        f"Uyumlu: {summary['compliant']}   Uyumsuz: {summary['violations']}"
    )
    cv2.rectangle(canvas, (0, 0), (w, 44), (0, 0, 0), -1)
    put_text_outlined(canvas, banner, (12, 30), scale=0.7, thickness=2)

    out_path = out_dir / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"  → {out_path}")
    for d in summary["details"]:
        print(f"  İşçi conf={d['confidence']:.2f}  {d['label_tr']}  "
              f"(baret={'+' if d['has_hardhat'] else '-'}{'!' if d['missing_hardhat'] else ''}, "
              f"yelek={'+' if d['has_vest'] else '-'}{'!' if d['missing_vest'] else ''})")
    print(f"  Toplam algılama: {len(dets)}  ({summary['persons']} işçi, "
          f"{len(dets) - summary['persons']} KKD)")
    return {"path": str(out_path), **summary}


def main():
    captures_dir = PROJECT_ROOT / "captures"
    out_dir = captures_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        images = [Path(p) for p in sys.argv[1:]]
    else:
        images = [
            captures_dir / "capture_04_144855.jpg",
            captures_dir / "capture_07_144913.jpg",
        ]
    for p in images:
        if not p.exists():
            print(f"[HATA] Görsel yok: {p}")
            sys.exit(1)

    model_path = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v3"
    detector = DetectorAdapter(
        detector_type="rtdetr",
        model_path=model_path,
        conf=0.25,   # low floor; post-processor will tighten it per class
        imgsz=640,
    )
    post = DetectionPostProcessor(
        class_names=detector.get_class_names(),
        frame_width=1280, frame_height=720,
    )
    print(f"[bilgi] Model: {model_path.name}   Post-process: ON")

    results = []
    for img in images:
        results.append(process_one(img, detector, post, out_dir))

    print("\n" + "=" * 64)
    print(f"Tamamlandı — {len(results)} görsel işlendi → {out_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
