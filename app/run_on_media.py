"""app/run_on_media.py - TEMP demo: run the Phase 2 engine on EXISTING media.

Runs the full pipeline (core.detector -> core.postprocess -> core.violations)
on a sample video, and runs detector->postprocess on a few training images,
saving annotated outputs to app/demo_output/. No camera. Safe to delete.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import postprocess
from core.detector import DetectorAdapter
from core.labels_tr import tr
from core.violations import IoUMatcher, ViolationDetector

OUT_DIR = PROJECT_ROOT / "app" / "demo_output"

GREEN = (60, 200, 60)
RED = (40, 40, 230)
CYAN = (220, 190, 60)
BLUE = (220, 130, 50)
ORANGE = (40, 150, 250)
WHITE = (255, 255, 255)
CLASS_COLOR = {"Person": BLUE, "Hardhat": GREEN, "Safety Vest": ORANGE}
CLASS_TR_KEY = {"Person": "person", "Hardhat": "hardhat", "Safety Vest": "vest"}

_TR_ASCII = str.maketrans({
    "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
    "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
})


def ascii_tr(key: str) -> str:
    return tr(key).translate(_TR_ASCII)


def find_video() -> "Path | None":
    samples = sorted((PROJECT_ROOT / "videos" / "samples").glob("*.mp4"))
    if samples:
        return samples[0]
    other = sorted((PROJECT_ROOT / "videos").glob("*.mp4"))
    return other[0] if other else None


def pick_images() -> list:
    base = PROJECT_ROOT / "data" / "training_images"
    plan = (("full_ppe", 3), ("no_hardhat", 3), ("no_vest", 2))
    out = []
    for mode, k in plan:
        out.extend(sorted((base / mode).glob("*.jpg"))[:k])
    return out


def draw_video(frame, persons, ppe, violators, fps, n_active):
    h, w = frame.shape[:2]
    for d in ppe:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, ascii_tr(CLASS_TR_KEY[d["class_name"]]), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
    for d in persons:
        tid = d.get("track_id")
        x1, y1, x2, y2 = d["bbox"]
        vt = violators.get(tid, [])
        color = RED if vt else GREEN
        if vt:
            label = f"W{tid}: " + ascii_tr(vt[0])
        else:
            label = (f"W{tid}: " if tid is not None else "") + ascii_tr("worker_ok")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if vt else 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    bh = min(40, h)
    band = frame[0:bh, 0:w]
    frame[0:bh, 0:w] = (band * 0.35).astype(band.dtype)
    cv2.putText(frame, f"FPS:{fps:5.1f}   Isci:{len(persons)}   Aktif ihlal:{n_active}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    return frame


def draw_image(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        color = CLASS_COLOR.get(d["class_name"], WHITE)
        label = f"{ascii_tr(CLASS_TR_KEY.get(d['class_name'], d['class_name']))} {d['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def run_video(detector, video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[HATA] Video acilamadi: {video_path}")
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUT_DIR / "annotated_demo.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
    if not writer.isOpened():
        print(f"[uyari] VideoWriter acilamadi; video islenecek ama kaydedilmeyecek.")
        writer = None

    matcher = IoUMatcher()
    vd = ViolationDetector(fps=src_fps)
    fps_ema = 0.0
    last_t = time.time()
    frames = 0
    total_new = 0
    print(f"[video] {video_path.name}  {w}x{h} @ {src_fps:.1f}fps  total~{total}  (max {max_frames})")
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames += 1
        ts = frames / src_fps
        dets = detector.detect_and_track(frame)
        cleaned = postprocess.process(dets, w, h)
        persons = [d for d in cleaned if d["class_name"] == "Person"]
        hardhats = [d for d in cleaned if d["class_name"] == "Hardhat"]
        vests = [d for d in cleaned if d["class_name"] == "Safety Vest"]
        matches = matcher.match_frame(persons, hardhats, vests, frames, ts)
        res = vd.process_matches(matches, frames, ts)
        total_new += len(res["new"])
        violators = {}
        for ev in vd.active():
            violators.setdefault(ev.worker_id, []).append(ev.violation_type)
        now = time.time()
        dt = now - last_t
        last_t = now
        inst = 1.0 / dt if dt > 0 else 0.0
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        annotated = draw_video(frame, persons, hardhats + vests, violators, fps_ema, len(vd.active()))
        if writer is not None:
            writer.write(annotated)
        if frames % 100 == 0:
            print(f"  ...{frames} kare islendi")
    cap.release()
    if writer is not None:
        writer.release()
    stats = detector.stats()
    all_events = vd.completed() + vd.active()
    by_type = dict(Counter(e.violation_type for e in all_events))
    return {"out": out_path if writer is not None else None, "frames": frames, "stats": stats,
            "n_violations": len(all_events), "by_type": by_type, "video": video_path.name}


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo: run the Phase 2 engine on existing media.")
    ap.add_argument("--detect-every-n", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.4, help="detector conf; postprocess floor (0.50) is the real gate")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = DetectorAdapter(conf_threshold=args.conf, detect_every_n=args.detect_every_n)

    print("=" * 60)
    print("VIDEO PHASE")
    print("=" * 60)
    video = find_video()
    vinfo = run_video(detector, video, args.max_frames) if video else None
    if vinfo is None:
        print("[bilgi] Uygun video bulunamadi; video fazi atlandi.")

    print("\n" + "=" * 60)
    print("IMAGE PHASE (detector -> postprocess)")
    print("=" * 60)
    images = pick_images()
    saved = []
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[uyari] okunamadi: {p}")
            continue
        h, w = img.shape[:2]
        detector.reset()  # fresh per-image detection (no cross-image tracking)
        raw = detector.detect_and_track(img)
        cleaned = postprocess.process(raw, w, h)
        out = OUT_DIR / f"sample_{i:02d}.jpg"
        cv2.imwrite(str(out), draw_image(img, cleaned))
        saved.append(out)
        summary = ", ".join(f"{d['class_name']}={d['confidence']:.2f}" for d in cleaned) or "(kesif yok)"
        print(f"sample_{i:02d}  [{p.parent.name}/{p.name}]  raw={len(raw)} -> kept={len(cleaned)}  :: {summary}")

    print("\n" + "=" * 60)
    print("DEMO OZETI / SUMMARY")
    print("=" * 60)
    if vinfo:
        s = vinfo["stats"]
        print(f"Video                : {vinfo['video']} -> {vinfo['out']}")
        print(f"  islenen kare       : {vinfo['frames']}")
        print(f"  ort. efektif FPS   : {s['effective_fps']}")
        print(f"  ort. cikarim (ms)  : {s['avg_inference_ms']}  ({s['inferences']} cikarim, her {s['detect_every_n']} karede)")
        print(f"  onaylanan ihlal    : {vinfo['n_violations']}  {vinfo['by_type']}")
    print(f"Kaydedilen goruntu   : {len(saved)}")
    print(f"Cikti klasoru        : {OUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
