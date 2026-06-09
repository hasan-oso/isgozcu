"""app/live_preview.py - live PPE preview + measurement (Phase 2 engine).

Binds the clean 3-class engine to a phone camera (DroidCam / USB webcam) or a
video file, draws compliant (green) / violator (red) boxes with Turkish labels,
and prints a session summary on exit.

    detector (frame-skip) -> postprocess (Layer 1) -> violations (Layers 2+3)

Run (camera; phone via DroidCam is usually index 1):
    python app/live_preview.py --index 1
Run (no camera; from a video file):
    python app/live_preview.py --source videos/samples/test_video_01.mp4

Keys:  Q = quit    S = save a snapshot of the current frame to app/snapshots/
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import postprocess
from core.detector import DetectorAdapter
from core.labels_tr import tr
from core.violations import IoUMatcher, ViolationDetector

WINDOW = "isgozcu live  [Q]=cikis  [S]=anlik goruntu"
SNAPSHOT_DIR = PROJECT_ROOT / "app" / "snapshots"

# colors (BGR)
GREEN = (60, 200, 60)
RED = (40, 40, 230)
CYAN = (220, 190, 60)
WHITE = (255, 255, 255)

# Turkish -> ASCII for OpenCV's Hershey font (cannot render TR diacritics).
_TR_ASCII = str.maketrans({
    "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
    "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
})


def ascii_tr(key: str) -> str:
    """Turkish string from labels_tr, transliterated to ASCII for OpenCV display."""
    return tr(key).translate(_TR_ASCII)


def open_capture(args) -> "cv2.VideoCapture":
    """Open a video file (--source) or a camera index (CAP_DSHOW on Windows)."""
    if args.source:
        return cv2.VideoCapture(args.source)
    cap = cv2.VideoCapture(int(args.index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(args.index))
    return cap


def draw_overlay(frame, persons, ppe, violators, fps, n_active):
    """Draw PPE boxes, green/red worker boxes with Turkish labels, and a top bar."""
    h, w = frame.shape[:2]

    # PPE boxes (thin cyan) with Turkish class name
    for d in ppe:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 1, cv2.LINE_AA)
        name = ascii_tr("hardhat") if d["class_name"] == "Hardhat" else ascii_tr("vest")
        cv2.putText(frame, name, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

    # person boxes: red if an active violation, green otherwise
    for d in persons:
        tid = d.get("track_id")
        x1, y1, x2, y2 = d["bbox"]
        vtypes = violators.get(tid, [])
        if vtypes:
            color = RED
            label = f"W{tid}: " + ascii_tr(vtypes[0])           # e.g. "Baret takilmamis"
        else:
            color = GREEN
            wid = f"W{tid}: " if tid is not None else ""
            label = wid + ascii_tr("worker_ok")                 # "Kurallara uygun"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if vtypes else 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # top status bar (translucent)
    band_h = min(40, h)
    band = frame[0:band_h, 0:w]
    frame[0:band_h, 0:w] = (band * 0.35).astype(band.dtype)
    bar = f"FPS:{fps:5.1f}   Isci:{len(persons)}   Aktif ihlal:{n_active}"
    cv2.putText(frame, bar, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="Live PPE preview + measurement (3 classes).")
    ap.add_argument("--index", type=int, default=1, help="Camera index (phone via DroidCam ~ 1).")
    ap.add_argument("--detect-every-n", type=int, default=3, help="Run v3 detection every N frames.")
    ap.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold.")
    ap.add_argument("--source", default=None, help="Optional video path instead of a camera.")
    args = ap.parse_args()

    cap = open_capture(args)
    if not cap.isOpened():
        src = args.source if args.source else f"index {args.index}"
        print(f"[HATA] Kaynak acilamadi: {src}")
        return 1

    detector = DetectorAdapter(conf_threshold=args.conf, detect_every_n=args.detect_every_n)
    matcher = IoUMatcher()
    # fps here only sizes the sliding-window deque; the 8s debounce is real-time
    # (driven by wall-clock timestamps), so it is correct at any actual FPS.
    vd = ViolationDetector(fps=30.0)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    start = time.time()
    last_t = start
    fps_ema = 0.0
    frames = 0
    total_violations = 0
    last_canvas = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[bilgi] Kaynak bitti / kare okunamadi.")
                break
            frames += 1
            h, w = frame.shape[:2]
            ts = time.time() - start

            dets = detector.detect_and_track(frame)          # frame-skip inside
            cleaned = postprocess.process(dets, w, h)         # Layer 1
            persons = [d for d in cleaned if d["class_name"] == "Person"]
            hardhats = [d for d in cleaned if d["class_name"] == "Hardhat"]
            vests = [d for d in cleaned if d["class_name"] == "Safety Vest"]

            matches = matcher.match_frame(persons, hardhats, vests, frame_number=frames, timestamp=ts)
            result = vd.process_matches(matches, frame_number=frames, timestamp=ts)  # Layers 2+3
            total_violations += len(result["new"])

            violators: dict[int, list[str]] = {}
            for ev in vd.active():
                violators.setdefault(ev.worker_id, []).append(ev.violation_type)

            # effective-FPS EMA (includes frame-skipped frames = real throughput)
            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * inst)

            last_canvas = draw_overlay(frame, persons, hardhats + vests, violators, fps_ema, len(vd.active()))
            cv2.imshow(WINDOW, last_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")) and last_canvas is not None:
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                out = SNAPSHOT_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(out), last_canvas)
                print(f"[anlik goruntu] {out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    st = detector.stats()
    print("\n==== OTURUM OZETI / SESSION SUMMARY ====")
    print(f"Toplam kare           : {frames}")
    print(f"Sure (s)              : {elapsed:.1f}")
    print(f"Ort. efektif FPS      : {st['effective_fps']}")
    print(f"Ort. cikarim suresi   : {st['avg_inference_ms']} ms  "
          f"({st['inferences']} cikarim, her {st['detect_every_n']} karede bir; soguk baslangic amortise)")
    print(f"Tespit edilen ihlal   : {total_violations}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
