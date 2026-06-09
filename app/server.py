"""app/server.py - local kiosk server with safety-net + evidence snapshots.

A background thread runs the Phase 2 pipeline (core.detector -> core.postprocess
-> core.violations) on a source, draws annotated frames, publishes the latest
JPEG, and saves an evidence snapshot whenever a violation is CONFIRMED (after the
8s debounce). It can hot-switch between a LIVE source (camera / --source) and a
FALLBACK video (--fallback) without restarting -- manually via POST /toggle-source
or automatically after repeated live-frame read failures.

On-frame text uses ASCII transliteration (OpenCV Hershey font); the /stats JSON
and the HTML UI use FULL Turkish Unicode.

Run:
    python app/server.py --source videos/samples/sample_01.mp4 --fallback videos/samples/sample_01.mp4
    python app/server.py --index 1 --fallback videos/samples/test_video_01.mp4
then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import postprocess
from core.detector import DetectorAdapter
from core.labels_tr import tr
from core.violations import IoUMatcher, ViolationDetector

UI_HTML = PROJECT_ROOT / "ui" / "kiosk.html"
STATUS_HTML = PROJECT_ROOT / "ui" / "status.html"
SNAP_DIR = PROJECT_ROOT / "app" / "snapshots"
JPEG_QUALITY = 80
FAIL_LIMIT = 30  # consecutive failed live-camera reads -> auto-switch to fallback

GREEN = (60, 200, 60)
RED = (40, 40, 230)
CYAN = (220, 190, 60)
WHITE = (255, 255, 255)
CLASS_TR_KEY = {"Person": "person", "Hardhat": "hardhat", "Safety Vest": "vest"}

# Turkish -> ASCII for ON-FRAME text only (OpenCV Hershey font limitation).
_TR_ASCII = str.maketrans({
    "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
    "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
})


def ascii_tr(key: str) -> str:
    return tr(key).translate(_TR_ASCII)


# ---------------------------------------------------------------- shared state
class FrameBuffer:
    def __init__(self) -> None:
        self._jpeg = None
        self._seq = 0
        self._lock = threading.Lock()

    def update(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1

    def get(self):
        with self._lock:
            return self._jpeg, self._seq


class SharedStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = {"fps": 0.0, "workers": 0, "active_violations": 0, "violations": [], "inference_ms": 0.0}

    def set(self, data: dict) -> None:
        with self._lock:
            self._data = data

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)


FRAME_BUFFER = FrameBuffer()
STATS = SharedStats()
WORKER = None


# ---------------------------------------------------------------- worker thread
class ProcessingWorker(threading.Thread):
    def __init__(self, source, index, fallback, detect_every_n, conf) -> None:
        super().__init__(daemon=True, name="isg-worker")
        self.source = source          # live video path (testing) or None -> camera
        self.index = index            # live camera index
        self.fallback = fallback      # golden fallback video path or None
        self.detect_every_n = detect_every_n
        self.conf = conf
        self.mode = "live"            # "live" | "fallback"
        self.current_label = source if source else f"Kamera {index}"
        self.auto_switched = False
        self.auto_switch_reason = None
        self.error = None
        self._stop = threading.Event()
        self._mode_lock = threading.Lock()
        self._pending = None          # "toggle" | "live" | "fallback"
        self.snapshots = {}           # event_id -> url (worker-thread only)

    # ---- control API (called from request handlers) ----
    def request_mode(self, mode: str) -> None:
        with self._mode_lock:
            self._pending = mode

    def request_toggle(self) -> None:
        self.request_mode("toggle")

    def stop(self) -> None:
        self._stop.set()

    # ---- source helpers ----
    def _open(self, mode: str):
        """Return (cap, is_video, error)."""
        if mode == "fallback":
            if not self.fallback:
                return None, True, "yedek video tanımlı değil"
            return cv2.VideoCapture(self.fallback), True, None
        # live
        if self.source:
            return cv2.VideoCapture(self.source), True, None
        cap = cv2.VideoCapture(int(self.index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(self.index))
        return cap, False, None

    def _label(self, mode: str) -> str:
        if mode == "fallback":
            return "Yedek video"
        return self.source if self.source else f"Kamera {self.index}"

    def _save_snapshot(self, frame, ev):
        """Persist the annotated frame as evidence; return its public URL."""
        try:
            day = datetime.now().strftime("%Y-%m-%d")
            d = SNAP_DIR / day
            d.mkdir(parents=True, exist_ok=True)
            fname = f"W{ev.worker_id:03d}_{ev.violation_type}_{datetime.now():%H%M%S}.jpg"
            cv2.imwrite(str(d / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            return f"/snapshots/{day}/{fname}"
        except Exception:
            return None

    # ---- main loop ----
    def run(self) -> None:
        try:
            detector = DetectorAdapter(conf_threshold=self.conf, detect_every_n=self.detect_every_n)
        except Exception as e:
            self.error = f"model yüklenemedi: {e}"
            return
        cap, cur_is_video, err = self._open(self.mode)
        if cap is None or not cap.isOpened():
            self.error = f"kaynak açılamadı: {self._label(self.mode)} ({err})"
            return
        self.current_label = self._label(self.mode)

        matcher = IoUMatcher()
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if src_fps <= 0:
            src_fps = 30.0
        vd = ViolationDetector(fps=src_fps)
        fps_ema = 0.0
        last_t = time.time()
        frame_no = 0
        consecutive_fail = 0

        while not self._stop.is_set():
            # --- handle a requested manual switch ---
            with self._mode_lock:
                pending = self._pending
                self._pending = None
            if pending:
                target = ("fallback" if self.mode == "live" else "live") if pending == "toggle" else pending
                if target != self.mode:
                    newcap, nv, nerr = self._open(target)
                    if newcap is not None and newcap.isOpened():
                        cap.release()
                        cap, cur_is_video, self.mode = newcap, nv, target
                        self.current_label = self._label(target)
                        self.auto_switched = False
                        self.auto_switch_reason = None
                        detector.reset()
                        consecutive_fail = 0
                    else:
                        if newcap is not None:
                            newcap.release()
                        self.error = f"{target} kaynağına geçilemedi: {nerr}"

            ok, frame = cap.read()
            if not ok or frame is None:
                if cur_is_video:                       # video ended -> loop it
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    detector.reset()
                    continue
                # live camera read failure
                consecutive_fail += 1
                if self.mode == "live" and self.fallback and consecutive_fail >= FAIL_LIMIT:
                    newcap, nv, nerr = self._open("fallback")
                    if newcap is not None and newcap.isOpened():
                        cap.release()
                        cap, cur_is_video, self.mode = newcap, nv, "fallback"
                        self.current_label = self._label("fallback")
                        self.auto_switched = True
                        self.auto_switch_reason = f"canlı kaynak başarısız ({consecutive_fail} kare) -> yedeğe geçildi"
                        detector.reset()
                        consecutive_fail = 0
                time.sleep(0.05)
                continue
            consecutive_fail = 0
            frame_no += 1
            ts = frame_no / src_fps

            try:
                h, w = frame.shape[:2]
                dets = detector.detect_and_track(frame)
                cleaned = postprocess.process(dets, w, h)
                persons = [d for d in cleaned if d["class_name"] == "Person"]
                hardhats = [d for d in cleaned if d["class_name"] == "Hardhat"]
                vests = [d for d in cleaned if d["class_name"] == "Safety Vest"]
                matches = matcher.match_frame(persons, hardhats, vests, frame_no, ts)
                result = vd.process_matches(matches, frame_no, ts)

                violators: dict[int, list[str]] = {}
                for ev in vd.active():
                    violators.setdefault(ev.worker_id, []).append(ev.violation_type)
                self._draw(frame, persons, hardhats + vests, violators, fps_ema, len(vd.active()))

                # evidence snapshot on each newly-confirmed violation
                for ev in result["new"]:
                    url = self._save_snapshot(frame, ev)
                    if url:
                        self.snapshots[ev.event_id] = url

                okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if okj:
                    FRAME_BUFFER.update(buf.tobytes())

                now = datetime.now()
                vlist = [{
                    "type_tr": tr(ev.violation_type),
                    "track_id": ev.worker_id,
                    "duration": round((now - ev.started_at).total_seconds(), 1),
                    "confidence": round(ev.confidence_avg, 2),
                    "started": ev.started_at.strftime("%H:%M:%S"),
                    "snapshot": self.snapshots.get(ev.event_id),
                } for ev in vd.active()]
                STATS.set({"fps": round(fps_ema, 1), "workers": len(persons),
                           "active_violations": len(vlist), "violations": vlist,
                           "inference_ms": round(detector.stats()["avg_inference_ms"], 1)})
            except Exception:
                pass  # a single bad frame must never crash the server

            now_t = time.time()
            dt = now_t - last_t
            last_t = now_t
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst

        cap.release()

    def _draw(self, frame, persons, ppe, violators, fps, n_active) -> None:
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


def start_processing(source=None, index=1, fallback=None, detect_every_n=3, conf=0.5) -> ProcessingWorker:
    global WORKER
    WORKER = ProcessingWorker(source, index, fallback, detect_every_n, conf)
    WORKER.start()
    return WORKER


# ---------------------------------------------------------------- FastAPI app
app = FastAPI(title="İşGözcü Kiosk")


@app.get("/", response_class=HTMLResponse)
def index():
    if not UI_HTML.exists():
        return HTMLResponse("<h1>ui/kiosk.html bulunamadı</h1>", status_code=500)
    return HTMLResponse(UI_HTML.read_text(encoding="utf-8"))


@app.get("/status", response_class=HTMLResponse)
def status_page():
    if not STATUS_HTML.exists():
        return HTMLResponse("<h1>ui/status.html bulunamadı</h1>", status_code=500)
    return HTMLResponse(STATUS_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    jpeg, _ = FRAME_BUFFER.get()
    w = WORKER
    return {
        "ok": True,
        "streaming": jpeg is not None,
        "mode": w.mode if w else None,
        "source": w.current_label if w else None,
        "auto_switched": w.auto_switched if w else False,
        "auto_switch_reason": w.auto_switch_reason if w else None,
        "fallback_available": bool(w.fallback) if w else False,
        "error": w.error if w else None,
    }


@app.get("/stats")
def stats():
    return JSONResponse(STATS.get())


@app.post("/toggle-source")
def toggle_source():
    if WORKER is None:
        return JSONResponse({"ok": False, "error": "işleyici hazır değil"}, status_code=503)
    prev = WORKER.mode
    WORKER.request_toggle()
    return {"ok": True, "previous_mode": prev, "switching_to": "fallback" if prev == "live" else "live"}


@app.get("/snapshots/{day}/{fname}")
def snapshot_file(day: str, fname: str):
    safe = (SNAP_DIR / day / fname).resolve()
    snap_root = SNAP_DIR.resolve()
    if snap_root not in safe.parents or not safe.exists():
        return JSONResponse({"error": "bulunamadı"}, status_code=404)
    return FileResponse(str(safe), media_type="image/jpeg")


@app.get("/stream")
def stream():
    def gen():
        last = -1
        while True:
            jpeg, seq = FRAME_BUFFER.get()
            if jpeg is None:
                time.sleep(0.1)
                continue
            if seq == last:
                time.sleep(0.02)
                continue
            last = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


def main() -> int:
    ap = argparse.ArgumentParser(description="Isgozcu local kiosk server.")
    ap.add_argument("--source", default=os.environ.get("ISG_SOURCE"),
                    help="Live video file path. If omitted, uses --index camera.")
    ap.add_argument("--index", type=int, default=int(os.environ.get("ISG_INDEX", "1")),
                    help="Live camera index (phone via DroidCam ~ 1).")
    ap.add_argument("--fallback", default=os.environ.get("ISG_FALLBACK"),
                    help="Fallback 'golden' video used on demand / on live failure.")
    ap.add_argument("--detect-every-n", type=int, default=int(os.environ.get("ISG_EVERY_N", "3")))
    ap.add_argument("--conf", type=float, default=float(os.environ.get("ISG_CONF", "0.5")))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ISG_PORT", "8000")))
    args = ap.parse_args()

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    start_processing(source=args.source, index=args.index, fallback=args.fallback,
                     detect_every_n=args.detect_every_n, conf=args.conf)
    import uvicorn
    live = args.source or f"kamera {args.index}"
    print(f"[isgozcu] canli={live}  yedek={args.fallback or 'yok'}  port={args.port}")
    print(f"[isgozcu] tarayicida ac: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
