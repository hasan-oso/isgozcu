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
SUNUM_HTML = PROJECT_ROOT / "ui" / "sunum.html"
SNAP_DIR = PROJECT_ROOT / "app" / "snapshots"
JPEG_QUALITY = 70        # default MJPEG encode quality (override via --jpeg-quality)
STREAM_FPS_CAP = 20.0    # cap browser stream encode rate; detection runs every frame
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
    def __init__(self, source, index, fallback, detect_every_n, conf,
                 jpeg_quality=JPEG_QUALITY, infer_size=512, stream_fps_cap=STREAM_FPS_CAP,
                 tracker="bytetrack", backend="rtdetr_v3") -> None:
        super().__init__(daemon=True, name="isg-worker")
        self.source = source          # live video path (testing) or None -> camera
        self.index = index            # live camera index
        self.fallback = fallback      # golden fallback video path or None
        self.detect_every_n = detect_every_n
        self.conf = conf
        self.tracker = tracker
        self.backend = backend
        self.jpeg_quality = int(jpeg_quality)
        self.infer_size = int(infer_size)
        self.stream_fps_cap = float(stream_fps_cap)
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
            cv2.imwrite(str(d / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            return f"/snapshots/{day}/{fname}"
        except Exception:
            return None

    # ---- main loop ----
    def run(self) -> None:
        try:
            detector = DetectorAdapter(conf_threshold=self.conf, detect_every_n=self.detect_every_n,
                                       inference_max_size=self.infer_size, tracker=self.tracker,
                                       backend=self.backend)
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
        last_emit = 0.0
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

                # Stream-rate cap: encode for the browser at most stream_fps_cap/s.
                # Detection + violations above run EVERY frame; this only throttles
                # the JPEG encode + buffer churn that competes with inference.
                now_emit = time.time()
                if (now_emit - last_emit) >= (1.0 / self.stream_fps_cap):
                    okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if okj:
                        FRAME_BUFFER.update(buf.tobytes())
                        last_emit = now_emit

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


def start_processing(source=None, index=1, fallback=None, detect_every_n=3, conf=0.5,
                     jpeg_quality=JPEG_QUALITY, infer_size=512, stream_fps_cap=STREAM_FPS_CAP,
                     tracker="bytetrack", backend="rtdetr_v3") -> ProcessingWorker:
    global WORKER
    WORKER = ProcessingWorker(source, index, fallback, detect_every_n, conf,
                              jpeg_quality=jpeg_quality, infer_size=infer_size, stream_fps_cap=stream_fps_cap,
                              tracker=tracker, backend=backend)
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


@app.get("/sunum", response_class=HTMLResponse)
def sunum_page():
    if not SUNUM_HTML.exists():
        return HTMLResponse("<h1>ui/sunum.html bulunamadı</h1>", status_code=500)
    return HTMLResponse(SUNUM_HTML.read_text(encoding="utf-8"))


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


def run_benchmark(index: int, seconds: float = 15.0, conf: float = 0.4) -> int:
    """One-off perf sweep over (infer-size x detect_every_n) on the live camera.

    Reports HONEST true throughput (frames / elapsed, with CAP_PROP_BUFFERSIZE=1 so
    the loop tracks the real camera rate) -- NOT the inflated live fps_ema. A preview
    banner shows the current combo so the user stays in frame. Loads the model for
    inference only; writes nothing.
    """
    import gc
    import torch

    combos = [("A", 512, 3), ("B", 416, 3), ("C", 512, 2), ("D", 416, 2)]
    win = "isgozcu benchmark"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    rows = []
    for name, isz, n in combos:
        cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            print(f"[{name}] kamera acilamadi", flush=True)
            continue
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        det = DetectorAdapter(conf_threshold=conf, detect_every_n=n, inference_max_size=isz)
        matcher = IoUMatcher()
        vd = ViolationDetector(fps=30.0)
        ok, f = cap.read()
        if ok and f is not None:
            det.detect_and_track(f)  # cold warmup, excluded from timing
        c0, t0sum = det._infer_count, det._infer_time_sum
        frames = kept_sum = conf_n = 0
        conf_sum = 0.0
        fno = 0
        start = time.time()
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            h, w = frame.shape[:2]
            dets = det.detect_and_track(frame)
            cleaned = postprocess.process(dets, w, h)
            persons = [d for d in cleaned if d["class_name"] == "Person"]
            hh = [d for d in cleaned if d["class_name"] == "Hardhat"]
            vv = [d for d in cleaned if d["class_name"] == "Safety Vest"]
            ts = fno / 30.0
            vd.process_matches(matcher.match_frame(persons, hh, vv, fno, ts), fno, ts)
            kept_sum += len(dets)
            for d in dets:
                conf_sum += d["confidence"]
                conf_n += 1
            frames += 1
            fno += 1
            disp = frame.copy()
            for d in cleaned:
                b = d["bbox"]
                cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (60, 200, 60), 2)
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(disp, f"{name}: infer={isz} n={n}  ({seconds - (time.time() - start):.0f}s)",
                        (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
        el = max(1e-6, time.time() - start)
        fps = frames / el
        gpu_ms = (det._infer_time_sum - t0sum) / max(1, det._infer_count - c0) * 1000.0
        rows.append((name, isz, n, fps, gpu_ms, kept_sum / max(1, frames), conf_sum / max(1, conf_n), frames))
        print(f"[{name}] infer={isz} n={n} -> fps={fps:.1f} gpu_infer={gpu_ms:.0f}ms "
              f"dets/frame={kept_sum / max(1, frames):.2f} avg_conf={conf_sum / max(1, conf_n):.2f} "
              f"(frames={frames})", flush=True)
        cap.release()
        del det
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.5)
    cv2.destroyAllWindows()
    print("\n==== BENCHMARK (gercek throughput) ====", flush=True)
    print(f"{'combo':<6}{'infer':>6}{'n':>4}{'FPS':>8}{'GPUinfer_ms':>13}{'dets/frame':>12}{'avg_conf':>10}", flush=True)
    for (nm, isz, n, fps, gpu, kpf, avgc, fr) in rows:
        print(f"{nm:<6}{isz:>6}{n:>4}{fps:>8.1f}{gpu:>13.0f}{kpf:>12.2f}{avgc:>10.2f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Isgozcu local kiosk server.")
    ap.add_argument("--source", default=os.environ.get("ISG_SOURCE"),
                    help="Live video file path. If omitted, uses --index camera.")
    ap.add_argument("--index", type=int, default=int(os.environ.get("ISG_INDEX", "1")),
                    help="Live camera index (phone via DroidCam ~ 1).")
    ap.add_argument("--fallback", default=os.environ.get("ISG_FALLBACK"),
                    help="Fallback 'golden' video used on demand / on live failure.")
    ap.add_argument("--detect-every-n", type=int, default=int(os.environ.get("ISG_EVERY_N", "2")),
                    help="Detect every Nth frame (default 2 — benchmark winner with GPU max-perf: "
                         "fresh boxes on motion at ~29fps, full 512 resolution).")
    ap.add_argument("--conf", type=float, default=float(os.environ.get("ISG_CONF", "0.5")))
    ap.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("ISG_JPEG_Q", "70")),
                    help="MJPEG encode quality 1-100 (default 70; lower = lighter stream).")
    ap.add_argument("--infer-size", type=int, default=int(os.environ.get("ISG_INFER_SIZE", "512")),
                    help="Model inference square size (default 512; smaller = faster inference).")
    ap.add_argument("--stream-fps", type=float, default=float(os.environ.get("ISG_STREAM_FPS", "20")),
                    help="Max browser stream encode rate (default 20; detection runs every frame).")
    ap.add_argument("--tracker", default=os.environ.get("ISG_TRACKER", "bytetrack"),
                    choices=["bytetrack", "ocsort"],
                    help="Person tracker (default bytetrack; ocsort = appearance-free, for crowded same-uniform scenes).")
    ap.add_argument("--backend", default=os.environ.get("ISG_BACKEND", "rtdetr_v3"),
                    choices=["rtdetr_v3", "rfdetr"],
                    help="Detection backend (default rtdetr_v3 = v3 via transformers; rfdetr = RF-DETR-Nano).")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ISG_PORT", "8000")))
    ap.add_argument("--benchmark", action="store_true",
                    help="Run a one-off (infer-size x detect_every_n) perf sweep on the camera, then exit.")
    ap.add_argument("--bench-seconds", type=float, default=15.0)
    ap.add_argument("--bench-conf", type=float, default=0.4)
    args = ap.parse_args()

    if args.benchmark:
        return run_benchmark(args.index, args.bench_seconds, args.bench_conf)

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    start_processing(source=args.source, index=args.index, fallback=args.fallback,
                     detect_every_n=args.detect_every_n, conf=args.conf,
                     jpeg_quality=args.jpeg_quality, infer_size=args.infer_size, stream_fps_cap=args.stream_fps,
                     tracker=args.tracker, backend=args.backend)
    import uvicorn
    live = args.source or f"kamera {args.index}"
    print(f"[isgozcu] canli={live}  yedek={args.fallback or 'yok'}  model={args.backend}  izleyici={args.tracker}  port={args.port}")
    print(f"[isgozcu] tarayicida ac: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
