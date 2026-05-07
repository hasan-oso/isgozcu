"""Live stream simulator: loops a video, runs YOLO + ByteTrack, shows a CCTV-style HUD.

Interactive (window) by default. For automated verification a `--headless` mode
runs the full pipeline (timing, HUD compositing, looping, snapshots) without GUI.

Usage:
    python backend/inference/live_simulator.py
    python backend/inference/live_simulator.py --speed 2.0
    python backend/inference/live_simulator.py --input videos/samples/test_video_01.mp4 \
        --camera-id CCTV_03 --conf 0.25
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Windows console may default to a non-UTF8 codepage (e.g. cp1256 on TR/AR
# locales) which crashes when we print Turkish characters. Force UTF-8 on
# stdout/stderr so logs work everywhere.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.detector_adapter import DetectorAdapter  # noqa: E402


_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont"] = {}


def _get_font(size_px: int):
    if size_px in _FONT_CACHE:
        return _FONT_CACHE[size_px]
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",  # Segoe UI — full Turkish support, Windows default
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            f = ImageFont.truetype(path, size_px)
            _FONT_CACHE[size_px] = f
            return f
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# Colors (BGR)
COLOR_PERSON = (200, 100, 0)
COLOR_HARDHAT = (0, 255, 0)
COLOR_VEST = (0, 255, 0)
COLOR_NO_HARDHAT = (0, 0, 255)
COLOR_NO_VEST = (0, 0, 255)
COLOR_OTHER = (128, 128, 128)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)


CLASS_COLORS = {
    "Person": COLOR_PERSON,
    "Hardhat": COLOR_HARDHAT,
    "Safety Vest": COLOR_VEST,
    "NO-Hardhat": COLOR_NO_HARDHAT,
    "NO-Safety Vest": COLOR_NO_VEST,
}
CLASS_THICK = {
    "Person": 2,
    "Hardhat": 2,
    "Safety Vest": 2,
    "NO-Hardhat": 3,
    "NO-Safety Vest": 3,
}
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}

DISPLAY_W, DISPLAY_H = 1280, 720
WINDOW_TITLE = "İşGözcü - Live Stream Simulator"

SPEED_OPTIONS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]


def _ensure_env(root: Path) -> None:
    env = root / ".env"
    example = root / ".env.example"
    if not env.exists() and example.exists():
        env.write_bytes(example.read_bytes())
    load_dotenv(env)


def _put_text_outlined(img, text, org, scale=0.6, fg=COLOR_WHITE, bg=COLOR_BLACK, thickness=1):
    """Draw outlined Unicode-safe text on a BGR numpy array (in place).

    Uses Pillow on a tight ROI around the text bbox so each call is cheap even
    though we round-trip through PIL for proper Turkish-character rendering.
    `org` matches cv2.putText semantics (baseline-bottom-left).
    """
    size_px = max(11, int(round(scale * 22)))
    font = _get_font(size_px)
    stroke = max(2, thickness + 1)
    # font.getbbox -> (x0, y0, x1, y1) for text at origin; gives required pixels
    try:
        bx0, by0, bx1, by1 = font.getbbox(text)
    except (AttributeError, TypeError):  # default bitmap font fallback
        bx0, by0, bx1, by1 = 0, 0, len(text) * size_px // 2, size_px
    x, y = int(org[0]), int(org[1])
    pil_y = y - size_px  # cv2 baseline → PIL top (approx)
    pad = stroke + 1
    rx0 = x + bx0 - pad
    ry0 = pil_y + by0 - pad
    rx1 = x + bx1 + pad
    ry1 = pil_y + by1 + pad
    H, W = img.shape[:2]
    cx0, cy0 = max(0, rx0), max(0, ry0)
    cx1, cy1 = min(W, rx1), min(H, ry1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = img[cy0:cy1, cx0:cx1]
    pil = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(
        (x - cx0, pil_y - cy0),
        text,
        font=font,
        fill=(fg[2], fg[1], fg[0]),
        stroke_width=stroke,
        stroke_fill=(bg[2], bg[1], bg[0]),
    )
    sub[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_translucent_box(frame, x, y, w, h, alpha=0.55):
    h_img, w_img = frame.shape[:2]
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    x, y = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return
    sub = frame[y:y2, x:x2]
    overlay = np.zeros_like(sub)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)


class LiveStreamSimulator:
    def __init__(
        self,
        video_path: Path,
        camera_id: str,
        conf: float,
        speed: float,
        loop_enabled: bool,
        detector_type: str = "rtdetr",
        model_path: Optional[Path] = None,
        headless: bool = False,
        max_frames: Optional[int] = None,
        snapshot_every: Optional[int] = None,
    ) -> None:
        self.video_path = video_path
        self.camera_id = camera_id
        self.conf = conf
        self.speed = speed
        self.loop_enabled = loop_enabled
        self.headless = headless
        self.max_frames = max_frames
        self.snapshot_every = snapshot_every
        self.detector_type = detector_type

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Video açılamadı / could not open: {video_path}")
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_src_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.detector = DetectorAdapter(
            detector_type=detector_type,
            model_path=str(model_path) if model_path else None,
            conf=conf,
            imgsz=640,
        )
        self.class_names = {i: n for i, n in enumerate(self.detector.get_class_names())}
        self.imgsz = 640

        # State
        self.loop_count = 0
        self.total_frames_processed = 0
        self.unique_workers_seen: set[int] = set()
        self.violation_track_ids: set[int] = set()
        self.detection_history: deque = deque(maxlen=300)
        self.detections_per_class: dict[str, int] = defaultdict(int)
        self.session_start_time = time.time()
        self.is_paused = False
        self.pause_started_at: Optional[float] = None
        self.pause_duration_total = 0.0
        self.last_canvas: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_result = None
        self.loop_overlay_until = 0.0
        self.lag_indicator_until = 0.0
        self.fps_ema = 0.0
        self.last_loop_t = time.time()
        self.auto_snapshots_taken = 0

        self.snapshot_dir = PROJECT_ROOT / "data" / "snapshots" / "live_session"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- speed / pause / reset ----------

    @property
    def speed_index(self) -> int:
        if self.speed in SPEED_OPTIONS:
            return SPEED_OPTIONS.index(self.speed)
        return min(range(len(SPEED_OPTIONS)), key=lambda i: abs(SPEED_OPTIONS[i] - self.speed))

    def speed_up(self) -> None:
        self.speed = SPEED_OPTIONS[min(self.speed_index + 1, len(SPEED_OPTIONS) - 1)]

    def speed_down(self) -> None:
        self.speed = SPEED_OPTIONS[max(self.speed_index - 1, 0)]

    def toggle_pause(self) -> None:
        if self.is_paused:
            if self.pause_started_at is not None:
                self.pause_duration_total += time.time() - self.pause_started_at
            self.pause_started_at = None
        else:
            self.pause_started_at = time.time()
        self.is_paused = not self.is_paused

    def reset_stats(self) -> None:
        self.loop_count = 0
        self.total_frames_processed = 0
        self.unique_workers_seen.clear()
        self.violation_track_ids.clear()
        self.detection_history.clear()
        self.detections_per_class = defaultdict(int)
        self.session_start_time = time.time()
        self.pause_duration_total = 0.0
        self.pause_started_at = time.time() if self.is_paused else None

    def session_seconds(self) -> float:
        elapsed = time.time() - self.session_start_time - self.pause_duration_total
        if self.is_paused and self.pause_started_at is not None:
            elapsed -= time.time() - self.pause_started_at
        return max(0.0, elapsed)

    # ---------- frame I/O + detection ----------

    def _read_frame(self):
        ok, frame = self.cap.read()
        if not ok and self.loop_enabled:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.loop_count += 1
            self.loop_overlay_until = time.time() + 1.2
            ok, frame = self.cap.read()
        return ok, frame

    def _detect(self, frame):
        try:
            return self.detector.detect_and_track(frame)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
            if "out of memory" in str(e).lower() and self.imgsz > 416:
                print(f"[UYARI] CUDA OOM at imgsz={self.imgsz}; switching to 416.")
                torch.cuda.empty_cache()
                self.imgsz = 416
                # NOTE: re-creating the detector with smaller imgsz is the right
                # OOM recovery; for now just propagate.
                raise
            raise

    def _update_stats(self, detections: list[dict]) -> tuple[int, int]:
        if not detections:
            self.detection_history.append({"violations": 0, "persons": 0})
            return 0, 0
        active_violations = 0
        persons_in_frame = 0
        for d in detections:
            name = d["class_name"]
            tid = d["track_id"]
            self.detections_per_class[name] += 1
            if name == "Person":
                persons_in_frame += 1
                if tid is not None:
                    self.unique_workers_seen.add(int(tid))
            elif name in VIOLATION_CLASSES:
                active_violations += 1
                if tid is not None:
                    self.violation_track_ids.add(int(tid))
        self.detection_history.append({"violations": active_violations, "persons": persons_in_frame})
        return active_violations, persons_in_frame

    # ---------- drawing ----------

    def _draw_detections(self, canvas, detections: list[dict]) -> None:
        if not detections:
            return
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            name = d["class_name"]
            conf = d["confidence"]
            tid = d["track_id"]
            color = CLASS_COLORS.get(name, COLOR_OTHER)
            thick = CLASS_THICK.get(name, 1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1 - 6, th + 6)
            cv2.rectangle(canvas, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
            cv2.putText(canvas, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1, cv2.LINE_AA)
            if name == "Person" and tid is not None:
                worker_label = f"W{int(tid):03d}"
                _put_text_outlined(canvas, worker_label, (x1, max(y1 - 30, 22)), scale=0.6, fg=COLOR_YELLOW, thickness=2)

    def _draw_hud(self, canvas, active_violations: int, persons_in_frame: int) -> None:
        h, w = canvas.shape[:2]

        # Top-left CAM ID
        _draw_translucent_box(canvas, 8, 8, 280, 40, alpha=0.55)
        _put_text_outlined(canvas, self.camera_id, (20, 38), scale=0.85, thickness=2)

        # Top-right LIVE + timestamp
        _draw_translucent_box(canvas, w - 288, 8, 280, 60, alpha=0.55)
        elapsed = time.time() - self.session_start_time
        opacity = 0.5 + 0.5 * math.sin(elapsed * 2 * math.pi)
        red_intensity = int(80 + 175 * opacity)
        cv2.circle(canvas, (w - 268, 28), 8, (0, 0, red_intensity), -1, cv2.LINE_AA)
        _put_text_outlined(canvas, "LIVE", (w - 248, 35), scale=0.7, thickness=2)
        ts = datetime.now().strftime("%d.%m.%Y - %H:%M:%S")
        _put_text_outlined(canvas, ts, (w - 268, 60), scale=0.55)

        # Bottom-left session info (above the controls strip)
        bl_y = h - 128
        _draw_translucent_box(canvas, 8, bl_y, 320, 80, alpha=0.55)
        sess_s = int(self.session_seconds())
        _put_text_outlined(canvas, f"Oturum: {sess_s // 60:02d}:{sess_s % 60:02d}", (20, bl_y + 26), scale=0.65, thickness=2)
        _put_text_outlined(canvas, f"Tur: #{self.loop_count}", (20, bl_y + 50), scale=0.6, thickness=2)
        _put_text_outlined(canvas, f"FPS: {self.fps_ema:.1f}   Hız: {self.speed:g}x", (20, bl_y + 72), scale=0.55)

        # Bottom-right stats
        br_x = w - 328
        br_y = h - 148
        _draw_translucent_box(canvas, br_x, br_y, 320, 100, alpha=0.55)
        _put_text_outlined(canvas, f"Aktif İşçi: {persons_in_frame}", (br_x + 12, br_y + 24), scale=0.6, thickness=2)
        total_dets = sum(self.detections_per_class.values())
        _put_text_outlined(canvas, f"Toplam Algılama: {total_dets}", (br_x + 12, br_y + 46), scale=0.55)
        viol_color = COLOR_RED if active_violations else COLOR_WHITE
        _put_text_outlined(canvas, f"Aktif İhlal: {active_violations}", (br_x + 12, br_y + 70), scale=0.65, thickness=2, fg=viol_color)
        _put_text_outlined(canvas, f"Toplam İhlal: {len(self.violation_track_ids)}", (br_x + 12, br_y + 92), scale=0.55)

        # Bottom controls strip
        _draw_translucent_box(canvas, 0, h - 40, w, 40, alpha=0.6)
        controls = "[Q] Çıkış   [Space] Duraklat   [S] Anlık Görüntü   [R] Sıfırla   [+/-] Hız"
        _put_text_outlined(canvas, controls, (12, h - 14), scale=0.55)

        # Loop overlay (centered, 1.2s)
        if time.time() < self.loop_overlay_until:
            big = f"TUR #{self.loop_count}"
            (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)
            _put_text_outlined(canvas, big, ((w - tw) // 2, (h + th) // 2), scale=2.5, thickness=6, fg=COLOR_YELLOW)

        # Pause overlay
        if self.is_paused:
            big = "DURAKLATILDI"
            (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
            _put_text_outlined(canvas, big, ((w - tw) // 2, (h + th) // 2), scale=2.0, thickness=5, fg=COLOR_YELLOW)

        # Lag indicator
        if time.time() < self.lag_indicator_until:
            _put_text_outlined(canvas, "LAG", (w // 2 - 30, 80), scale=0.9, thickness=2, fg=COLOR_RED)

    def _resize_for_display(self, frame):
        if (frame.shape[1], frame.shape[0]) != (DISPLAY_W, DISPLAY_H):
            return cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
        return frame

    def _save_snapshot(self, canvas, label: str = "manual") -> Path:
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        path = self.snapshot_dir / f"{ts}_{label}.jpg"
        cv2.imwrite(str(path), canvas)
        return path

    # ---------- main loop ----------

    def run(self) -> int:
        if not self.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, DISPLAY_W, DISPLAY_H)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        next_frame_time = time.time()
        try:
            while True:
                if self.max_frames is not None and self.total_frames_processed >= self.max_frames:
                    print(f"[bilgi] max_frames ({self.max_frames}) ulaşıldı.")
                    break

                proc_start = time.time()

                if self.is_paused:
                    canvas = self.last_canvas.copy() if self.last_canvas is not None else np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                    last_p = self.detection_history[-1]["persons"] if self.detection_history else 0
                    last_v = self.detection_history[-1]["violations"] if self.detection_history else 0
                    # Re-render HUD so timestamp updates
                    base = self._resize_for_display(self.last_frame).copy() if self.last_frame is not None else canvas
                    if self.last_result is not None:
                        self._draw_detections(base, self.last_result)
                    self._draw_hud(base, last_v, last_p)
                    canvas = base
                else:
                    ok, frame = self._read_frame()
                    if not ok:
                        print("[bilgi] Video sona erdi.")
                        break
                    display_frame = self._resize_for_display(frame)
                    detections = self._detect(display_frame)
                    active_v, persons = self._update_stats(detections)
                    canvas = display_frame.copy()
                    self._draw_detections(canvas, detections)
                    self._draw_hud(canvas, active_v, persons)
                    self.last_frame = display_frame
                    self.last_result = detections
                    self.total_frames_processed += 1

                self.last_canvas = canvas

                if not self.headless:
                    cv2.imshow(WINDOW_TITLE, canvas)

                if (
                    self.headless
                    and self.snapshot_every
                    and self.total_frames_processed > 0
                    and self.total_frames_processed % self.snapshot_every == 0
                ):
                    p = self._save_snapshot(canvas, label=f"auto_{self.auto_snapshots_taken:02d}")
                    self.auto_snapshots_taken += 1
                    print(f"[snapshot] {p}")

                # FPS EMA
                now = time.time()
                dt = now - self.last_loop_t
                self.last_loop_t = now
                inst = 1.0 / dt if dt > 0 else 0.0
                self.fps_ema = inst if self.fps_ema == 0 else (0.9 * self.fps_ema + 0.1 * inst)

                # Timing — pace to (source_fps * speed)
                target_dur = 1.0 / max(0.001, self.source_fps * self.speed)
                proc_dur = time.time() - proc_start
                if proc_dur > target_dur and not self.is_paused:
                    self.lag_indicator_until = time.time() + 0.5
                next_frame_time += target_dur
                sleep_for = next_frame_time - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_frame_time = time.time()

                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("[bilgi] Kullanıcı çıkışı.")
                        break
                    elif key == 32:
                        self.toggle_pause()
                        print(f"[bilgi] Duraklatma: {self.is_paused}")
                    elif key in (ord("s"), ord("S")):
                        p = self._save_snapshot(canvas)
                        print(f"[snapshot] {p}")
                    elif key in (ord("r"), ord("R")):
                        self.reset_stats()
                        print("[bilgi] Sayaçlar sıfırlandı.")
                    elif key in (ord("+"), ord("=")):
                        self.speed_up()
                        next_frame_time = time.time()
                        print(f"[bilgi] Hız: {self.speed}x")
                    elif key in (ord("-"), ord("_")):
                        self.speed_down()
                        next_frame_time = time.time()
                        print(f"[bilgi] Hız: {self.speed}x")
        finally:
            self.cleanup()
        return 0

    def cleanup(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        elapsed = self.session_seconds()
        avg_fps = self.total_frames_processed / elapsed if elapsed > 0 else 0.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        print()
        print("=" * 64)
        print("OTURUM ÖZETİ / SESSION SUMMARY")
        print("=" * 64)
        print(f"Toplam süre / total:      {timedelta(seconds=int(elapsed))}")
        print(f"Toplam tur / loops:       {self.loop_count}")
        print(f"İşlenen kare / frames:    {self.total_frames_processed}")
        print(f"Ortalama FPS / avg:       {avg_fps:.2f}")
        print(f"Tepe VRAM / peak:         {peak_vram_mb:.2f} MB")
        print(f"Benzersiz işçi / unique:  {len(self.unique_workers_seen)}")
        print(f"Benzersiz ihlal track ID: {len(self.violation_track_ids)}")
        print()
        print("Sınıf bazında algılama / detections per class:")
        if self.detections_per_class:
            for k, v in sorted(self.detections_per_class.items(), key=lambda kv: -kv[1]):
                print(f"  {k:<20} {v}")
        else:
            print("  (yok / none)")
        print("=" * 64)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IsGozcu live stream simulator with looping CCTV display.")
    p.add_argument("--input", default="videos/samples/test_video_01.mp4")
    p.add_argument("--camera-id", default="CCTV_01")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--detector", choices=["yolo", "rtdetr"], default="rtdetr",
                   help="Which model backend to use (default: rtdetr).")
    p.add_argument("--model-path", default=None,
                   help="Override model path. Defaults: backend/models/ppe_yolov8n.pt for yolo, "
                        "backend/models/rtdetr_isgozcu_ppe_v1 for rtdetr.")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--headless", action="store_true", help="No GUI; for automated testing.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_env(PROJECT_ROOT)
    args = _parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    if not in_path.exists():
        print(f"[HATA] Video yok: {in_path}")
        return 1

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
    else:
        if args.detector == "yolo":
            model_path = PROJECT_ROOT / "backend" / "models" / "ppe_yolov8n.pt"
        else:
            model_path = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v1"
    if not model_path.exists():
        print(f"[HATA] Model yok: {model_path}")
        return 1

    sim = LiveStreamSimulator(
        video_path=in_path,
        camera_id=args.camera_id,
        conf=args.conf,
        speed=args.speed,
        loop_enabled=not args.no_loop,
        detector_type=args.detector,
        model_path=model_path,
        headless=args.headless,
        max_frames=args.max_frames,
        snapshot_every=args.snapshot_every,
    )
    return sim.run()


if __name__ == "__main__":
    sys.exit(main())
