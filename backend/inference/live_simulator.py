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
import json
import math
import os
import sys
import time
import uuid
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.detector_adapter import DetectorAdapter  # noqa: E402
from backend.inference.snapshot_saver import SnapshotSaver  # noqa: E402
from backend.inference.text_render import put_text_outlined as _put_text_outlined  # noqa: E402
from backend.inference.violation_engine import (  # noqa: E402
    VIOLATION_LABEL_TR,
    IoUMatcher,
    ViolationDetector,
)


# Supabase imports are lazy: only loaded if --supabase is passed, so the
# default code path stays fast and doesn't error if .env is missing keys.
def _import_supabase():
    from backend.storage.supabase_client import (  # noqa: WPS433
        EventUploader,
        SiteManager,
        SupabaseClient,
        WorkerRegistry,
    )
    return SupabaseClient, SiteManager, WorkerRegistry, EventUploader


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


# put_text_outlined imported above from backend.inference.text_render


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
        video_path: Path | int,
        camera_id: str,
        conf: float,
        speed: float,
        loop_enabled: bool,
        detector_type: str = "rtdetr",
        model_path: Optional[Path] = None,
        headless: bool = False,
        max_frames: Optional[int] = None,
        snapshot_every: Optional[int] = None,
        enable_violations: bool = True,
        min_violation_duration: float = 5.0,
        violation_threshold: float = 0.7,
        recovery_threshold: float = 0.3,
        enable_supabase: bool = False,
        site_name: str = "Demo Şantiye",
    ) -> None:
        self.video_path = video_path
        self.is_webcam = isinstance(video_path, int)
        self.camera_id = camera_id
        self.conf = conf
        self.speed = speed
        # Live webcam streams have no end-of-file → looping is meaningless.
        self.loop_enabled = loop_enabled and not self.is_webcam
        self.headless = headless
        self.max_frames = max_frames
        self.snapshot_every = snapshot_every
        self.detector_type = detector_type
        self.enable_violations = enable_violations

        if self.is_webcam:
            # CAP_DSHOW is the most reliable backend on Windows.
            self.cap = cv2.VideoCapture(int(video_path), cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Video açılamadı / could not open: {video_path}")
        # Webcams often report fps=0 from the driver; fall back to 30 fps so the
        # violation buffer / video-time clock advance at a sane rate.
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.is_webcam and self.source_fps < 1.0:
            self.source_fps = 30.0
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # CAP_PROP_FRAME_COUNT is meaningless / -1 for webcams.
        self.total_src_frames = (
            0 if self.is_webcam else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        self.detector = DetectorAdapter(
            detector_type=detector_type,
            model_path=str(model_path) if model_path else None,
            conf=conf,
            imgsz=640,
        )
        self.class_names = {i: n for i, n in enumerate(self.detector.get_class_names())}
        self.imgsz = 640

        # Violation engine
        self.session_id = uuid.uuid4().hex[:12]
        self.iou_matcher: Optional[IoUMatcher] = None
        self.violation_detector: Optional[ViolationDetector] = None
        self.snapshot_saver: Optional[SnapshotSaver] = None
        if self.enable_violations:
            # Buffer is sized at SOURCE fps because we observe each frame once.
            # End-to-end pipeline FPS doesn't matter for buffer sizing — what
            # matters is how many observations span `min_violation_duration`
            # seconds of video time.
            self.iou_matcher = IoUMatcher()
            self.violation_detector = ViolationDetector(
                min_violation_duration=min_violation_duration,
                violation_threshold=violation_threshold,
                recovery_threshold=recovery_threshold,
                fps=self.source_fps,
                camera_id=camera_id,
            )
            self.snapshot_saver = SnapshotSaver()
        # worker_id -> list of active violation_types (rebuilt every frame)
        self.active_worker_violations: dict[int, list[str]] = {}
        self._session_started_at = datetime.now()

        # Supabase persistence (opt-in via --supabase). Failures degrade
        # gracefully to local-only mode.
        self.supabase_enabled = False
        self.event_uploader = None
        self.worker_registry = None
        self.supabase_status: str = "disabled"  # disabled | idle | syncing | ok | error
        self.supabase_uploaded_count: int = 0
        if enable_supabase:
            try:
                SupabaseClient, SiteManager, WorkerRegistry, EventUploader = _import_supabase()
                sb = SupabaseClient()
                if not sb.health_check():
                    print("[uyarı] Supabase health check başarısız — yerel modda devam.")
                else:
                    site_mgr = SiteManager(sb)
                    site_id = site_mgr.get_or_create_site(site_name, location="Ankara, Türkiye")
                    cam_uuid = site_mgr.get_or_create_camera(site_id, camera_id) if site_id else None
                    if site_id and cam_uuid:
                        self.worker_registry = WorkerRegistry(sb, site_id)
                        self.event_uploader = EventUploader(sb, cam_uuid, self.worker_registry)
                        self.supabase_enabled = True
                        self.supabase_status = "idle"
                        print(f"[bilgi] Supabase aktif (site={site_id[:8]}.. cam={cam_uuid[:8]}..).")
                    else:
                        print("[uyarı] Supabase bağlamı oluşturulamadı — yerel modda devam.")
            except Exception as e:
                print(f"[uyarı] Supabase başlatma hatası: {e} — yerel modda devam.")

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
            # On loop boundary the tracker may reuse old IDs for new physical
            # workers — clear the in-memory tracker_id → worker_uuid cache so
            # we re-resolve via DB on the next sighting.
            if self.worker_registry is not None:
                self.worker_registry.reset_session_cache()
            ok, frame = self.cap.read()
        return ok, frame

    def _video_time(self) -> float:
        """Cumulative video-time seconds across loops (advances at 1/source_fps per processed frame)."""
        return self.total_frames_processed / max(0.001, self.source_fps)

    def _process_violations(self, detections: list[dict], frame_image: np.ndarray) -> None:
        """Run IoU matching + state machine. Updates self.active_worker_violations."""
        assert self.iou_matcher is not None and self.violation_detector is not None

        persons = [d for d in detections if d["class_name"] == "Person" and d["track_id"] is not None]
        groups = {
            "Hardhat": [], "NO-Hardhat": [],
            "Safety Vest": [], "NO-Safety Vest": [],
            "Mask": [], "NO-Mask": [],
        }
        for d in detections:
            if d["class_name"] in groups:
                groups[d["class_name"]].append(d)

        ts = self._video_time()
        matches = self.iou_matcher.match_frame(
            persons=persons,
            hardhats=groups["Hardhat"],
            no_hardhats=groups["NO-Hardhat"],
            vests=groups["Safety Vest"],
            no_vests=groups["NO-Safety Vest"],
            masks=groups["Mask"],
            no_masks=groups["NO-Mask"],
            frame_number=self.total_frames_processed,
            timestamp=ts,
        )
        result = self.violation_detector.process_matches(
            matches=matches,
            frame_number=self.total_frames_processed,
            timestamp=ts,
            frame_image=frame_image,
            snapshot_saver=self.snapshot_saver,
        )

        for ev in result["new"]:
            print(f"[ihlal başladı] W{ev.worker_id:03d} - {ev.violation_label_tr} "
                  f"(conf {ev.confidence_avg:.2f}) -> {ev.snapshot_path}")
            self._supabase_upload_new(ev)
        for ev in result["resolved"]:
            print(f"[ihlal bitti] W{ev.worker_id:03d} - {ev.violation_label_tr} - "
                  f"{ev.duration_seconds:.1f}s sürdü")
            self._supabase_upload_resolution(ev)

        self.active_worker_violations = self.violation_detector.active_worker_violations()

    def _supabase_upload_new(self, ev) -> None:
        if not self.supabase_enabled or self.event_uploader is None:
            return
        self.supabase_status = "syncing"
        try:
            event_uuid = self.event_uploader.upload_event(ev)
            if event_uuid is not None:
                self.supabase_uploaded_count += 1
                self.supabase_status = "ok"
            else:
                self.supabase_status = "error"
        except Exception as e:
            print(f"[uyarı] supabase upload error: {e}")
            self.supabase_status = "error"

    def _supabase_upload_resolution(self, ev) -> None:
        if not self.supabase_enabled or self.event_uploader is None:
            return
        self.supabase_status = "syncing"
        try:
            ok = self.event_uploader.upload_event_resolution(ev)
            self.supabase_status = "ok" if ok else "error"
        except Exception as e:
            print(f"[uyarı] supabase resolution upload error: {e}")
            self.supabase_status = "error"

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
        # Pulsing factor for active violations (1 Hz)
        pulse = 0.5 + 0.5 * math.sin((time.time() - self.session_start_time) * 2 * math.pi)
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            name = d["class_name"]
            conf = d["confidence"]
            tid = d["track_id"]
            color = CLASS_COLORS.get(name, COLOR_OTHER)
            thick = CLASS_THICK.get(name, 1)

            # Active-violation visual treatment for the Person box
            is_violator = (
                name == "Person"
                and tid is not None
                and tid in self.active_worker_violations
            )
            if is_violator:
                color = COLOR_RED
                thick = 4

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1 - 6, th + 6)
            cv2.rectangle(canvas, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
            cv2.putText(canvas, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1, cv2.LINE_AA)

            if name == "Person" and tid is not None:
                worker_label = f"W{int(tid):03d}"
                if is_violator:
                    # Red ID badge with white text
                    badge_color = (int(60 + 195 * pulse), int(60 + 60 * pulse), int(220 + 20 * pulse))
                    (bw, bh), _ = cv2.getTextSize(worker_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    bx, by = x1, max(y1 - 28, bh + 6)
                    cv2.rectangle(canvas, (bx - 2, by - bh - 6), (bx + bw + 8, by + 4), badge_color, -1)
                    _put_text_outlined(canvas, worker_label, (bx + 3, by - 2), scale=0.65, fg=COLOR_WHITE, thickness=2)
                    # Warning text on top showing the most-critical active violation
                    types = self.active_worker_violations.get(tid, [])
                    primary = next(
                        (t for t in ("no_hardhat", "no_vest", "no_mask") if t in types),
                        types[0] if types else None,
                    )
                    if primary:
                        warn = f"⚠ {VIOLATION_LABEL_TR.get(primary, primary)}"
                        _put_text_outlined(
                            canvas, warn, (x1, max(y1 - 56, 22)),
                            scale=0.7, fg=COLOR_RED, thickness=2,
                        )
                else:
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
        # Supabase sync dot (next to LIVE) — green=ok/idle, yellow=syncing,
        # red=error, gray=disabled.
        _sync_color = {
            "ok": (60, 200, 60),
            "idle": (60, 200, 60),
            "syncing": (0, 200, 220),
            "error": (60, 60, 220),
            "disabled": (130, 130, 130),
        }.get(self.supabase_status, (130, 130, 130))
        cv2.circle(canvas, (w - 196, 28), 6, _sync_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (w - 196, 28), 6, COLOR_BLACK, 1, cv2.LINE_AA)
        _put_text_outlined(canvas, "SYNC", (w - 184, 33), scale=0.5)
        ts = datetime.now().strftime("%d.%m.%Y - %H:%M:%S")
        _put_text_outlined(canvas, ts, (w - 268, 60), scale=0.55)

        # Bottom-left session info (above the controls strip)
        bl_y = h - 128
        _draw_translucent_box(canvas, 8, bl_y, 320, 80, alpha=0.55)
        sess_s = int(self.session_seconds())
        _put_text_outlined(canvas, f"Oturum: {sess_s // 60:02d}:{sess_s % 60:02d}", (20, bl_y + 26), scale=0.65, thickness=2)
        _put_text_outlined(canvas, f"Tur: #{self.loop_count}", (20, bl_y + 50), scale=0.6, thickness=2)
        _put_text_outlined(canvas, f"FPS: {self.fps_ema:.1f}   Hız: {self.speed:g}x", (20, bl_y + 72), scale=0.55)

        # Bottom-right stats — prefer violation engine counts if enabled
        br_x = w - 328
        br_y = h - 148
        _draw_translucent_box(canvas, br_x, br_y, 320, 100, alpha=0.55)
        _put_text_outlined(canvas, f"Aktif İşçi: {persons_in_frame}", (br_x + 12, br_y + 24), scale=0.6, thickness=2)
        total_dets = sum(self.detections_per_class.values())
        _put_text_outlined(canvas, f"Toplam Algılama: {total_dets}", (br_x + 12, br_y + 46), scale=0.55)
        if self.violation_detector is not None:
            active_count = len(self.violation_detector.get_active_violations())
            total_count = active_count + len(self.violation_detector.get_completed_violations())
        else:
            active_count = active_violations
            total_count = len(self.violation_track_ids)
        viol_color = COLOR_RED if active_count else COLOR_WHITE
        _put_text_outlined(canvas, f"Aktif İhlal: {active_count}", (br_x + 12, br_y + 70), scale=0.65, thickness=2, fg=viol_color)
        # Either "Toplam İhlal" alone, or with a "Yüklenen" suffix when Supabase is on.
        if self.supabase_enabled:
            line = f"Toplam İhlal: {total_count}   Yüklenen: {self.supabase_uploaded_count}"
        else:
            line = f"Toplam İhlal: {total_count}"
        _put_text_outlined(canvas, line, (br_x + 12, br_y + 92), scale=0.55)

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
                    active_v_count, persons = self._update_stats(detections)
                    # Violation processing (uses video time, not wall clock)
                    if self.enable_violations:
                        self._process_violations(detections, display_frame)
                    canvas = display_frame.copy()
                    self._draw_detections(canvas, detections)
                    self._draw_hud(canvas, active_v_count, persons)
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

        # Finalize any still-active violations and write the JSON report
        report_path: Optional[Path] = None
        if self.violation_detector is not None:
            still_active_ids = {ev.event_id for ev in self.violation_detector.get_active_violations()}
            self.violation_detector.finalize(self._video_time())
            # Push the resolution to Supabase for events that were active at exit.
            if self.supabase_enabled:
                for ev in self.violation_detector.get_completed_violations():
                    if ev.event_id in still_active_ids:
                        self._supabase_upload_resolution(ev)
            report_path = self._write_session_report(elapsed)

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
        if self.violation_detector is not None:
            all_v = self.violation_detector.all_violations()
            print(f"Toplam ihlal sayısı:      {len(all_v)}")
            by_type = defaultdict(int)
            for ev in all_v:
                by_type[ev.violation_type] += 1
            for vt in ("no_hardhat", "no_vest", "no_mask"):
                print(f"  {vt:<14} {by_type[vt]}")
            if report_path:
                print(f"Rapor / report:           {report_path}")
        else:
            print(f"Benzersiz ihlal track ID: {len(self.violation_track_ids)}")
        print()
        print("Sınıf bazında algılama / detections per class:")
        if self.detections_per_class:
            for k, v in sorted(self.detections_per_class.items(), key=lambda kv: -kv[1]):
                print(f"  {k:<20} {v}")
        else:
            print("  (yok / none)")
        print("=" * 64)

    def _write_session_report(self, elapsed_s: float) -> Optional[Path]:
        if self.violation_detector is None:
            return None
        all_violations = self.violation_detector.all_violations()
        by_type: dict[str, int] = defaultdict(int)
        durations: list[float] = []
        unique_violators: set[int] = set()
        for ev in all_violations:
            by_type[ev.violation_type] += 1
            durations.append(ev.duration_seconds)
            unique_violators.add(ev.worker_id)
        avg_dur = float(sum(durations) / len(durations)) if durations else 0.0

        report = {
            "session_id": self.session_id,
            "started_at": self._session_started_at.isoformat(timespec="milliseconds"),
            "ended_at": datetime.now().isoformat(timespec="milliseconds"),
            "total_duration_seconds": round(elapsed_s, 3),
            "camera_id": self.camera_id,
            "video_source": (
                f"webcam:{int(self.video_path)}"
                if self.is_webcam
                else str(self.video_path)
            ),
            "detector_type": self.detector_type,
            "loop_count": self.loop_count,
            "frames_processed": self.total_frames_processed,
            "total_violations": len(all_violations),
            "violations_by_type": {
                "no_hardhat": by_type.get("no_hardhat", 0),
                "no_vest": by_type.get("no_vest", 0),
                "no_mask": by_type.get("no_mask", 0),
            },
            "unique_workers_violated": len(unique_violators),
            "average_violation_duration": round(avg_dur, 3),
            "violations": [ev.to_dict() for ev in all_violations],
        }
        out_dir = PROJECT_ROOT / "data" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.session_id}_violations.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path


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
                        "backend/models/rtdetr_isgozcu_ppe_v2 for rtdetr (--model-version v1 to roll back).")
    p.add_argument("--model-version", choices=["v1", "v2"], default="v2",
                   help="Which RT-DETR checkpoint to load when --detector rtdetr "
                        "and no --model-path is given (default: v2).")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--headless", action="store_true", help="No GUI; for automated testing.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=None)
    p.add_argument("--enable-violations", dest="enable_violations", action="store_true", default=True,
                   help="Enable the violation engine (default).")
    p.add_argument("--disable-violations", dest="enable_violations", action="store_false",
                   help="Turn the violation engine off.")
    p.add_argument("--min-duration", type=float, default=5.0,
                   help="Min violation duration in video seconds before firing (default 5.0).")
    p.add_argument("--violation-threshold", type=float, default=0.7,
                   help="Non-compliance ratio that triggers a violation (default 0.7).")
    p.add_argument("--recovery-threshold", type=float, default=0.3,
                   help="Ratio below which an active violation ends. Must be < violation-threshold (default 0.3).")
    p.add_argument("--supabase", action="store_true",
                   help="Upload violation events + snapshots to Supabase (requires .env). Off by default.")
    p.add_argument("--site-name", default="Demo Şantiye",
                   help="Site name used when --supabase is on (default 'Demo Şantiye').")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_env(PROJECT_ROOT)
    args = _parse_args(argv)

    # `--input N` (digit string) is a webcam index; everything else is a path.
    raw_input = args.input.strip() if isinstance(args.input, str) else args.input
    if isinstance(raw_input, str) and raw_input.isdigit():
        in_source: Path | int = int(raw_input)
        print(f"[bilgi] Webcam modu: device index {in_source}")
    else:
        in_path = Path(raw_input)
        if not in_path.is_absolute():
            in_path = PROJECT_ROOT / in_path
        if not in_path.exists():
            print(f"[HATA] Video yok: {in_path}")
            return 1
        in_source = in_path

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
    else:
        if args.detector == "yolo":
            model_path = PROJECT_ROOT / "backend" / "models" / "ppe_yolov8n.pt"
        else:
            model_path = PROJECT_ROOT / "backend" / "models" / f"rtdetr_isgozcu_ppe_{args.model_version}"
    if not model_path.exists():
        print(f"[HATA] Model yok: {model_path}")
        return 1

    sim = LiveStreamSimulator(
        video_path=in_source,
        camera_id=args.camera_id,
        conf=args.conf,
        speed=args.speed,
        loop_enabled=not args.no_loop,
        detector_type=args.detector,
        model_path=model_path,
        headless=args.headless,
        max_frames=args.max_frames,
        snapshot_every=args.snapshot_every,
        enable_violations=args.enable_violations,
        min_violation_duration=args.min_duration,
        violation_threshold=args.violation_threshold,
        recovery_threshold=args.recovery_threshold,
        enable_supabase=args.supabase,
        site_name=args.site_name,
    )
    return sim.run()


if __name__ == "__main__":
    sys.exit(main())
