"""Live test app: webcam → virtual PPE filter → YOLO detection → side-by-side display.

Keyboard controls:
    H  toggle virtual hardhat
    V  toggle virtual vest
    S  save snapshot to data/snapshots/
    Q  / ESC  quit
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.virtual_ppe_filter import VirtualPPEFilter  # noqa: E402


WINDOW_TITLE = "İşGözcü Live Test"
PANEL_W, PANEL_H = 800, 600  # each side of the side-by-side display

# Class -> category for box coloring (BGR)
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Mask"}
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
PERSON_CLASS = "Person"
COLOR_COMPLIANT = (60, 200, 60)     # green
COLOR_VIOLATION = (60, 60, 220)     # red
COLOR_PERSON = (220, 130, 50)       # blue
COLOR_OTHER = (200, 200, 200)       # grey


def _category_color(cls_name: str) -> tuple[int, int, int]:
    if cls_name in COMPLIANT_CLASSES:
        return COLOR_COMPLIANT
    if cls_name in VIOLATION_CLASSES:
        return COLOR_VIOLATION
    if cls_name == PERSON_CLASS:
        return COLOR_PERSON
    return COLOR_OTHER


def _put_text_outlined(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.6,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def _draw_detections(frame: np.ndarray, result, class_names: dict[int, str]) -> dict[str, int]:
    """Draw YOLO boxes onto `frame` in place. Returns per-class detection count."""
    counts: dict[str, int] = defaultdict(int)
    if result.boxes is None or len(result.boxes) == 0:
        return counts

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    cls_idx = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    ids = (
        result.boxes.id.cpu().numpy().astype(int)
        if result.boxes.id is not None
        else [None] * len(boxes)
    )

    for (x1, y1, x2, y2), c, conf, tid in zip(boxes, cls_idx, confs, ids):
        cls_name = class_names.get(int(c), str(c))
        counts[cls_name] += 1
        color = _category_color(cls_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label_parts = [cls_name, f"{conf:.2f}"]
        if tid is not None:
            label_parts.append(f"id{int(tid)}")
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return counts


def _compliance_summary(counts: dict[str, int]) -> str:
    person = counts.get(PERSON_CLASS, 0)
    hardhat = counts.get("Hardhat", 0)
    vest = counts.get("Safety Vest", 0)
    no_hardhat = counts.get("NO-Hardhat", 0)
    no_vest = counts.get("NO-Safety Vest", 0)
    if no_hardhat or no_vest:
        status = "İhlal / VIOLATION"
    elif person and (hardhat or vest):
        status = "Uyumlu / OK"
    elif person:
        status = "Belirsiz / unknown"
    else:
        status = "Kişi yok / no person"
    return f"P:{person} H:{hardhat} V:{vest} NH:{no_hardhat} NV:{no_vest} | {status}"


def _resize_panel(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _compose_display(left: np.ndarray, right: np.ndarray, hud_state: dict) -> np.ndarray:
    """Stack two panels horizontally and overlay HUD text."""
    left_r = _resize_panel(left, PANEL_W, PANEL_H)
    right_r = _resize_panel(right, PANEL_W, PANEL_H)
    canvas = np.concatenate([left_r, right_r], axis=1)
    h, w = canvas.shape[:2]

    # Top-left: title
    _put_text_outlined(canvas, "İşGözcü - Live Test Mode", (12, 28), scale=0.75, thickness=2)
    # Top-right: timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    _put_text_outlined(canvas, ts, (w - tw - 12, 28), scale=0.6)

    # Centered FPS readout
    fps_txt = f"FPS: {hud_state.get('fps', 0.0):.1f}"
    _put_text_outlined(canvas, fps_txt, (w // 2 - 50, 28), scale=0.7, thickness=2)

    # Filter state strip (just under the title)
    filt_txt = "Filtre / Filter:  " + (
        f"Hardhat={'ON' if hud_state.get('hardhat') else 'OFF'}   "
        f"Vest={'ON' if hud_state.get('vest') else 'OFF'}"
    )
    _put_text_outlined(canvas, filt_txt, (12, 56), scale=0.55)

    # Panel labels
    _put_text_outlined(canvas, "Filtreli giriş / Filtered input", (12, h - 50), scale=0.55)
    _put_text_outlined(canvas, "YOLO tespiti / YOLO detections", (PANEL_W + 12, h - 50), scale=0.55)

    # Bottom-left: controls hint
    controls = "H: Hardhat ON/OFF | V: Vest ON/OFF | S: Snapshot | Q: Quit"
    _put_text_outlined(canvas, controls, (12, h - 16), scale=0.55)

    # Bottom-right: detection summary
    summary = hud_state.get("summary", "")
    (tw, _), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _put_text_outlined(canvas, summary, (w - tw - 12, h - 16), scale=0.55)
    return canvas


class LiveTestApp:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        conf: float = 0.4,
        imgsz: int = 640,
        half: bool = True,
        cam_index: int = 0,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA mevcut değil ama device='cuda' istendi.")
        print(f"[LiveTestApp] Model yükleniyor: {model_path}")
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = half
        self.cam_index = cam_index
        self.filter = VirtualPPEFilter(model_complexity=1)

        self.hardhat_enabled = True
        self.vest_enabled = False

    def run(self) -> int:
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[HATA] Webcam açılamadı / could not open webcam.")
            return 1

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[LiveTestApp] Webcam {width}x{height}. Pencere açılıyor...")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, PANEL_W * 2, PANEL_H)

        snapshot_dir = PROJECT_ROOT / "data" / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        frames = 0
        loop_start = time.perf_counter()
        ema_fps = 0.0
        all_counts: dict[str, int] = defaultdict(int)
        last_t = loop_start

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[uyarı] Kare okunamadı, çıkılıyor.")
                    break

                # Apply virtual filter
                filtered = self.filter.process(
                    frame.copy(),
                    hardhat=self.hardhat_enabled,
                    vest=self.vest_enabled,
                )

                # Run YOLO + tracking
                results = self.model.track(
                    source=filtered,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf,
                    imgsz=self.imgsz,
                    half=self.half,
                    device=self.device,
                    verbose=False,
                )
                r = results[0]
                annotated = filtered.copy()
                counts = _draw_detections(annotated, r, self.class_names)
                for k, v in counts.items():
                    all_counts[k] += v

                # FPS (EMA over last frames)
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                ema_fps = inst_fps if frames == 0 else (ema_fps * 0.9 + inst_fps * 0.1)

                hud = {
                    "fps": ema_fps,
                    "hardhat": self.hardhat_enabled,
                    "vest": self.vest_enabled,
                    "summary": _compliance_summary(counts),
                }
                canvas = _compose_display(filtered, annotated, hud)
                cv2.imshow(WINDOW_TITLE, canvas)
                frames += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # ESC
                    print("[LiveTestApp] Çıkış / quit.")
                    break
                elif key in (ord("h"), ord("H")):
                    self.hardhat_enabled = not self.hardhat_enabled
                    print(f"[LiveTestApp] Hardhat = {self.hardhat_enabled}")
                elif key in (ord("v"), ord("V")):
                    self.vest_enabled = not self.vest_enabled
                    print(f"[LiveTestApp] Vest = {self.vest_enabled}")
                elif key in (ord("s"), ord("S")):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    out_path = snapshot_dir / f"live_test_{ts}.jpg"
                    cv2.imwrite(str(out_path), canvas)
                    print(f"[LiveTestApp] Snapshot kaydedildi / saved: {out_path}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.filter.close()

        elapsed = time.perf_counter() - loop_start
        avg_fps = frames / elapsed if elapsed > 0 else 0.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        print()
        print("=" * 60)
        print("ÖZET / SUMMARY")
        print("=" * 60)
        print(f"Toplam kare / frames: {frames}")
        print(f"Süre / elapsed:       {elapsed:.2f} s")
        print(f"Ortalama FPS:         {avg_fps:.2f}")
        print(f"Tepe VRAM:            {peak_vram_mb:.2f} MB")
        print("Sınıf başına tespit / detections per class:")
        if all_counts:
            for k, v in sorted(all_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {k:<20} {v}")
        else:
            print("  (yok / none)")
        print("=" * 60)
        return 0


def _ensure_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    example_path = project_root / ".env.example"
    if not env_path.exists() and example_path.exists():
        env_path.write_bytes(example_path.read_bytes())
    load_dotenv(env_path)


def main() -> int:
    _ensure_env(PROJECT_ROOT)
    model_path = Path(os.getenv("YOLO_MODEL_PATH", "backend/models/ppe_yolov8n.pt"))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    conf = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    if not model_path.exists():
        print(f"[HATA] Model yok / model missing: {model_path}")
        return 1
    app = LiveTestApp(model_path=model_path, conf=conf)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
