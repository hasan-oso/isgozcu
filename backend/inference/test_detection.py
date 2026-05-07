"""PPE detection on a video using YOLOv8 + ByteTrack.

Usage (from project root, with venv active):
    python backend/inference/test_detection.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PPEDetector:
    """YOLOv8 PPE detector with ByteTrack tracking and per-frame stats."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        conf: float = 0.4,
        imgsz: int = 640,
        half: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı / not found: {self.model_path}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA mevcut değil / CUDA unavailable, but device='cuda' requested.")

        print(f"[PPEDetector] Model yükleniyor: {self.model_path.name}")
        self.model = YOLO(str(self.model_path))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = half
        self.class_names: dict[int, str] = self.model.names
        print(f"[PPEDetector] Model yüklendi. Sınıflar / classes: {list(self.class_names.values())}")

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        tracker: str = "bytetrack.yaml",
    ) -> dict:
        """Run detection + tracking on a video and write an annotated copy.

        Returns a stats dict: total_frames, elapsed_s, avg_fps, detections_per_class,
        unique_track_ids, output_size_mb, peak_vram_mb.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Giriş videosu yok / input missing: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı / cannot open: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Çıktı videosu yazılamadı / cannot write: {output_path}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        detection_counts: dict[str, int] = defaultdict(int)
        unique_ids: set[int] = set()
        processed = 0
        start = time.perf_counter()

        try:
            with tqdm(total=total_frames, unit="frame", desc="Tespit / Detect") as pbar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    results = self.model.track(
                        source=frame,
                        persist=True,
                        tracker=tracker,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        half=self.half,
                        device=self.device,
                        verbose=False,
                    )
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        cls_idx = r.boxes.cls.cpu().numpy().astype(int)
                        for c in cls_idx:
                            detection_counts[self.class_names.get(int(c), str(c))] += 1
                        if r.boxes.id is not None:
                            for tid in r.boxes.id.cpu().numpy().astype(int):
                                unique_ids.add(int(tid))

                    annotated = r.plot()
                    writer.write(annotated)

                    processed += 1
                    pbar.update(1)
                    if processed % 30 == 0:
                        elapsed_so_far = time.perf_counter() - start
                        pbar.set_postfix(fps=f"{processed/elapsed_so_far:.2f}")
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - start
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total_frames": processed,
            "elapsed_s": elapsed,
            "avg_fps": processed / elapsed if elapsed > 0 else 0.0,
            "detections_per_class": dict(detection_counts),
            "unique_track_ids": len(unique_ids),
            "output_size_mb": output_path.stat().st_size / (1024 * 1024)
            if output_path.exists()
            else 0.0,
            "peak_vram_mb": peak_vram_mb,
        }


def _ensure_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    example_path = project_root / ".env.example"
    if not env_path.exists() and example_path.exists():
        env_path.write_bytes(example_path.read_bytes())
        print(f"[main] .env oluşturuldu / created from .env.example")
    load_dotenv(env_path)


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg


def main() -> int:
    _ensure_env(PROJECT_ROOT)

    model_path = Path(os.getenv("YOLO_MODEL_PATH", "backend/models/ppe_yolov8n.pt"))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    conf = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

    input_video = PROJECT_ROOT / "videos" / "samples" / "sample_01.mp4"
    output_video = PROJECT_ROOT / "videos" / "outputs" / "sample_01_detected.mp4"

    summary: Optional[dict] = None
    last_err: Optional[BaseException] = None

    for imgsz in (640, 416):
        try:
            detector = PPEDetector(
                model_path=model_path,
                device="cuda",
                conf=conf,
                imgsz=imgsz,
                half=True,
            )
            print(f"[main] Tespit başlatılıyor / starting (imgsz={imgsz}) ...")
            summary = detector.process_video(
                input_path=input_video,
                output_path=output_video,
                tracker="bytetrack.yaml",
            )
            summary["imgsz_used"] = imgsz
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
            if _is_oom(e):
                last_err = e
                torch.cuda.empty_cache()
                print(f"[UYARI] CUDA OOM at imgsz={imgsz}: {e}. Daha düşük imgsz ile deneniyor.")
                continue
            raise

    if summary is None:
        print(f"[HATA] Tespit başarısız / detection failed. Last error: {last_err}")
        return 1

    print()
    print("=" * 64)
    print("ÖZET / SUMMARY")
    print("=" * 64)
    print(f"Input:                {summary['input_path']}")
    print(f"Output:               {summary['output_path']}")
    print(f"imgsz used:           {summary['imgsz_used']}")
    print(f"Toplam kare / frames: {summary['total_frames']}")
    print(f"Süre / elapsed:       {summary['elapsed_s']:.2f} s")
    print(f"Ortalama FPS:         {summary['avg_fps']:.2f}")
    print(f"Benzersiz ID / IDs:   {summary['unique_track_ids']}")
    print(f"Çıktı boyutu / size:  {summary['output_size_mb']:.2f} MB")
    print(f"Tepe VRAM:            {summary['peak_vram_mb']:.2f} MB")
    print()
    print("Sınıf başına tespit / detections per class:")
    if summary["detections_per_class"]:
        for cls, count in sorted(
            summary["detections_per_class"].items(), key=lambda kv: -kv[1]
        ):
            print(f"  {cls:<20} {count}")
    else:
        print("  (yok / none)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
