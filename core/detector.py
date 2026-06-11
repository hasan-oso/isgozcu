"""core/detector.py - clean v3 inference + ByteTrack + frame-skipping.

Wraps the production v3 RT-DETR checkpoint (model/weights/v3/) and tracks
people across frames with supervision's ByteTrack. Only the 3 whitelisted
classes (Person / Hardhat / Safety Vest) are ever returned.

Frame-skipping (key for live performance): a full v3 detection runs only every
`detect_every_n` frames; on the in-between frames the last tracked result is
reused (positions held), lifting effective FPS ~detect_every_n x. No new
inference happens on skipped frames.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from core.labels_tr import WHITELIST

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "weights" / "v3"
PERSON_CLASS = "Person"


class Detection(TypedDict):
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (pixels)
    class_name: str                    # v3 name, always within WHITELIST
    confidence: float
    track_id: Optional[int]            # set for Person, None for PPE


class DetectorAdapter:
    """v3 detector + ByteTrack person tracking with frame-skipping."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        conf_threshold: float = 0.25,
        detect_every_n: int = 3,
        device: Optional[str] = None,
        use_fp16: bool = True,
        lost_track_buffer: int = 300,
        inference_max_size: int = 512,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"v3 model not found: {self.model_dir}")
        self.conf_threshold = float(conf_threshold)
        self.detect_every_n = max(1, int(detect_every_n))
        self.inference_max_size = int(inference_max_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16) and self.device == "cuda"

        self.processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
        # Perf lever: run the model at a smaller SQUARE resolution than v3's
        # default 640x640. RT-DETR resizes inputs to processor.size, so this cuts
        # inference compute by ~(S/640)^2. Boxes remain correct because _infer
        # post-processes with target_sizes=(orig_h, orig_w): the model emits
        # normalised box coords that map back to full-frame pixels regardless of
        # the resize. The v3 processor uses square resize (do_pad=false), so a
        # smaller square keeps the SAME preprocessing the model was trained on --
        # no aspect-ratio or box drift, just lower resolution. Drawing later
        # happens on the ORIGINAL full-res frame (the detector only resizes its
        # own model input, never the returned boxes or the caller's frame).
        try:
            s = self.inference_max_size
            self.processor.size = {"height": s, "width": s}
        except Exception:
            pass
        self.model = (
            AutoModelForObjectDetection.from_pretrained(str(self.model_dir))
            .to(self.device)
            .eval()
        )
        if self.use_fp16:
            self.model = self.model.half()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        # ByteTrack tracks PEOPLE only (anonymous worker ids). lost_track_buffer
        # keeps the same id across short occlusions / loop boundaries.
        self.tracker = sv.ByteTrack(
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=2,
        )

        # frame-skip + measurement state
        self._frame_index = 0
        self._last_result: list[Detection] = []
        self._infer_count = 0
        self._infer_time_sum = 0.0
        self._first_ts: Optional[float] = None
        self._frames_seen = 0

    # ---------- core inference ----------
    def _infer(self, frame_bgr: np.ndarray) -> list[dict]:
        """Run v3 once; return whitelisted raw detections (no track_id yet)."""
        h, w = frame_bgr.shape[:2]
        image = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.use_fp16:
            inputs = {
                k: (v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v)
                for k, v in inputs.items()
            }
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self._infer_time_sum += time.perf_counter() - t0
        self._infer_count += 1

        target = torch.tensor([[h, w]], device=self.device)
        res = self.processor.post_process_object_detection(
            outputs, target_sizes=target, threshold=self.conf_threshold
        )[0]
        raw: list[dict] = []
        for score, label, box in zip(
            res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()
        ):
            name = self.id2label.get(int(label), "?")
            if name not in WHITELIST:
                continue  # drop the 7 non-target classes here
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            raw.append({"bbox": (x1, y1, x2, y2), "class_name": name, "confidence": float(score)})
        return raw

    def _track_persons(self, raw: list[dict]) -> list[Detection]:
        """Assign track_id to Person detections via ByteTrack; PPE passes through."""
        persons = [d for d in raw if d["class_name"] == PERSON_CLASS]
        others = [d for d in raw if d["class_name"] != PERSON_CLASS]

        if persons:
            dets = sv.Detections(
                xyxy=np.array([d["bbox"] for d in persons], dtype=np.float32),
                confidence=np.array([d["confidence"] for d in persons], dtype=np.float32),
                class_id=np.zeros(len(persons), dtype=int),
            )
        else:
            dets = sv.Detections.empty()
        tracked = self.tracker.update_with_detections(dets)

        out: list[Detection] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (int(v) for v in tracked.xyxy[i])
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            out.append({"bbox": (x1, y1, x2, y2), "class_name": PERSON_CLASS,
                        "confidence": conf, "track_id": tid})
        for d in others:
            out.append({"bbox": d["bbox"], "class_name": d["class_name"],
                        "confidence": d["confidence"], "track_id": None})
        return out

    # ---------- public ----------
    def detect_and_track(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect + track with frame-skipping. Returns whitelisted Detections."""
        if self._first_ts is None:
            self._first_ts = time.perf_counter()
        self._frames_seen += 1
        run_detect = (self._frame_index % self.detect_every_n) == 0
        self._frame_index += 1
        if run_detect:
            self._last_result = self._track_persons(self._infer(frame_bgr))
        # on skipped frames reuse the last tracked result (positions held)
        return list(self._last_result)

    def reset(self) -> None:
        self.tracker.reset()
        self._frame_index = 0
        self._last_result = []

    def stats(self) -> dict:
        wall = (time.perf_counter() - self._first_ts) if self._first_ts else 0.0
        avg_ms = (self._infer_time_sum / self._infer_count * 1000.0) if self._infer_count else 0.0
        eff_fps = (self._frames_seen / wall) if wall > 0 else 0.0
        return {
            "frames_seen": self._frames_seen,
            "inferences": self._infer_count,
            "avg_inference_ms": round(avg_ms, 2),
            "effective_fps": round(eff_fps, 2),
            "detect_every_n": self.detect_every_n,
        }


__all__ = ["DetectorAdapter", "Detection"]
