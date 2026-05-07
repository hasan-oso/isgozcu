"""RT-DETR PPE detector with ByteTrack tracking via supervision.

Wraps a HuggingFace RT-DETR model + RTDetrImageProcessor and exposes a
detect-and-track API that returns supervision Detections (which carry
xyxy boxes, class ids, confidences, and per-frame tracker ids).

Apache-2.0 friendly stack:
    - transformers (Apache 2.0)
    - supervision   (MIT)
    - PyTorch       (BSD-style)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class RTDetrDetector:
    """RT-DETR based PPE detector with built-in ByteTrack tracking via supervision."""

    def __init__(
        self,
        model_path: str | Path = "backend/models/rtdetr_isgozcu_ppe_v1",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        use_fp16: bool = True,
        track_buffer: int = 150,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RT-DETR model dizini yok / not found: {self.model_path}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA mevcut değil ama device='cuda' istendi.")
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_fp16 = use_fp16 and device == "cuda"

        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_path).to(self.device).eval()
        if self.use_fp16:
            self.model = self.model.half()

        # config.json may store keys as strings; normalize to int -> str
        raw = self.model.config.id2label or {}
        self.id2label: dict[int, str] = {int(k): v for k, v in raw.items()}
        self.class_names: list[str] = [self.id2label[i] for i in sorted(self.id2label)]

        self.tracker = sv.ByteTrack(
            track_activation_threshold=max(0.25, conf_threshold * 0.6),
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.85,
            frame_rate=30,
            minimum_consecutive_frames=2,
        )

    # ---------- inference ----------

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Run detection on a BGR frame (cv2 convention). Returns sv.Detections (no tracker_id)."""
        # cv2 BGR -> RGB -> PIL
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.use_fp16:
            inputs = {
                k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold,
        )[0]

        if len(results["boxes"]) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=results["boxes"].detach().cpu().numpy().astype(np.float32),
            confidence=results["scores"].detach().cpu().numpy().astype(np.float32),
            class_id=results["labels"].detach().cpu().numpy().astype(int),
        )

    def detect_and_track(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Detection + ByteTrack tracking. Returns Detections with tracker_id populated.

        Note: when there are no detections, supervision still calls the tracker
        with an empty Detections so internal state advances correctly.
        """
        detections = self.detect(frame_bgr)
        return self.tracker.update_with_detections(detections)

    def reset_tracker(self) -> None:
        """Reset tracker state (e.g. on video loop boundary)."""
        self.tracker.reset()

    # ---------- helpers ----------

    def get_class_name(self, class_id: int) -> str:
        return self.id2label.get(int(class_id), f"unknown_{class_id}")
