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


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3", "v4")


def _path_for_version(version: str) -> Path:
    """Map 'v1'..'v4' to the on-disk checkpoint directory."""
    return _PROJECT_ROOT / "backend" / "models" / f"rtdetr_isgozcu_ppe_{version}"


class RTDetrDetector:
    """RT-DETR based PPE detector with built-in ByteTrack tracking via supervision."""

    def __init__(
        self,
        # Phase 2A.5.B: opting `model_path` into Optional and adding the
        # convenience `model_version` arg. If both are supplied, model_path
        # wins (legacy callers stay safe). If neither, we resolve the default
        # version below.
        #
        # Default version stays "v3" today — v4 is opt-in via
        # `RTDetrDetector(model_version="v4")` until we A/B-validate it on
        # webcam. v4 uses a brand-new 6-class schema (Person, helmet, vest,
        # gloves, no_helmet, no_gloves) trained from scratch on the
        # Ultralytics Construction-PPE dataset; v1/v2/v3 share the original
        # 10-class schema (Hardhat, Mask, NO-*, Person, Safety Cone, etc.).
        #
        # See docs/migration_yolo_to_rtdetr.md for the full lineage.
        model_path: str | Path | None = None,
        model_version: str = "v3",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        use_fp16: bool = True,
        # Phase 2B.1: bumped 150 → 300. Workers fragmented into multiple
        # track_ids when the looping CCTV video rewound; a longer buffer keeps
        # the same physical worker on the same ID across loop boundaries and
        # short occlusions. See docs/violation_engine.md.
        track_buffer: int = 300,
    ) -> None:
        # Resolve model path: explicit path > version shortcut > default ("v3").
        if model_path is None:
            if model_version not in SUPPORTED_VERSIONS:
                raise ValueError(
                    f"Unknown model_version '{model_version}'; "
                    f"expected one of {SUPPORTED_VERSIONS}."
                )
            model_path = _path_for_version(model_version)
        self.model_path = Path(model_path)
        # Best-effort version label for callers — used by classes_v4 routing,
        # not load semantics (load is driven by self.model.config).
        self.model_version = model_version
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

        # Settings mirror the YOLO-side bytetrack_custom.yaml so the two
        # backends behave consistently w.r.t. loop-boundary continuity.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=max(0.25, conf_threshold * 0.6),
            lost_track_buffer=track_buffer,  # 300 from default
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
