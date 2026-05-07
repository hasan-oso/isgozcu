"""Unified detector adapter for YOLOv8 (legacy) and RT-DETR (current).

Both detectors expose a single common method:

    DetectorAdapter.detect_and_track(frame_bgr) -> list[dict]

Each dict has the keys: bbox (x1, y1, x2, y2), class_id, class_name, confidence,
track_id (Optional[int]). This lets downstream consumers (the live simulator,
violation engine, etc.) stay detector-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, TypedDict


class StandardDetection(TypedDict):
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int]


DetectorType = Literal["yolo", "rtdetr"]


class DetectorAdapter:
    """Unified interface for YOLOv8 (ultralytics) and RT-DETR (transformers + supervision)."""

    def __init__(
        self,
        detector_type: DetectorType,
        model_path: str | Path | None = None,
        conf: float = 0.4,
        imgsz: int = 640,
    ) -> None:
        self.detector_type = detector_type
        self.conf = conf
        self.imgsz = imgsz

        if detector_type == "yolo":
            from ultralytics import YOLO

            default_path = Path("backend/models/ppe_yolov8n.pt")
            self._yolo_path = Path(model_path) if model_path else default_path
            print(f"[DetectorAdapter] YOLO yükleniyor: {self._yolo_path.name}")
            self.model = YOLO(str(self._yolo_path))
            self._class_names_map: dict[int, str] = self.model.names

            # Use custom bytetrack config if it exists, otherwise default.
            cfg = Path("backend/inference/configs/bytetrack_custom.yaml")
            self._tracker_yaml = str(cfg) if cfg.exists() else "bytetrack.yaml"

        elif detector_type == "rtdetr":
            from backend.inference.rtdetr_detector import RTDetrDetector

            default_path = Path("backend/models/rtdetr_isgozcu_ppe_v1")
            self._rtdetr_path = Path(model_path) if model_path else default_path
            print(f"[DetectorAdapter] RT-DETR yükleniyor: {self._rtdetr_path.name}")
            self.model = RTDetrDetector(
                model_path=self._rtdetr_path,
                conf_threshold=conf,
            )
            self._class_names_map = {i: name for i, name in enumerate(self.model.class_names)}

        else:
            raise ValueError(f"Bilinmeyen detector_type / unknown: {detector_type}")

    # ---------- public API ----------

    def detect_and_track(self, frame_bgr) -> list[StandardDetection]:
        if self.detector_type == "yolo":
            return self._yolo_detect_and_track(frame_bgr)
        return self._rtdetr_detect_and_track(frame_bgr)

    def reset_tracker(self) -> None:
        if self.detector_type == "rtdetr":
            self.model.reset_tracker()
        else:
            # Recreate tracker by calling track() with persist=False on next call.
            # ultralytics rebuilds internal trackers when persist toggles, so we
            # simulate a clean state by setting a flag the next call honors.
            self._yolo_reset_pending = True

    def get_class_names(self) -> list[str]:
        if self.detector_type == "yolo":
            return [self._class_names_map[i] for i in sorted(self._class_names_map)]
        return self.model.class_names

    # ---------- yolo path ----------

    _yolo_reset_pending = False

    def _yolo_detect_and_track(self, frame_bgr) -> list[StandardDetection]:
        persist = not self._yolo_reset_pending
        self._yolo_reset_pending = False
        results = self.model.track(
            source=frame_bgr,
            persist=persist,
            tracker=self._tracker_yaml,
            conf=self.conf,
            imgsz=self.imgsz,
            half=True,
            device="cuda",
            verbose=False,
        )
        r = results[0]
        out: list[StandardDetection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        cls_idx = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        ids = (
            r.boxes.id.cpu().numpy().astype(int)
            if r.boxes.id is not None
            else [None] * len(xyxy)
        )
        for (x1, y1, x2, y2), c, conf, tid in zip(xyxy, cls_idx, confs, ids):
            name = self._class_names_map.get(int(c), str(c))
            out.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": int(c),
                    "class_name": name,
                    "confidence": float(conf),
                    "track_id": int(tid) if tid is not None else None,
                }
            )
        return out

    # ---------- rtdetr path ----------

    def _rtdetr_detect_and_track(self, frame_bgr) -> list[StandardDetection]:
        d = self.model.detect_and_track(frame_bgr)
        out: list[StandardDetection] = []
        if len(d) == 0:
            return out
        for i in range(len(d)):
            x1, y1, x2, y2 = d.xyxy[i].astype(int).tolist()
            cid = int(d.class_id[i])
            tid = int(d.tracker_id[i]) if d.tracker_id is not None else None
            out.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cid,
                    "class_name": self.model.get_class_name(cid),
                    "confidence": float(d.confidence[i]),
                    "track_id": tid,
                }
            )
        return out
