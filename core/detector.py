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
DEFAULT_RFDETR_WEIGHTS = PROJECT_ROOT / "model" / "weights" / "rfdetr" / "rfdetr_ppe.pth"
PERSON_CLASS = "Person"
# RF-DETR-only Safety-Vest floor. RF-DETR detects REAL vests with high confidence
# (~0.9+), so a higher floor cuts low-confidence FALSE vests (on non-vest objects)
# without dropping real ones. Applied ONLY on the rfdetr path; v3 keeps its own
# 0.38 vest floor in postprocess (v3's real vests sit lower, near the cut-off).
RFDETR_VEST_FLOOR = 0.60


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


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
        lost_track_buffer: int = 600,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.85,
        minimum_consecutive_frames: int = 2,
        frame_rate: int = 30,
        inference_max_size: int = 512,
        tracker: str = "bytetrack",
        ocsort_det_thresh: float = 0.4,
        ocsort_max_age: int = 90,
        ocsort_min_hits: int = 2,
        ocsort_iou: float = 0.3,
        backend: str = "rtdetr_v3",
        rfdetr_weights: str | Path = DEFAULT_RFDETR_WEIGHTS,
        rfdetr_vest_floor: float = RFDETR_VEST_FLOOR,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.conf_threshold = float(conf_threshold)
        self.detect_every_n = max(1, int(detect_every_n))
        self.lost_track_buffer = int(lost_track_buffer)
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self.frame_rate = int(frame_rate)
        self.inference_max_size = int(inference_max_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16) and self.device == "cuda"
        self.rfdetr_vest_floor = float(rfdetr_vest_floor)

        # Selectable detection backend. "rtdetr_v3" (transformers, the original) is
        # the DEFAULT and stays available as a fallback; "rfdetr" loads the trained
        # RF-DETR-Nano checkpoint via the rfdetr library. Both emit the same
        # StandardDetection (bbox/class_name/confidence), so postprocess/violations
        # /tracking are unchanged. RF-DETR is NMS-free; the person de-dup already
        # uses the safe 0.90 IoU (postprocess.DEDUP_IOU_PERSON) so it never merges
        # distinct people for either backend -- no backend-specific change needed.
        self.backend = str(backend).lower()
        self.rfdetr = None
        if self.backend == "rfdetr":
            self._load_rfdetr(Path(rfdetr_weights))
        else:
            self.backend = "rtdetr_v3"
            self._load_v3()

        # ByteTrack tracks PEOPLE only (anonymous worker ids). Tuned for occlusion
        # robustness (workers passing behind one another in a crowded site):
        #  - lost_track_buffer=600: a lost track is retained for up to 600 *update*
        #    frames (= int(frame_rate/30 * buffer)). With detect_every_n>1 the
        #    tracker only steps on detection frames, so 600 spans ~25 s @ n=1/24fps
        #    and longer at higher n -- the same worker id survives a long occlusion
        #    instead of being re-numbered on re-appearance. (Measured: an id
        #    survives an occlusion iff max_time_lost >= the gap length.)
        #  - minimum_matching_threshold=0.85: ByteTrack matches when the fused cost
        #    1 - IoU*score <= threshold, so a HIGHER value is MORE tolerant during
        #    re-association (accepts a lower-overlap re-match after occlusion). Kept
        #    a mild step above the 0.80 default; raise further only if re-acquisition
        #    still fails, lower toward 0.80 if ids get *stolen* in dense crowds.
        #  - minimum_consecutive_frames=2: a NEW track needs 2 detections to confirm
        #    (suppresses spurious ids); a re-found lost track re-activates instantly.
        #  - frame_rate only scales max_time_lost; left at 30 so buffer == frames.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        # Selectable tracker. ByteTrack (above) is ALWAYS kept as the fallback and
        # stays the default; OC-SORT is an opt-in alternative for crowded same-
        # uniform scenes (appearance-free, motion-only; vendored in core/ocsort/,
        # depends only on numpy/scipy -- no filterpy/pip install). Both expose the
        # same StandardDetection output, so postprocess/violations are unaffected.
        self.tracker_kind = str(tracker).lower()
        self.ocsort = None
        if self.tracker_kind == "ocsort":
            from core.ocsort import OCSort
            self._ocsort_kwargs = dict(
                det_thresh=float(ocsort_det_thresh), max_age=int(ocsort_max_age),
                min_hits=int(ocsort_min_hits), iou_threshold=float(ocsort_iou),
                delta_t=3, asso_func="iou", inertia=0.2, use_byte=False,
            )
            self.ocsort = OCSort(**self._ocsort_kwargs)
        self._last_hw = (0, 0)

        # frame-skip + measurement state
        self._frame_index = 0
        self._last_result: list[Detection] = []
        self._infer_count = 0
        self._infer_time_sum = 0.0
        self._first_ts: Optional[float] = None
        self._frames_seen = 0

    # ---------- backend loaders ----------
    def _load_v3(self) -> None:
        """Load the v3 RT-DETR checkpoint via transformers (the default backend)."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"v3 model not found: {self.model_dir}")
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
        # Perf lever: run the model at a smaller SQUARE resolution than v3's default
        # 640. RT-DETR resizes inputs to processor.size; _infer post-processes with
        # target_sizes=(orig_h, orig_w) so boxes map back to full-frame pixels.
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

    def _load_rfdetr(self, weights: Path) -> None:
        """Load the trained RF-DETR-Nano checkpoint via the rfdetr library."""
        if not weights.exists():
            raise FileNotFoundError(f"RF-DETR weights not found: {weights}")
        from rfdetr import RFDETRNano   # lazy import: only when backend == 'rfdetr'
        self.rfdetr = RFDETRNano(pretrain_weights=str(weights))
        # the checkpoint embeds the class order; fall back to the known 3-class map.
        self.rfdetr_class_names = list(getattr(self.rfdetr, "class_names",
                                               ["Person", "Hardhat", "Safety Vest"]))

    # ---------- core inference ----------
    def _infer(self, frame_bgr: np.ndarray) -> list[dict]:
        """Run the detector once; return whitelisted raw detections (no track_id yet)."""
        h, w = frame_bgr.shape[:2]
        self._last_hw = (h, w)   # OC-SORT needs frame size (img_info==img_size -> scale 1)
        if self.backend == "rfdetr":
            return self._infer_rfdetr(frame_bgr)
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

    def _infer_rfdetr(self, frame_bgr: np.ndarray) -> list[dict]:
        """RF-DETR-Nano inference -> whitelisted raw detections (same shape as _infer).
        RF-DETR is NMS-free and returns a supervision Detections (xyxy/conf/class_id);
        class_id is 0-indexed in the trained order (0=Person, 1=Hardhat, 2=Safety Vest)."""
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])   # contiguous RGB (rfdetr rejects negative strides)
        t0 = time.perf_counter()
        dets = self.rfdetr.predict(rgb, threshold=self.conf_threshold)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self._infer_time_sum += time.perf_counter() - t0
        self._infer_count += 1
        raw: list[dict] = []
        names = self.rfdetr_class_names
        for i in range(len(dets)):
            cid = int(dets.class_id[i])
            name = names[cid] if 0 <= cid < len(names) else "?"
            if name not in WHITELIST:
                continue
            conf = float(dets.confidence[i])
            # rfdetr-only Safety-Vest floor: cut low-confidence FALSE vests; RF-DETR's
            # real vests are high-confidence so this keeps them.
            if name == "Safety Vest" and conf < self.rfdetr_vest_floor:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in dets.xyxy[i])
            raw.append({"bbox": (x1, y1, x2, y2), "class_name": name, "confidence": conf})
        return raw

    def _track_persons(self, raw: list[dict]) -> list[Detection]:
        """Assign track_id to Person detections (ByteTrack or OC-SORT); PPE passes through."""
        persons = [d for d in raw if d["class_name"] == PERSON_CLASS]
        others = [d for d in raw if d["class_name"] != PERSON_CLASS]
        if self.tracker_kind == "ocsort":
            out = self._track_persons_ocsort(persons)
        else:
            out = self._track_persons_bytetrack(persons)
        for d in others:
            out.append({"bbox": d["bbox"], "class_name": d["class_name"],
                        "confidence": d["confidence"], "track_id": None})
        return out

    def _track_persons_bytetrack(self, persons: list[dict]) -> list[Detection]:
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
        return out

    def _track_persons_ocsort(self, persons: list[dict]) -> list[Detection]:
        h, w = self._last_hw if self._last_hw != (0, 0) else (1, 1)
        if persons:
            arr = np.array([[*d["bbox"], d["confidence"]] for d in persons], dtype=np.float32)
        else:
            arr = np.empty((0, 5), dtype=np.float32)
        # img_info == img_size -> internal scale = 1, boxes stay in pixel coords.
        online = self.ocsort.update(arr, (h, w), (h, w))   # rows: [x1, y1, x2, y2, track_id]
        out: list[Detection] = []
        for row in online:
            box = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            tid = int(row[4])
            # OC-SORT returns KF-smoothed boxes without scores -> recover the
            # detection confidence from the best-IoU input; a KF-only (occluded,
            # un-matched) track gets a floor conf so postprocess keeps it.
            conf, best = 0.6, 0.0
            for d in persons:
                v = _iou_xyxy(d["bbox"], box)
                if v > best:
                    best, conf = v, d["confidence"]
            out.append({"bbox": box, "class_name": PERSON_CLASS,
                        "confidence": conf if best > 0.3 else 0.6, "track_id": tid})
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
        if self.tracker_kind == "ocsort" and self.ocsort is not None:
            from core.ocsort import OCSort
            self.ocsort = OCSort(**self._ocsort_kwargs)   # OCSort has no reset(); rebuild
        else:
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
            "backend": self.backend,
            "tracker": self.tracker_kind,
            "lost_track_buffer": self.lost_track_buffer,
            "minimum_matching_threshold": self.minimum_matching_threshold,
        }


__all__ = ["DetectorAdapter", "Detection"]
