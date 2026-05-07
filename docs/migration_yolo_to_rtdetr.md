# Migration: YOLOv8 → RT-DETR

## Why we migrated

Two reasons, in order of importance:

1. **License**. YOLOv8 (Ultralytics) is licensed **AGPL-3.0**. AGPL requires that any software using YOLOv8 — *including network-deployed services* — be released under AGPL when distributed or made available over a network. For İşGözcü, which is intended for commercial deployment to Turkish construction-site customers, AGPL is incompatible with our business model.

   RT-DETR (the model architecture) is published by Baidu under **Apache 2.0**, and the HuggingFace `transformers` implementation we use is also Apache 2.0. The supervision tracking library is **MIT**. The Construction Site Safety dataset we trained on is **CC BY 4.0**. The full inference stack is now clean for commercial use, requiring only attribution.

2. **Better small-object recall on PPE classes**. RT-DETR is a transformer-based detector with stronger small-object performance than YOLOv8n (the only YOLO size that fits a 4 GB GPU with comfortable headroom). Our benchmark below shows RT-DETR finding **2-3× more Hardhat and Safety Vest detections** on the same test video.

## License attribution

| Component | License | Notes |
|---|---|---|
| RT-DETR architecture | Apache 2.0 | © Baidu Inc. |
| Trained model weights | Apache 2.0 | Trained by us on the dataset below |
| HuggingFace `transformers` | Apache 2.0 | Provides `RTDetrForObjectDetection`, `RTDetrImageProcessor` |
| `supervision` (tracking + Detections) | MIT | Roboflow |
| Construction Site Safety dataset | CC BY 4.0 | Snehil Sanyal / Roboflow Universe; requires attribution if dataset is redistributed |
| PyTorch | BSD-style | |

## Performance comparison

Test bed: 100 frames of `videos/samples/test_video_01.mp4` (1280×720 @ 24 fps, fish-eye CCTV style), confidence threshold 0.4, FP16 on RTX 3050 Laptop GPU (4 GB VRAM).

| Metric                       | YOLOv8n | RT-DETR | Winner |
|------------------------------|--------:|--------:|--------|
| Avg FPS                      |   55.5  |    9.75 | YOLO (5.7×) |
| Avg inference time (ms)      |   16.3  |   100.9 | YOLO        |
| Median inference time (ms)   |   13.9  |    95.4 | YOLO        |
| Peak VRAM (MB)               |     17  |     187 | YOLO (10×)  |
| **Total detections**         |     580 |     **859** | **RT-DETR (+48%)** |

Per-class detections (sum over 100 frames):

| Class           | YOLOv8n | RT-DETR |
|-----------------|--------:|--------:|
| Person          |     294 |     207 |
| Hardhat         |      91 | **280** |
| Safety Vest     |     147 | **355** |
| machinery       |      —  |     —   |
| vehicle         |      17 |       0 |
| NO-Hardhat      |      10 |       0 |
| NO-Safety Vest  |      —  |     —   |
| NO-Mask         |      21 |      17 |
| Mask            |      —  |     —   |
| Safety Cone     |      —  |     —   |

End-to-end through the live simulator (with HUD overlay + tracking + display compositing):

| Detector | Effective FPS | Peak VRAM |
|----------|--------------:|----------:|
| YOLO     | 16.6 fps     | 16 MB    |
| RT-DETR  | 8.3 fps      | 152 MB   |

## Honest assessment of the trade

RT-DETR is the right choice for compliant-PPE detection — it sees roughly 3× as many helmets and 2.4× as many vests on this test footage. However it has two real downsides relative to YOLOv8n:

- **5-6× slower per frame.** RT-DETR is a transformer that runs ~100 ms/frame versus YOLO's ~16 ms. On RTX 3050 we get ~10 fps single-detector, ~8 fps end-to-end through the live simulator. Source video is 24 fps, so RT-DETR can't keep up real-time without frame skipping.
- **Misses some classes entirely on this test video** — `vehicle` (0 vs 17), `NO-Hardhat` (0 vs 10). Possibly a checkpoint-quality issue (recall on those classes was lower at training time), or a confidence-threshold artifact (the chosen 0.4 may sit above this model's NO-Hardhat distribution). Worth investigating either by lowering the threshold for those classes or by training another epoch with class-balanced loss.

For Phase 2 (violation engine), the missing `NO-Hardhat` is concerning — that's the headline violation class. Either:

1. **Lower the conf threshold** (e.g. 0.25) and let the violation engine debounce by `MIN_VIOLATION_DURATION_SECONDS` to reject false positives.
2. **Hybrid mode**: keep RT-DETR for the compliant detection path (where it dominates) and continue using YOLOv8 only for the `NO-Hardhat` / `NO-Safety Vest` / `vehicle` recall — but this re-introduces AGPL.
3. **Re-train**: another fine-tuning pass focused on under-detected classes.

Given the licensing constraint, option (1) is the cleanest first step. We can tune conf thresholds per-class once we have user-validated ground truth.

## How to switch detectors via CLI

```powershell
# default (RT-DETR)
python backend\inference\live_simulator.py

# explicitly RT-DETR
python backend\inference\live_simulator.py --detector rtdetr

# legacy YOLOv8 (still works, kept for comparison only)
python backend\inference\live_simulator.py --detector yolo

# point at a different model checkpoint
python backend\inference\live_simulator.py --detector rtdetr --model-path backend\models\rtdetr_isgozcu_ppe_v2

# lower conf threshold for RT-DETR to recover the missing NO-* classes
python backend\inference\live_simulator.py --detector rtdetr --conf 0.25

# headless A/B benchmark (no GUI)
python backend\inference\live_simulator.py --detector rtdetr --headless --max-frames 200 --snapshot-every 100
python backend\inference\live_simulator.py --detector yolo --headless --max-frames 200 --snapshot-every 100
```

## Internal architecture

The migration kept all UI / HUD / loop / snapshot code unchanged. The only structural change is the introduction of a `DetectorAdapter` that provides a unified `detect_and_track(frame_bgr) -> list[dict]` API regardless of backend.

```
backend/inference/
├── live_simulator.py          # unchanged HUD / Turkish text / controls
├── detector_adapter.py        # NEW: unified interface
├── rtdetr_detector.py         # NEW: RT-DETR + supervision ByteTrack
└── ...
```

Each standardized detection has the keys:

```python
{
    "bbox": (x1, y1, x2, y2),  # ints, original frame coords
    "class_id": int,
    "class_name": str,
    "confidence": float,
    "track_id": int | None,
}
```

Downstream consumers (`_draw_detections`, `_update_stats`) only see this dict — they don't know which model produced it. To plug in a new detector backend later, implement a class with the same interface and add a branch in `DetectorAdapter.__init__`.

## Threshold Tuning (Phase 2A.1)

After the initial migration, RT-DETR was missing all `NO-Hardhat` and `NO-Safety Vest` detections at the inherited `conf=0.4` threshold. We swept thresholds `[0.40, 0.30, 0.25, 0.20, 0.15]` on the same 100 frames of `test_video_01.mp4`, running inference *once per frame* and post-processing the same logits at each threshold (so comparisons are exact, not noisy). Script: `tests/test_rtdetr_thresholds.py`.

### Detection counts vs threshold

| class             |   0.40 |   0.30 |   **0.25** |   0.20 |   0.15 |
|-------------------|-------:|-------:|-----------:|-------:|-------:|
| Hardhat           |    344 |    395 |    **446** |    511 |    637 |
| Mask              |      0 |      3 |      **9** |     26 |     90 |
| NO-Hardhat        |      1 |      3 |      **7** |     17 |     45 |
| NO-Mask           |     19 |     34 |     **42** |     67 |    101 |
| NO-Safety Vest    |      0 |     23 |     **73** |    120 |    221 |
| Person            |    242 |    360 |    **482** |    659 |   1035 |
| Safety Cone       |      0 |      0 |      **3** |     10 |     23 |
| Safety Vest       |    448 |    603 |    **711** |    909 |   1293 |
| **TOTAL**         |   1054 |   1421 |   **1773** |   2319 |   3445 |
| **NO-* (violations)** |  20 |     60 |    **122** |    204 |    367 |
| **Compliant**     |    792 |   1001 |   **1166** |   1446 |   2020 |

`vehicle` was missing at every threshold (0 detections at 0.15 too) — that is a **training-recall problem, not a threshold problem**.

### Visual review at threshold 0.25

5 sample frames saved to `data/snapshots/threshold_test/` (frames 10, 30, 50, 70, 90). Inspection findings:

- **Frames 50, 70**: clean. Foreground worker correctly classified with high confidence (Hardhat 0.77–0.88, Safety Vest 0.62–0.89). Background workers also picked up.
- **Frame 10**: contains one false `NO-Safety Vest 0.28` overlapping a `Safety Vest 0.27` on a worker who is wearing a vest — model double-detects on the same area.
- **Frame 30**: contains one false `Mask 0.29` on a worker without a mask.
- Borderline detections at 0.25–0.40 confidence are common, mostly correct.

Most false positives are single-frame flickers. The **violation engine's `MIN_VIOLATION_DURATION_SECONDS=5` debouncing** will reject these — they need to persist across many frames to fire an alarm.

### Live simulator at the new threshold

```
python backend/inference/live_simulator.py --detector rtdetr --conf 0.25 --headless --max-frames 200
```

Result over 200 frames (1 loop of `test_video_01.mp4`):

| Metric                    | Was (conf=0.30) | Now (conf=0.25) |
|---------------------------|----------------:|----------------:|
| Avg FPS                   |            8.28 |        **8.33** |
| Peak VRAM                 |          152 MB |          152 MB |
| Unique workers            |               8 |          **14** |
| **Violation track IDs**   |           **0** |           **3** |
| Per-frame Hardhat         |            2.93 |            3.22 |
| Per-frame Safety Vest     |            4.27 |            5.20 |
| Per-frame Person          |            2.77 |            3.21 |

Compliance-class growth is moderate (+10–22% per frame), suggesting the lower threshold mostly recovers legit detections rather than flooding false positives. **Violations now fire end-to-end** through ByteTrack, which is the key product unlock.

### Decision

- **Default `RTDetrDetector.conf_threshold`: 0.40 → 0.25.**
- **Default `CONFIDENCE_THRESHOLD` in `.env.example`: 0.4 → 0.25.** (Existing `.env` is untouched so user overrides win.)
- **`vehicle` is NOT recoverable via threshold tuning** — needs a re-training pass or a hybrid setup (use YOLO for `vehicle` detection only). Track this as a known model-quality issue.

## Known differences vs YOLOv8

- **Tracker behavior**: RT-DETR uses supervision's ByteTrack with `minimum_consecutive_frames=2`, which means the first 1-2 frames after start return zero confirmed tracks even if `detect()` returns boxes. YOLO's ultralytics tracker confirms tracks immediately. Visible as a brief "warmup" delay on RT-DETR runs.
- **Loop behavior**: When the video loops, RT-DETR's tracker is *not* automatically reset (we keep `lost_track_buffer=150`, ≈5 s). Workers re-appearing within the buffer can keep their IDs across the loop boundary; workers absent longer get fresh IDs. YOLO's behavior is similar but driven by `bytetrack.yaml`'s `track_buffer=30` (1 s at 30 fps).
- **VRAM headroom**: RT-DETR uses ~150-190 MB peak versus YOLOv8n's ~16 MB. We have 4 GB available, so this is not a problem.
- **Cold start**: RT-DETR first inference takes ~600-1000 ms (CUDA + transformers JIT). Plan for this in any timing-sensitive caller.
