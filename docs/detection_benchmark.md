# Detection Benchmark — Phase 1.5

Benchmark of `backend/inference/test_detection.py` (YOLOv8n PPE model + ByteTrack)
on the first two test videos.

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.29 GB VRAM), CUDA 12.1
**Model:** `backend/models/ppe_yolov8n.pt` (6.25 MB, 10 PPE classes)
**Settings:** `device=cuda`, `conf=0.4`, `imgsz=640`, `half=True`, `tracker=bytetrack.yaml`

## Source video properties

| Property         | sample_01           | sample_02            |
| ---------------- | ------------------- | -------------------- |
| Resolution       | 1280 × 720 (landscape) | 1080 × 1920 (**portrait**) |
| Frame area (MP)  | 0.92                | 2.07 (2.25× larger)  |
| FPS (source)     | 29.97               | 59.94                |
| Frames           | 448                 | 748                  |
| Duration         | 14.95 s             | 12.48 s              |
| Codec            | h264                | h264                 |
| File size        | 5.07 MB             | 8.77 MB              |

## Detection results

| Metric                       | sample_01 | sample_02 |
| ---------------------------- | --------: | --------: |
| Frames processed             | 448       | 748       |
| Elapsed time (s)             | 16.47     | 45.45     |
| **Average FPS**              | **27.21** | **16.46** |
| **Steady-state FPS (>30)**   | **30.68** | **16.86** |
| Real-time vs source          | 1.02×     | 0.28×     |
| Unique tracked IDs           | 39        | 78        |
| Peak VRAM (MB)               | 15.97     | 15.97     |
| Output file size (MB)        | 61.82     | 140.25    |

> "Real-time vs source" = `steady_fps_30 / source_fps`. Above 1.0 means we can keep up live.

## Detections per class

| Class           | sample_01 | sample_02 |
| --------------- | --------: | --------: |
| Person          | 205       | 1295      |
| Hardhat         | 71        | 185       |
| Safety Vest     | 0         | 882       |
| machinery       | 16        | 63        |
| NO-Safety Vest  | 5         | 60        |
| NO-Mask         | 2         | 33        |
| NO-Hardhat      | 0         | 9         |
| vehicle         | 0         | 1         |
| Mask            | 0         | 0         |
| Safety Cone     | 0         | 0         |

(Counts are per-frame detections, not per-track.)

## Key observations

### Performance

- **sample_01 runs in real-time** (30.68 FPS steady-state vs 29.97 source). Good.
- **sample_02 runs at ~28% of real-time** (16.86 steady-state vs 59.94 source). Still fine for offline processing but not for live streams.
- VRAM is identical (15.97 MB) because YOLO resizes both inputs to `imgsz=640` internally — model compute is constant. The slowdown on sample_02 is from **frame I/O + pre/postprocessing**, not inference: decoding 1080×1920 frames, letterboxing them, drawing 1080×1920 annotations, encoding the output video. We have huge VRAM headroom either way.
- The first frame of each run takes ~1–2 s (CUDA + tracker JIT). Reflected in the gap between avg and steady-state FPS.

### Detection quality

- **`Safety Vest` finally fires** (882 detections in sample_02 vs 0 in sample_01). The model knows the class — sample_01 just didn't show vests at recognizable angles.
- **`NO-Hardhat` finally fires too** (9 in sample_02 vs 0 in sample_01). This means the model *can* detect bare-headed workers; sample_01 simply didn't have them.
- **Person:Hardhat ratio is suspicious**:
  - sample_01: 205 persons / 71 hardhats = 35% with helmets
  - sample_02: 1295 persons / 185 hardhats = **14% with helmets**
  - Even if many workers are wearing helmets, the model should produce far more `Hardhat` *or* `NO-Hardhat` detections per `Person`. Right now `Hardhat + NO-Hardhat` = 194 vs 1295 persons → the model is missing helmet/no-helmet labels on **~85% of person detections**. This is the headline weakness.
- `Mask` and `Safety Cone` were never detected in either video. Could be footage-driven (not present), but worth probing.

### What this tells us about the model

The Construction-Site-Safety YOLOv8n model is the smallest of the YOLO family (3M params). It's good at detecting *prominent* objects (workers, vests, machinery) but appears to under-recall small objects like helmets — especially when heads are far from the camera or partially occluded. This is consistent with known YOLOv8n limitations on small-object detection.

## Recommendation

For Phase 2 (violation engine):

1. **Don't gate Phase 2 on switching models yet.** YOLOv8n is producing usable signal — we have enough to build the violation engine and KVKK pipeline against. Validating the engine against existing detections is more valuable than chasing accuracy first.
2. **Plan to upgrade to YOLOv8m or YOLOv8s after the violation engine works.** YOLOv8m is ~25M params, expected ~3-5× slower but materially better small-object recall — still well under our 4 GB VRAM budget. We can A/B compare on the same videos using the benchmark in this file.
3. **For the portrait/high-FPS sample**: consider a frame-skip option (process every Nth frame) for offline jobs where 16 FPS sustained is tight. Live mode would always need frame skipping at 60 FPS anyway.
4. **Consider better output codec** (`avc1` H.264 instead of `mp4v`) — would shrink the 140 MB output to ~15-20 MB.
