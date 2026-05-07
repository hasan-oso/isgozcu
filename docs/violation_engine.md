# Violation Engine (Phase 2B)

Turns noisy per-frame PPE detections into structured, debounced violation
events with snapshots and JSON reports.

## Architecture

```
                        DetectorAdapter
              (RT-DETR or YOLO + ByteTrack)
                              |
                              v
              [list of standardized detections]
                  bbox / class / conf / track_id
                              |
        IoUMatcher.match_frame  (assigns PPE to person regions)
                              |
                              v
                      [list of PPEMatch]
              (worker_id, has_hardhat/vest/mask, frame, ts)
                              |
                              v
            ViolationDetector.process_matches
              |                       |
              v                       v
       WorkerStateBuffer       Active state machine
       (per-worker deque)      (fire / ongoing / resolve)
              |                       |
              `--> ratio --> compares against thresholds
                              |
                              v
          { 'new': [...], 'ongoing': [...], 'resolved': [...] }
                              |
                              v
              SnapshotSaver (on 'new')   +
              live_simulator HUD (red bbox / warning) +
              JSON report (on session end)
```

## Configuration

| Parameter                  | Default | Tuning notes |
|----------------------------|--------:|--------------|
| `min_violation_duration`   |    5.0 s | Sliding window over which the ratio is computed. Worker must have been observed ≥ this long before they can fire. Lower = faster alerts, more flicker risk. |
| `violation_threshold`      |    0.7  | Fraction of recent observations that must show non-compliance to FIRE. Higher = stricter, fewer false alarms; lower = more events. |
| `recovery_threshold`       |    0.3  | Fraction below which an active violation ENDS. Auto-clamped to `< violation_threshold` for proper hysteresis (auto-prints a warning if the user supplies a bad combo). |
| `fps`                      |   source FPS | Used to size the buffer (`window_size = round(window_seconds * fps) + 1`). Live simulator passes `source_fps` (not pipeline FPS). |

**Hysteresis matters.** If `recovery_threshold` ≥ `violation_threshold`, every fired event resolves on the very next frame and the report fills up with thousands of 0.05-second flickers. The engine auto-clamps to `violation_threshold * 0.5` and prints `[uyarı]` if the user supplies an invalid combo.

## IoU matching strategy

For each tracked Person bbox, three sub-regions are computed:

| Region | Definition                  | Used for                |
|--------|-----------------------------|-------------------------|
| Head   | Upper 30% of person bbox    | Hardhat / NO-Hardhat    |
| Face   | Upper 20% of person bbox    | Mask / NO-Mask          |
| Torso  | Rows 20% – 80% of bbox      | Safety Vest / NO-Vest   |

A PPE detection matches a person's region if **either**:
- the PPE bbox center is inside the region, OR
- the IoU between PPE bbox and the region ≥ 0.05

When both a positive (e.g. `Hardhat`) and a negative (`NO-Hardhat`) overlap the same region, `_resolve` gives the benefit of the doubt to the positive class (compliant) when its confidence is within 0.1 of the negative; otherwise the higher confidence wins. Rationale: PPE-detection models we have tested produce more false negatives than false positives, and incorrectly flagging a compliant worker is more harmful operationally than missing a violation in a single frame (the debouncing window will still catch sustained violations).

## Sample violation event JSON

```json
{
  "event_id": "53aa138e-f056-4434-876c-0aba0fa0f44b",
  "worker_id": 2,
  "violation_type": "no_hardhat",
  "violation_label_tr": "Baret yok",
  "camera_id": "CCTV_01",
  "started_at": "2026-05-07T22:05:14.848",
  "ended_at": "2026-05-07T22:05:30.077",
  "duration_seconds": 15.23,
  "snapshot_path": ".../data/snapshots/violations/2026-05-07/no_hardhat_W002_220514_166.jpg",
  "confidence_avg": 0.534,
  "is_active": false
}
```

`violation_type` is one of `no_hardhat`, `no_vest`, `no_mask` (stable identifiers). `violation_label_tr` is the Turkish display string (`Baret yok`, `Yelek yok`, `Maske yok`).

## Session report

Written on clean exit (Q / ESC) to `data/reports/{session_id}_violations.json`:

```json
{
  "session_id": "aa42bf4857db",
  "started_at": "...",
  "ended_at": "...",
  "total_duration_seconds": 24.358,
  "camera_id": "CCTV_01",
  "video_source": ".../videos/samples/test_video_01.mp4",
  "detector_type": "yolo",
  "loop_count": 2,
  "frames_processed": 500,
  "total_violations": 2,
  "violations_by_type": { "no_hardhat": 2, "no_vest": 0, "no_mask": 0 },
  "unique_workers_violated": 2,
  "average_violation_duration": 11.048,
  "violations": [ /* full ViolationEvent.to_dict() entries */ ]
}
```

## Snapshots

Saved to `data/snapshots/violations/{YYYY-MM-DD}/{type}_W{id:03d}_{HHMMSS}_{ms}.jpg`.

Each is a tight crop of the violator with 60 px padding (clamped to frame edges) and a translucent black header band overlaid:

- Line 1 (red, large): `Baret yok` / `Yelek yok` / `Maske yok`
- Line 2 (white): `İşçi: W007`
- Line 3 (white): `Zaman: 14:32:18`

JPEG quality 90. Snapshot is captured **only on the first frame the violation fires** — not every frame of the active period, to keep disk usage bounded.

## Live HUD treatment for active violations

Workers currently in an active violation get distinct visual styling on the live display:

| Element             | Compliant worker | Active-violation worker |
|---------------------|------------------|--------------------------|
| Person bbox color   | Blue (200,100,0) | **Red (0,0,255)**        |
| Person bbox thickness | 2 px           | **4 px**                 |
| Worker ID badge     | Yellow text      | **Red badge, white text** (pulses 1 Hz) |
| Above-bbox warning  | (none)           | `⚠ Baret yok` in red     |

The bottom-right HUD also flips `Aktif İhlal:` to red when `> 0`.

## How to read snapshots

The crop is always the offending worker's Person bbox + 60 px padding. Look at the worker carefully — the model can flag a worker as `NO-Hardhat` for several reasons:

1. **Real violation** — worker is genuinely bareheaded.
2. **Helmet color the model wasn't trained on** — e.g. dark brown or unusual shape.
3. **Camera angle** — worker's head occluded by other workers, equipment, or out of frame.
4. **Model false positive** — small heads at distance can confuse YOLOv8n in particular.

The `confidence_avg` field gives a hint of model certainty; below ~0.5 the snapshot is worth a manual review before counting as a real safety incident.

## Known limitations

1. **Cross-loop tracking IDs are not stable.** When the test video loops, ByteTrack assigns new track IDs to workers that re-appear after the buffer interval. So one physical worker can produce multiple `worker_id`s and therefore multiple separate violation events across a long session. Mitigation: longer ByteTrack `track_buffer` (Phase 2C+).
2. **Same physical worker, different IDs, same violation** appears as multiple events in the JSON. Aggregation by physical identity needs face-/gait-based re-id, which is out of scope (KVKK constraints, anyway — we deliberately don't do face recognition).
3. **`vehicle` class is not detected by RT-DETR** at any threshold (training-recall gap). Documented in `docs/migration_yolo_to_rtdetr.md`.
4. **Snapshot is one frame per event.** No video clip extraction yet; planned for Phase 2D when we add report generation.
5. **The matcher's "positive wins ties" rule** can suppress real violations when the model double-detects the same head as both Hardhat and NO-Hardhat. We chose this trade because operationally false alarms are worse than missed single-frame violations (debouncing catches sustained ones).

## CLI

```powershell
# default — RT-DETR, conf 0.3, violations on, 5 s / 70 % defaults
python backend\inference\live_simulator.py

# tighter violation logic
python backend\inference\live_simulator.py --min-duration 8.0 --violation-threshold 0.85

# loose for noisy or short videos (engine will auto-clamp recovery threshold)
python backend\inference\live_simulator.py --violation-threshold 0.3 --min-duration 2.0

# disable the engine entirely (just detection / tracking)
python backend\inference\live_simulator.py --disable-violations

# legacy YOLO + violations
python backend\inference\live_simulator.py --detector yolo --conf 0.4 --violation-threshold 0.5

# headless smoke run with explicit exit (writes JSON report)
python backend\inference\live_simulator.py --headless --max-frames 500
```

## Files

- `backend/inference/violation_engine.py` — `PPEMatch`, `ViolationEvent`, `IoUMatcher`, `WorkerStateBuffer`, `ViolationDetector`, plus Turkish label lookup.
- `backend/inference/snapshot_saver.py` — `SnapshotSaver` class.
- `backend/inference/text_render.py` — Turkish-safe Pillow text helper, shared by `live_simulator.py` and `snapshot_saver.py`.
- `backend/inference/live_simulator.py` — wires the engine in, drives the HUD treatment, writes the session JSON.
- `tests/test_iou_matcher.py` — 11 unit tests for `IoUMatcher` (geometry).
- `tests/test_violation_engine.py` — 6 synthetic tests for the state machine (fire / no-fire / resolve / finalize / dict).
