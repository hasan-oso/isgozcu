# Changelog · Değişiklik Günlüğü · سجل التغييرات

All notable changes to **İşGözcü** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows a phase-based release model (`Phase N.M`) rather than strict [Semantic Versioning](https://semver.org/) until the first stable release.

**🇬🇧 English** · **🇹🇷 Türkçe** · **🇸🇦 العربية** — each entry is bilingual where it matters; technical details are kept in English for grep-friendliness.

> ⚠️ **Pre-release.** Everything below ships as part of an internal development phase. Breaking changes can happen between any two phases without a major-version bump.

---

## [Unreleased]

### Added
- Trilingual project documentation: `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` (EN / TR / AR).
- GitHub Issue templates (Bug report, Feature request) in three languages.
- Formal `LICENSE` (Proprietary — all rights reserved) with EN/TR/AR text.
- `CHANGELOG.md` covering Phases 1 → 3.5.

### Changed
- `.gitignore` now excludes presentation artefacts (`isgozcu_sunum*`), `captures/`, `test.md`, and editor scratch files.

---

## [Phase 3.5] — 2026-05-11

### Changed
- **Dashboard**: replaced Supabase Realtime websocket subscription with a 2-second polling loop on the home page for demo reliability. The `Canlı bağlantı` indicator now reflects polling health, not socket state.

---

## [Phase 3.2] — 2026-05-08

### Added
- **Frontend**: violations, workers, and reports pages.
  - `/ihlaller` — full violations table with date-range / type / camera / status filters in URL params, snapshot thumbnails, detail dialog, pagination.
  - `/raporlar` — 4 stat cards, daily trend chart, type pie chart, hourly bar chart, time-range tabs.
  - `/isciler` — worker grid with search by tracker_id, detail dialog with full event timeline.
- **Realtime auto-refresh** on the home page via Supabase Realtime websocket.

---

## [Phase 3.1] — 2026-05-08

### Added
- **Next.js 16 dashboard** with dark-blue theme (`bg-bg-page`, `bg-bg-panel`, accent `#185fa5`).
- Layout shell, home page with 4 metric cards, live-stream placeholder, recent violations list.
- Inter font via `next/font/google`. All UI strings in Turkish; code in English.

---

## [Phase 2C.2] — 2026-05-08

### Added
- **Supabase persistence layer** (`backend/storage/supabase_client.py`).
- Auto-upload of violation events + JPEG snapshots when `--supabase` flag is on.
- KVKK-compliant schema: 4 tables (`sites`, `cameras`, `workers`, `events`); no biometric storage.
- HUD sync indicator (green/yellow/red/gray dot) + `Yüklenen: N` counter.
- Smoke test `tests/test_supabase_upload.py`.
- Documentation: `docs/supabase_integration.md` with full schema and RLS policy snippets.

### Notes
- Supabase failures never crash the live simulator; it falls back to local-only mode.

---

## [Phase 2B.1] — 2026-05-07

### Changed
- **ByteTrack tuning** for cross-loop ID stability:
  - YOLO pathway: created `backend/inference/configs/bytetrack_custom.yaml` with `track_buffer: 300` (10× ultralytics default), `track_high_thresh: 0.4`, `new_track_thresh: 0.5`, `match_thresh: 0.85`.
  - RT-DETR pathway: bumped `lost_track_buffer` 150 → 300.
- **Measured impact** (600 frames, 3 video loops): YOLO unique worker_ids 31 → 25 (-19%); per-loop IDs 12.4 → 8.3 (-33%).

---

## [Phase 2B] — 2026-05-07

### Added
- **Violation engine** (`backend/inference/violation_engine.py`):
  - `PPEMatch`, `ViolationEvent`, `IoUMatcher`, `WorkerStateBuffer`, `ViolationDetector`.
  - Debouncing via sliding window over `min_violation_duration` seconds.
  - Hysteresis: `violation_threshold` (fire) ≠ `recovery_threshold` (resolve).
  - Turkish display labels (`Baret yok`, `Yelek yok`, `Maske yok`).
- **Snapshot saver** with translucent warning header overlaid on the crop.
- **JSON session reports** written to `data/reports/{session_id}_violations.json` on exit.
- Live HUD treatment for active violations: red bbox (4 px), pulsing red ID badge, above-bbox warning.
- Documentation: `docs/violation_engine.md`.
- Tests: `tests/test_iou_matcher.py` (11), `tests/test_violation_engine.py` (6).

---

## [Phase 2A.7] — 2026-05-10

### Added
- **Post-processing pipeline** (`backend/inference/post_processor.py`):
  - Class-specific confidence floors (Person 0.75, Vest 0.65, NO-Vest 0.70, etc.).
  - Person geometric validation (area % + height/width aspect ratio).
  - Hardhat ↔ NO-Hardhat de-duplication on IoU ≥ 0.50.
  - Per-tracker rolling-30-frame vest temporal smoothing.
- HUD `PP: ON / OFF` pill; toggle with `--no-post-process`.
- 18 unit tests in `tests/test_post_processor.py`.

### Changed
- **Stricter violation-engine defaults** paired with the new post-processor:
  - `min_violation_duration`: 5.0 s → **8.0 s**
  - `violation_threshold`: 0.7 → **0.85**
  - `recovery_threshold`: 0.3 (unchanged, hysteresis preserved)

---

## [Phase 2A.6] — 2026-05-10

### Added
- **RT-DETR v5** checkpoint: produced by continuing v3 training on the cleaned Construction-PPE dataset locally on RTX 3050.
- Local training script `tests/train_v5_local.py`.

---

## [Phase 2A.5.B] — 2026-05-10

### Added
- **RT-DETR v4** integrated alongside v3, selectable via `--model-version v4`.
- 6-class Construction-PPE schema (cleaner labels than the original 10-class set).

---

## [Phase 2A.5] — 2026-05-10

### Added
- Tooling to prepare the Construction-PPE dataset for v4 training (6 clean classes):
  - `tests/prepare_construction_ppe_for_v4.py`
  - `tests/yolo_to_coco_v4.py`

---

## [Phase 2A.4] — 2026-05-10

### Added
- Tools to select and re-annotate 500 Roboflow images for v4:
  - `tests/select_images_for_reannotation.py`
  - `tests/annotate_roboflow_images.py`

---

## [Phase 2A.3.D] — 2026-05-10

### Added
- **RT-DETR v3** checkpoint: fine-tuned with self-captured images for improved indoor / personal-camera performance.

---

## [Phase 2A.3.B] — 2026-05-10

### Added
- Interactive annotation tool with auto-detect + manual correction (`tests/annotate_training_images.py`).

---

## [Phase 2A.3] (data prep) — 2026-05-09

### Added
- Self-image capture tool for v3 fine-tuning dataset (`tests/capture_training_images.py`).
- `live_simulator.py`: webcam input via numeric `--input 0` argument.
- Webcam test script for real-world PPE verification with v2 (`tests/test_webcam.py`).

---

## [Phase 2A.2] — 2026-05-08

### Added
- **RT-DETR v2** checkpoint (continued training from v1 + 15 epochs).
- Class oversampling for `Mask` and `NO-Hardhat` (×3 in batch sampler).
- Stronger augmentation: motion blur, simulated cast shadows, sun-flare overlays.
- v2 is the new default; v1 stays on disk for rollback (`--model-version v1`).

### Changed
- Default detector → v2. Validation loss 9.91 → 9.68.
- Per-class recall: `Mask` +174 %, `NO-Hardhat` +205 %, `vehicle` 0 → 2 detections per 200 frames.
- Headline regressions: `NO-Mask` -23 %, `NO-Safety Vest` -17 % (likely re-classified as positive class — see `docs/migration_yolo_to_rtdetr.md` for analysis).

---

## [Phase 2A.1] — 2026-05-07

### Changed
- RT-DETR confidence threshold default: 0.40 → **0.25** to recover rare `NO-*` classes.
- `.env.example`: `CONFIDENCE_THRESHOLD=0.4` → `0.25`.
- Live simulator now reports 14 unique tracked workers (was 8) and fires violations end-to-end through ByteTrack.

---

## [Phase 2A] (migration) — 2026-05-07

### Changed
- **License-driven migration** from YOLOv8 (AGPL-3.0) to RT-DETR (Apache 2.0).
- Inference stack is now clean for commercial deployment.
- Added `DetectorAdapter` providing a unified `detect_and_track() -> list[dict]` API.

### Added
- `backend/inference/rtdetr_detector.py` — RT-DETR + supervision ByteTrack.
- `backend/inference/detector_adapter.py` — unified detector interface.
- Benchmark: RT-DETR sees +48 % total detections vs YOLOv8n on the same 100 frames; specifically +2-3× on Hardhat and Safety Vest classes.

### Notes
- RT-DETR is 5-6× slower per frame (~100 ms vs ~16 ms) and uses ~10× more VRAM (~187 MB vs ~17 MB), but still well within our 4 GB budget.
- YOLOv8 remains selectable via `--detector yolo` for benchmarking only; not in any production path.

---

## [Phase 2A] (webcam tool) — 2026-05-07

### Added
- Webcam live test app with virtual PPE filter (early experimentation).

---

## [Phase 1.7] — 2026-05-07

### Added
- Live stream simulator with looping CCTV-style display (`backend/inference/live_simulator.py`).

---

## [Phase 1.6] — 2026-05-07

### Added
- PPE detection tested on CCTV-style still images.

---

## [Phase 1.5] — 2026-05-07

### Added
- Detection tested on a second video (`sample_02.mp4`, 1080×1920 portrait).
- Benchmark report: `docs/detection_benchmark.md`.

### Notes
- Steady-state FPS: 30.68 on `sample_01` (landscape 720 p), 16.86 on `sample_02` (portrait 1080 p — 2.25× frame area).

---

## [Phase 1] — 2026-05-07

### Added
- Initial detection pipeline (YOLOv8n + ultralytics tracker) working on the first test video.

---

## [Phase 0] (bootstrap) — 2026-05-07

### Added
- Initial project setup: Python virtualenv, folder structure (`backend/`, `frontend/`, `docs/`, `notebooks/`, `videos/`, `data/`, `tests/`).
- `.gitignore`, `.env.example`, `README.md`, `requirements.txt`.

---

[Unreleased]: ../../compare/master...HEAD
