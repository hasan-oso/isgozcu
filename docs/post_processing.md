# Detection Post-Processing (Phase 2A.7)

A small library of class-specific filters that sit between RT-DETR's raw
output and the violation engine. None of them require retraining the model
— they exploit domain knowledge to scrub failure modes that v3 has
inherited from its training data.

## Why this exists

After v1–v5 training experiments, three v3 failure modes proved impossible
to fix at the model level on our hardware budget:

| Symptom | Root cause |
|---|---|
| `Person` boxes on shadows / chairs / wall textures | The Roboflow training set has noisy `Person` annotations; v3 reproduces them. |
| `Safety Vest` ↔ `NO-Safety Vest` flicker (52 : 48 ratio in webcam runs) | Vest features look similar at low confidence; the model oscillates frame-to-frame. |
| Spurious `NO-Hardhat` over a real Hardhat box | Both heads activate at low conf on the same head region. |

Rather than retrain (expensive on RTX 3050, with diminishing returns), we
fix these in software. The post-processor is opt-out (`--no-post-process`)
so we can A/B at any time.

## Architecture

```
  ┌────────────────────────────┐
  │   RTDetrDetector            │  raw confidences (low conf=0.25 floor)
  │   conf_threshold = 0.25     │  ────────────────┐
  └────────────────────────────┘                    │
                                                    ▼
                              ┌──────────────────────────────────┐
                              │  DetectionPostProcessor (NEW)    │
                              │   1. class-specific conf floors  │  ← stricter
                              │      (Person 0.75, Vest 0.65 …)  │     than 0.25
                              │   2. Person geom validation      │  ← shadow / chair
                              │      area, aspect ratio          │     rejection
                              │   3. Hardhat / NO-Hardhat dedup  │  ← higher-conf wins
                              │      (IoU ≥ 0.50)                │
                              │   4. Vest temporal smoothing     │  ← 30-frame majority
                              │      per track_id, 65% threshold │     vote anti-flicker
                              └──────────────────────────────────┘
                                                    │
                                                    ▼
                              ┌──────────────────────────────────┐
                              │  Violation Engine                │
                              │  threshold=0.85, min_dur=8.0s    │  ← also stricter
                              │  hysteresis: recovery=0.30       │
                              └──────────────────────────────────┘
```

The four filters are applied **in order**; if step N drops a detection it
never sees step N+1. The pipeline is single-pass, no per-frame allocations
beyond the deque windows.

## Filter details

### 1. Class-specific confidence floors

Each class has its own minimum confidence. Looser floors for classes
RT-DETR is reliably good at; stricter for known-flicker classes.

| Class | Floor | Rationale |
|---|---:|---|
| Person | **0.75** | Hardest class; shadows + chairs love to hit 0.50 |
| NO-Safety Vest | 0.70 | Headline flicker source — accept only confident calls |
| Safety Vest | 0.65 | Same flicker family, slightly looser |
| NO-Mask | 0.65 | Often spurious on chins / beards |
| Mask | 0.60 | |
| NO-Hardhat | 0.55 | Important violation class — don't be too strict |
| machinery | 0.55 | |
| vehicle | 0.55 | |
| Hardhat | 0.50 | Reliable; lower floor = more recall |
| Safety Cone | 0.50 | Rare class, recall-favoured |

To tune for your site, edit `CONFIDENCE_FLOORS` in
[backend/inference/post_processor.py](../backend/inference/post_processor.py).

### 2. Person geometric validation

A `Person` box must:

- cover **0.5 % – 85 %** of frame area (rejects 50×50 px shadows AND a sky-
  filling false positive),
- have **height/width ∈ [0.20, 5.0]** (rejects horizontal shadow strips and
  pencil-thin slivers).

These checks only apply to `Person` — other classes like `Hardhat` are
expected to be small relative to the frame and skip this step.

### 3. Hardhat / NO-Hardhat de-duplication

When a `Hardhat` box and a `NO-Hardhat` box overlap with **IoU ≥ 0.50**,
they're predicting opposite states for the same head — only one can be
right. We keep the higher-confidence one and drop the other.

Two `Hardhat` boxes (or two `NO-Hardhat` boxes) overlapping each other are
not de-duplicated by this step; that's NMS territory and lives elsewhere.

### 4. Vest temporal smoothing

Per-tracker rolling window over the last 30 frames (~1 second at 30 fps).
For each track_id we record what it showed (`vest` / `no_vest` / nothing)
on each frame. To classify the *current* frame's verdict we compute:

`vest_ratio = #vest_frames / (#vest_frames + #no_vest_frames)` over the window.

| Condition | Verdict | Action this frame |
|---|---|---|
| `vest_ratio ≥ 0.65` | confident **vest** | drop any `NO-Safety Vest` blip |
| `1 - vest_ratio ≥ 0.65` | confident **no_vest** | drop any `Safety Vest` blip |
| else (ambiguous, history present) | none | drop **both** vest classes this frame |
| < 5 frames of history | (warmup) | accept whatever the model says |

Detections without a `track_id` skip smoothing entirely — there's no
history to compare them to.

The window is **per track_id**, so different workers don't bleed into each
other. When ByteTrack drops a track, that history is reclaimed via
`reset_tracker(tid)`. Video-loop boundaries call `reset_all()`.

## Pairing with the violation engine

PostProcessor + violation-engine defaults work together. The defaults
shipped in Phase 2A.7 are paired:

| Parameter | Old (Phase 2B) | **New (Phase 2A.7)** | Why |
|---|---:|---:|---|
| `min_violation_duration` | 5.0 s | **8.0 s** | Stricter; PP eliminates flicker so we can wait longer |
| `violation_threshold` | 0.7 | **0.85** | Stricter; PP feeds higher-quality signal |
| `recovery_threshold` | 0.3 | 0.3 (unchanged) | Hysteresis between fire (0.85) and resolve (0.30) |

If you turn PP off (`--no-post-process`), you'll see the raw model
behaviour again: Person on shadows, vest flicker, Hardhat double-fires.
The violation engine will then count those toward the ratio and *can* fire
spurious violations — useful for diagnostic A/B but not for production.

## CLI

```powershell
# default (PP on, stricter thresholds)
python backend\inference\live_simulator.py --input 0 --detector rtdetr --supabase

# A/B: same run, raw model output
python backend\inference\live_simulator.py --input 0 --detector rtdetr --no-post-process --supabase

# tune the violation thresholds back down (e.g. for noisy footage)
python backend\inference\live_simulator.py --violation-threshold 0.7 --min-duration 5.0
```

The HUD shows a small `PP: ON` (green) / `PP: OFF` (gray) pill on the
bottom-right of the controls strip so you always know which mode you're in.

## When to disable PP

Three legitimate reasons:

1. **Diagnostic A/B**: confirm that an observed problem is the model's
   fault, not the PP's. Toggle off, see if the symptom changes.
2. **Sparse-footage benchmarks**: when validating against ground-truth
   COCO mAP on the test set, you want raw model output.
3. **Tuning a new threshold**: lower `CONFIDENCE_FLOORS` for a class that
   matters more than the defaults assume, then re-test.

For everything else — webcam, recorded videos, dashboard, Supabase ingest
— leave PP on.

## Tests

```powershell
python tests\test_post_processor.py
```

18 tests, ~3 ms total. Coverage:

- Confidence-floor filtering (3 tests)
- Person geometry rejection (5 tests: too small / huge / thin / wide / normal-kept)
- Hardhat dedup (3 tests: each direction wins, plus non-overlapping pair)
- Vest smoothing (4 tests: blip rejected, consistent NO-Vest kept, no track_id passes through, reset clears history)
- Edge cases (empty input, full realistic frame chain)

Existing test suites continue to pass:
- `tests/test_iou_matcher.py` — 11/11
- `tests/test_violation_engine.py` — 6/6 (defaults bump didn't break them; all tests pass explicit thresholds)

## Files

```
backend/inference/post_processor.py       new — DetectionPostProcessor
backend/inference/live_simulator.py       wire-in: enable_post_process flag,
                                          process() call between detect()
                                          and violation engine, HUD pill,
                                          reset_all() on loop boundary,
                                          new --post-process / --no-post-process
                                          CLI args, default thresholds bumped.
backend/inference/violation_engine.py     defaults bumped to match.
tests/test_post_processor.py              new — 18 tests
docs/post_processing.md                   this file
```
