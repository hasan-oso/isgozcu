# Supabase Integration (Phase 2C.2)

Optional persistence layer that streams violation events and evidence snapshots from the live simulator to a Supabase project.

## Architecture

```
                  Live Simulator (single process, no threads)
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │  DetectorAdapter ─► IoUMatcher ─► ViolationDetector          │
   │                                          │                   │
   │                                          ▼                   │
   │                              ┌──────────────────────┐        │
   │                              │   result['new']      │        │
   │                              │   result['resolved'] │        │
   │                              └──────────────────────┘        │
   │                                          │                   │
   │     ┌────────────────────────────────────┴──────────────┐    │
   │     │                                                   │    │
   │     ▼                                                   ▼    │
   │  SnapshotSaver       (local: data/snapshots/violations/.. )  │
   │  JSON report writer  (local: data/reports/<session>.json )   │
   │  EventUploader  ─►  Supabase                                 │
   │                          │                                   │
   └──────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
                  ┌─────────────────────────────────────┐
                  │ Supabase project                    │
                  │  ─ events     (1 row per violation) │
                  │  ─ workers    (per-tracker_id PII-  │
                  │                 free, KVKK-friendly)│
                  │  ─ cameras    (1 row per CCTV)      │
                  │  ─ sites      (1 row per şantiye)   │
                  │  ─ snapshots/ (public bucket, JPEG) │
                  └─────────────────────────────────────┘
```

The Supabase layer is **opt-in** (`--supabase`) and **never blocks** the local pipeline. Every network call is wrapped in `try/except`; failure logs at ERROR and the HUD's sync dot turns red. The simulator keeps producing local snapshots and JSON reports regardless.

## Schema

| Table | Columns | Notes |
|---|---|---|
| `sites` | `id` (uuid pk), `name`, `location` | One row per construction site. Demo: "Demo Şantiye" / "Ankara, Türkiye". |
| `cameras` | `id` (uuid pk), `site_id` (fk), `name` | UNIQUE(site_id, name). Demo: "CCTV_01". |
| `workers` | `id` (uuid pk), `site_id` (fk), `tracker_id` (text), `first_seen`, `last_seen`, `total_violations` | UNIQUE(site_id, tracker_id). `tracker_id` is a string like `"W007"` derived from ByteTrack — *never* a real name. |
| `events` | `id` (uuid pk), `camera_id` (fk), `worker_id` (fk), `violation_type`, `violation_label_tr`, `started_at`, `ended_at`, `duration_seconds`, `snapshot_url`, `confidence_avg`, `is_active`, `metadata` (jsonb), `created_at` | One row per violation. INSERT on FIRE, UPDATE on RESOLVE. `id` is the same UUID we use as the snapshot file name. |
| `snapshots` (storage bucket) | path: `violations/YYYY-MM-DD/{event_id}.jpg`, public read | The DB row stores the public URL in `events.snapshot_url`. |

## How to use the `--supabase` flag

Smoke test first (verifies your `.env` and bucket policy):

```powershell
python tests\test_supabase_upload.py
```

Expected output: 9 lines starting with `PASS` and `RESULT: SMOKE TEST PASSED`. The script writes a 200×200 red square + a fake `ViolationEvent`, verifies the row is queryable and the public URL returns the same bytes, then deletes both.

Then run the live simulator with uploads enabled:

```powershell
# Default (RT-DETR v2 + violation engine + Supabase, no GUI)
python backend\inference\live_simulator.py --supabase --headless --max-frames 600

# Looser thresholds to actually fire events on a mostly-compliant test video
python backend\inference\live_simulator.py --supabase ^
    --detector yolo --conf 0.4 --violation-threshold 0.15 --min-duration 3.0 ^
    --headless --max-frames 600

# Custom site / camera (creates rows on first use)
python backend\inference\live_simulator.py --supabase ^
    --site-name "Şantiye 2" --camera-id "CCTV_GIRIS"
```

## What the HUD shows

When `--supabase` is on:

- **Sync dot** next to the LIVE badge in the top-right.
  - Green: last upload succeeded (or idle)
  - Yellow / cyan: in flight
  - Red: most recent upload errored (logs have details)
  - Gray: Supabase not enabled (default) or init failed
- **`Yüklenen: N`** in the bottom-right stats — count of successful event INSERTs in this session.

## Sample event row

```json
{
  "id": "1dd0abbc-1f1a-471c-ade0-52ffdf349dc4",
  "camera_id": "eb5927e4-d3a1-4fe4-9055-926bb5e5f101",
  "worker_id": "eab7add8-045c-4439-ad52-46217b5815e9",
  "violation_type": "no_hardhat",
  "violation_label_tr": "Baret yok",
  "started_at": "2026-05-08T02:11:44.980726+00:00",
  "ended_at":   "2026-05-08T02:11:50.146672+00:00",
  "duration_seconds": 5.17,
  "snapshot_url": "https://xxxxxxx.supabase.co/storage/v1/object/public/snapshots/violations/2026-05-08/1dd0abbc-1f1a-471c-ade0-52ffdf349dc4.jpg",
  "confidence_avg": 0.534,
  "is_active": false,
  "metadata": { "tracker_id": "W041" },
  "created_at": "2026-05-08T02:11:46.999097+00:00"
}
```

## Privacy / KVKK note

İşGözcü stores **no biometric data**. The `workers` table only contains:

- `tracker_id` — the ephemeral string `"W007"` produced by ByteTrack. This identifier is generated per-session per-camera by an object tracker; it cannot be linked back to a real person.
- `first_seen`, `last_seen`, `total_violations` — operational counters only.

There is no facial recognition, no face embeddings, no name field, no employee ID. Snapshots are cropped person bounding boxes used as evidence; they are *not* used for training and not used for re-identification.

When a worker disappears for longer than ByteTrack's `track_buffer` (300 frames, ≈12.5 s @ 24 fps), they are unlinkable from any future detection. This is a deliberate KVKK-friendly trade: same physical worker may appear as multiple `tracker_id`s across long sessions, but no PII is ever stored.

## Failure modes & fallbacks

| Failure | What happens | What you see |
|---|---|---|
| `.env` missing `SUPABASE_URL` or `SUPABASE_SERVICE_ROLE_KEY` | `SupabaseClient` raises in `__init__`; live simulator catches and falls back to local-only | `[uyarı] Supabase başlatma hatası: ... — yerel modda devam.` HUD dot stays gray. |
| Network down at startup | `health_check()` returns False → simulator runs local-only | `[uyarı] Supabase health check başarısız — yerel modda devam.` |
| Network down mid-session | individual upload methods log ERROR and return None; simulator keeps running | HUD dot turns red. Local snapshots + JSON report still written. |
| Event ID collision (re-running with same id) | INSERT raises, but the `upload_event` try/except catches it | ERROR in log, sync dot red, simulator continues. |
| Snapshot upload succeeds but DB INSERT fails | Orphan file in bucket, no DB row | Cleanup is manual today; future cron job will sweep orphans. |
| Stale `is_active=true` events from a crashed session | They sit in DB with `duration_seconds=0` | Future improvement: cron that closes events whose `started_at` is older than N hours. |

## Troubleshooting

**`SupabaseClient init failed: SUPABASE_SERVICE_ROLE_KEY not set in .env`**
Copy `.env.example` to `.env` and fill in the real values. Don't commit `.env` (it's in `.gitignore`).

**`PermissionError` / `403` / `policy violation` on storage upload**
Bucket isn't public or RLS denies your service-role key. Easiest fix: set the `snapshots` bucket to "public" in the Supabase dashboard. The service role key normally bypasses RLS but bucket-level public read is what makes the URLs accessible to the future frontend.

**Events fire locally but no rows appear in Supabase**
Check the console for `[uyarı] Supabase ...` lines. The most common cause is `--supabase` was not passed (it's opt-in). Confirm with `--help` that the flag is there and look for `[bilgi] Supabase aktif` at startup.

**Snapshot URL returns 404**
Bucket is private. Either make it public, or generate signed URLs (not implemented yet — Phase 2C.3 candidate).

## Internal files

- `backend/storage/__init__.py` — re-exports the four classes
- `backend/storage/supabase_client.py` — `SupabaseClient`, `SiteManager`, `WorkerRegistry`, `EventUploader`
- `backend/inference/live_simulator.py` — `--supabase` / `--site-name` CLI args, `_supabase_upload_new` / `_supabase_upload_resolution` hooks, HUD sync dot + `Yüklenen` counter, finalize step that closes still-active events on session exit
- `tests/test_supabase_upload.py` — end-to-end smoke test (insert + verify + cleanup)
