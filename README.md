# İşGözcü - Şantiye Güvenlik Zekası

AI-powered construction site safety monitoring system for the Turkish market.

Türkiye pazarı için yapay zeka destekli şantiye iş güvenliği izleme sistemi.

## Features / Özellikler

- Real-time PPE detection (helmet, vest, gloves, safety shoes)
- Restricted zone monitoring
- Anonymous worker tracking (KVKK compliant)
- Hourly, daily, weekly, monthly compliance reports
- PDF export for legal records

- Gerçek zamanlı KKD tespiti (baret, yelek, eldiven, iş ayakkabısı)
- Yasaklı bölge izleme
- Anonim işçi takibi (KVKK uyumlu)
- Saatlik, günlük, haftalık, aylık uyum raporları
- Yasal kayıtlar için PDF dışa aktarımı

## Setup / Kurulum

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked by execution policy, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Copy `.env.example` to `.env` and fill in your Supabase credentials:

```powershell
Copy-Item .env.example .env
```

## Project Structure / Proje Yapısı

```
isgozcu/
├── backend/
│   ├── inference/   # YOLO + ByteTrack detection logic
│   ├── api/         # FastAPI endpoints
│   ├── models/      # Downloaded YOLO model weights (.pt)
│   └── utils/       # Shared helpers
├── frontend/        # Next.js dashboard (added later)
├── notebooks/       # Jupyter experiments
├── videos/
│   ├── samples/     # Test input videos
│   └── outputs/     # Processed videos with detections drawn
├── data/
│   └── snapshots/   # Violation evidence images
├── docs/            # Architecture and design docs
├── tests/           # Unit and integration tests
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Technology / Teknoloji

- **Detection**: RT-DETR (Apache 2.0) via HuggingFace `transformers`, fine-tuned on the Construction Site Safety dataset (CC BY 4.0). Active checkpoint is **v2** (`backend/models/rtdetr_isgozcu_ppe_v2`) — continued training from v1 with oversampling for `Mask` / `NO-Hardhat` and stronger augmentation; v1 stays on disk for rollback (`--model-version v1`).
- **Tracking**: ByteTrack via `supervision` (MIT)
- **Legacy**: YOLOv8 (AGPL-3.0) is kept around for benchmarking only — selectable with `--detector yolo` in the live simulator. **Not used in production paths.**
- **Backend**: FastAPI + Uvicorn
- **Database**: Supabase (PostgreSQL)
- **Frontend**: Next.js (separate folder)
- **GPU**: NVIDIA CUDA 12.1 (PyTorch)

### License compliance

The production inference stack is **clean for commercial deployment**:

| Component                              | License     |
|----------------------------------------|-------------|
| RT-DETR architecture (Baidu)            | Apache 2.0  |
| Trained model weights                   | Apache 2.0  |
| `transformers` (HuggingFace)            | Apache 2.0  |
| `supervision` (Roboflow)                | MIT         |
| Construction Site Safety dataset        | CC BY 4.0   |
| PyTorch                                 | BSD-style   |

YOLOv8 (AGPL-3.0) is no longer in any production inference path. See [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md) for the full rationale and the YOLO-vs-RT-DETR benchmark.

### Supabase Setup

Optional persistence layer for violation events + evidence snapshots.

1. Create a Supabase project + a public storage bucket named `snapshots`.
2. Apply the schema (4 tables: `sites`, `cameras`, `workers`, `events`). See [docs/supabase_integration.md](docs/supabase_integration.md) for the schema reference.
3. Copy `.env.example` to `.env` and fill in `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`.
4. Run the smoke test: `python tests\test_supabase_upload.py`.
5. Enable on the live simulator with `--supabase`:
   ```powershell
   python backend\inference\live_simulator.py --supabase
   ```

Supabase failures never crash the simulator — it falls back to local-only mode and prints `[uyarı] Supabase ...` on the console. The HUD shows a small green / yellow / red / gray dot next to LIVE indicating last sync state, plus `Yüklenen: N` in the bottom-right stats panel.

### Frontend Setup (Phase 3.1)

Dark-blue Next.js 16 dashboard for the safety operator. Lives in [frontend/](frontend/) with its own `package.json`, no shared deps with the Python backend.

```powershell
cd frontend
# .env.local is created automatically the first time you run the simulator;
# you can also copy it from the parent .env (only NEXT_PUBLIC_SUPABASE_URL +
# NEXT_PUBLIC_SUPABASE_ANON_KEY are needed).
npm install
npm run dev   # http://localhost:3000
npm run build # production build (Turbopack)
```

Routes today (Phase 3.2):

| Path | Purpose |
|---|---|
| `/`         | Live dashboard — 4 metric cards, live-stream placeholder, recent violations, **realtime auto-refresh** when new events fire |
| `/ihlaller` | Violations table — date range / type / camera / status filters in URL params, snapshot thumbnails, "Detay" dialog with full snapshot + metadata, pagination |
| `/raporlar` | Reports — 4 stat cards (Toplam İhlal / Ortalama Süre / Etkilenen İşçi / En Sık İhlal), daily trend line chart, type pie chart, hourly bar chart. Time range tabs (Bugün / Bu Hafta / Bu Ay / Son 30 Gün) |
| `/isciler`  | Worker grid — search by tracker_id, per-worker card with last violation, detail dialog with full event timeline |
| `/ayarlar`  | Settings (placeholder) |

**Realtime**: the home page subscribes to `events` table changes via Supabase Realtime websocket. When a new violation INSERTs (or one resolves), the page calls `router.refresh()` and the metric cards + recent-violations list update without a full reload. A small "Canlı bağlantı" pill in the page header shows the websocket status (green = connected, yellow = connecting, red = disconnected).

Theme: a dark blue palette (`bg-bg-page`, `bg-bg-panel`, accent `#185fa5`, danger `#d83b3b`) defined in [globals.css](frontend/src/app/globals.css). Inter font via `next/font/google`. All UI strings are in Turkish; code, file names and component identifiers are English.

> **Note**: anonymous reads on the Supabase tables require RLS policies allowing `SELECT` for the `anon` role. See [docs/supabase_integration.md](docs/supabase_integration.md) for the SQL.

## Compliance / Uyum

KVKK (Türkiye Kişisel Verilerin Korunması Kanunu) compliant:

- Workers are tracked anonymously by track ID, not by biometric features
- No facial recognition is performed
- Snapshots are stored only for active violation evidence
- All report exports are audit-logged

## License / Lisans

Proprietary - All rights reserved.
Tescilli - Tüm hakları saklıdır.
