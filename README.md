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

## Compliance / Uyum

KVKK (Türkiye Kişisel Verilerin Korunması Kanunu) compliant:

- Workers are tracked anonymously by track ID, not by biometric features
- No facial recognition is performed
- Snapshots are stored only for active violation evidence
- All report exports are audit-logged

## License / Lisans

Proprietary - All rights reserved.
Tescilli - Tüm hakları saklıdır.
