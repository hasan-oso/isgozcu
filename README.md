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

- **Detection & Tracking**: YOLOv8 + ByteTrack (via `ultralytics`)
- **Backend**: FastAPI + Uvicorn
- **Database**: Supabase (PostgreSQL)
- **Frontend**: Next.js (separate folder)
- **GPU**: NVIDIA CUDA 12.1 (PyTorch)

## Compliance / Uyum

KVKK (Türkiye Kişisel Verilerin Korunması Kanunu) compliant:

- Workers are tracked anonymously by track ID, not by biometric features
- No facial recognition is performed
- Snapshots are stored only for active violation evidence
- All report exports are audit-logged

## License / Lisans

Proprietary - All rights reserved.
Tescilli - Tüm hakları saklıdır.
