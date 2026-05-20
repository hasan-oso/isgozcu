<div align="center">

# İşGözcü

### Şantiye Güvenlik Zekâsı · AI-Powered Construction Site Safety · ذكاء سلامة مواقع البناء

![Status](https://img.shields.io/badge/status-in%20development-orange)
![Phase](https://img.shields.io/badge/phase-3.5-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Proprietary-red)
![KVKK](https://img.shields.io/badge/KVKK-compliant-green)
![Supervisor](https://img.shields.io/badge/supervisor-Dr.%20Murat%20%C5%9Eim%C5%9Fek-blueviolet)

**🇬🇧 [English](#-english)** · **🇹🇷 [Türkçe](#-türkçe)** · **🇸🇦 [العربية](#-العربية)**

---

> ⚠️ **Pre-release · Geliştirme aşamasında · قيد التطوير**
> This project is under active development. Features, APIs and the data model may change without notice. Not yet production-ready.

</div>

---

## 🇬🇧 English

**İşGözcü** is a real-time computer-vision system that monitors construction sites for Personal Protective Equipment (PPE) compliance. It detects whether workers are wearing helmets, vests, masks and other safety equipment, and produces auditable violation reports — all while staying compliant with Turkey's KVKK personal-data law.

### 🎓 Academic Supervision

This project was developed under the academic supervision of  
**Dr. Murat Şimşek**  
OSTİM Technical University, Department of Artificial Intelligence Engineering, Ankara

The methodological framework and computer vision approach were guided by Dr. Şimşek's expertise.

---


### ✨ Key features

- **Real-time PPE detection** — helmet, vest, mask, gloves, safety shoes (10 classes)
- **Restricted-zone monitoring** with anonymous worker tracking (ByteTrack)
- **KVKK-compliant** — no facial recognition, no biometric storage
- **Post-processing pipeline** — class-specific confidence floors, geometric validation, temporal smoothing
- **Violation engine** with debouncing + hysteresis to suppress single-frame flicker
- **Hourly, daily, weekly, monthly compliance reports** with PDF export
- **Next.js dashboard** with live MJPEG stream + real-time violation feed
- **Optional Supabase persistence** for events, snapshots and dashboards

### 🛠 Tech stack

| Layer        | Technology |
|--------------|------------|
| Detection    | RT-DETR (Apache 2.0) via HuggingFace `transformers`, fine-tuned on Construction Site Safety (CC BY 4.0) |
| Tracking     | ByteTrack via `supervision` (MIT) |
| Backend      | FastAPI + Uvicorn |
| Database     | Supabase (PostgreSQL) |
| Frontend     | Next.js 16 (dark-blue dashboard) |
| GPU          | NVIDIA CUDA 12.1 (PyTorch 2.5) |

### 🚀 Quick start

```powershell
# 1) Create Python environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Configure secrets
Copy-Item .env.example .env
# then edit .env and fill in your Supabase credentials

# 3) Run the live simulator (webcam)
python backend\inference\live_simulator.py --input 0 --detector rtdetr --supabase --stream

# 4) Run the dashboard (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:3000
```

> If PowerShell blocks activation, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 📁 Project structure

```
isgozcu/
├── backend/
│   ├── inference/   # RT-DETR + ByteTrack + violation engine + post-processor
│   ├── streaming/   # MJPEG HTTP server (FastAPI)
│   ├── storage/     # Supabase client
│   ├── api/         # FastAPI endpoints
│   ├── models/      # Downloaded model weights (.gitignored)
│   └── utils/
├── frontend/        # Next.js 16 operator dashboard
├── docs/            # Architecture, benchmarks, design notes
├── notebooks/       # Jupyter experiments
├── videos/          # Test footage (.gitignored)
├── data/snapshots/  # Violation evidence (.gitignored)
├── data/reports/    # Session JSON reports (.gitignored)
├── tests/           # Unit + integration + capture tools
├── .env.example
├── README.md
└── requirements.txt
```

### 📚 Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | High-level system architecture |
| [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md) | YOLOv8 → RT-DETR migration & benchmarks |
| [docs/violation_engine.md](docs/violation_engine.md) | Violation state machine, IoU matching, tuning |
| [docs/post_processing.md](docs/post_processing.md) | Confidence floors, geometric validation, smoothing |
| [docs/supabase_integration.md](docs/supabase_integration.md) | Persistence layer & RLS policies |
| [docs/detection_benchmark.md](docs/detection_benchmark.md) | Detector throughput on test footage |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Internal contribution workflow |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting |

### ⚖️ License compliance for commercial deployment

The production inference stack is clean for commercial use:

| Component | License |
|---|---|
| RT-DETR architecture (Baidu) | Apache 2.0 |
| Trained model weights | Apache 2.0 |
| `transformers` (HuggingFace) | Apache 2.0 |
| `supervision` (Roboflow) | MIT |
| Construction Site Safety dataset | CC BY 4.0 |
| PyTorch | BSD-style |

YOLOv8 (AGPL-3.0) is kept for benchmarking only and is **not** in any production inference path. See [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md).

### 🔒 KVKK / privacy

- Workers are tracked anonymously by ByteTrack ID — never by biometrics
- No facial recognition is performed
- Snapshots are stored only as evidence for active violations
- Every report export is audit-logged

### 📄 License

**Proprietary — All rights reserved.** See [LICENSE](LICENSE).

---

## 🇹🇷 Türkçe

**İşGözcü**, şantiyelerde Kişisel Koruyucu Donanım (KKD) kullanımını gerçek zamanlı izleyen bir bilgisayarlı görü sistemidir. İşçilerin baret, yelek, maske ve diğer güvenlik ekipmanlarını giyip giymediğini tespit eder ve denetlenebilir ihlal raporları üretir — bunu da KVKK ile tam uyumlu yaparak.

### 🎓 Akademik Danışman

Bu proje, **Dr. Murat Şimşek** akademik danışmanlığında geliştirilmiştir.  
OSTİM Teknik Üniversitesi, Yapay Zeka Mühendisliği Bölümü, Ankara

Metodolojik çerçeve ve bilgisayarlı görü yaklaşımı, Dr. Şimşek'in uzmanlığı doğrultusunda şekillendirilmiştir.

---

### ✨ Öne çıkan özellikler

- **Gerçek zamanlı KKD tespiti** — baret, yelek, maske, eldiven, iş ayakkabısı (10 sınıf)
- **Yasaklı bölge izleme** ve anonim işçi takibi (ByteTrack)
- **KVKK uyumlu** — yüz tanıma yok, biyometrik veri depolanmaz
- **Son işleme hattı** — sınıfa özgü güven eşikleri, geometrik doğrulama, zamansal yumuşatma
- **İhlal motoru** — debouncing + histerezis ile tek-kare gürültüsünü bastırır
- **Saatlik / günlük / haftalık / aylık uyum raporları** + PDF dışa aktarımı
- **Next.js gösterge paneli** — canlı MJPEG yayını + gerçek zamanlı ihlal akışı
- **İsteğe bağlı Supabase kalıcılığı** — olaylar, kanıtlar ve panolar için

### 🛠 Teknoloji yığını

| Katman | Teknoloji |
|---|---|
| Tespit | RT-DETR (Apache 2.0) + HuggingFace `transformers`, Construction Site Safety verisetinde (CC BY 4.0) ince ayarlı |
| Takip | `supervision` üzerinden ByteTrack (MIT) |
| Backend | FastAPI + Uvicorn |
| Veritabanı | Supabase (PostgreSQL) |
| Frontend | Next.js 16 (koyu mavi panel) |
| GPU | NVIDIA CUDA 12.1 (PyTorch 2.5) |

### 🚀 Hızlı başlangıç

```powershell
# 1) Python ortamı oluştur
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Gizli bilgileri ayarla
Copy-Item .env.example .env
# sonra .env dosyasını açıp Supabase bilgilerini doldur

# 3) Canlı simülatörü çalıştır (webcam)
python backend\inference\live_simulator.py --input 0 --detector rtdetr --supabase --stream

# 4) Paneli çalıştır (ayrı bir terminalde)
cd frontend
npm install
npm run dev   # http://localhost:3000
```

> PowerShell betiği engellerse, bir kez şu komutu çalıştırın:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 📁 Proje yapısı

```
isgozcu/
├── backend/
│   ├── inference/   # RT-DETR + ByteTrack + ihlal motoru + son işleme
│   ├── streaming/   # MJPEG HTTP sunucusu (FastAPI)
│   ├── storage/     # Supabase istemcisi
│   ├── api/         # FastAPI uç noktaları
│   ├── models/      # İndirilen model ağırlıkları (.gitignore'da)
│   └── utils/
├── frontend/        # Next.js 16 operatör paneli
├── docs/            # Mimari, kıyaslamalar, tasarım notları
├── notebooks/       # Jupyter denemeleri
├── videos/          # Test görüntüleri (.gitignore'da)
├── data/snapshots/  # İhlal kanıtları (.gitignore'da)
├── data/reports/    # Oturum JSON raporları (.gitignore'da)
├── tests/           # Birim + entegrasyon + yakalama araçları
├── .env.example
├── README.md
└── requirements.txt
```

### 📚 Belgeler

| Belge | Açıklama |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Sistem mimarisi |
| [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md) | YOLOv8 → RT-DETR geçişi ve kıyaslamalar |
| [docs/violation_engine.md](docs/violation_engine.md) | İhlal durum makinesi, IoU eşleştirme, ayar |
| [docs/post_processing.md](docs/post_processing.md) | Güven eşikleri, geometrik doğrulama, yumuşatma |
| [docs/supabase_integration.md](docs/supabase_integration.md) | Kalıcılık katmanı ve RLS politikaları |
| [docs/detection_benchmark.md](docs/detection_benchmark.md) | Test görüntülerinde dedektör performansı |
| [CHANGELOG.md](CHANGELOG.md) | Sürüm geçmişi |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dahili katkı süreci |
| [SECURITY.md](SECURITY.md) | Güvenlik açığı bildirimi |

### ⚖️ Ticari dağıtım için lisans uyumu

Üretim çıkarım yığını ticari kullanım için temizdir:

| Bileşen | Lisans |
|---|---|
| RT-DETR mimarisi (Baidu) | Apache 2.0 |
| Eğitilmiş model ağırlıkları | Apache 2.0 |
| `transformers` (HuggingFace) | Apache 2.0 |
| `supervision` (Roboflow) | MIT |
| Construction Site Safety veriseti | CC BY 4.0 |
| PyTorch | BSD-style |

YOLOv8 (AGPL-3.0) yalnızca kıyaslama amacıyla tutulmaktadır ve **hiçbir** üretim yolunda yer almaz. Ayrıntılar: [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md).

### 🔒 KVKK / gizlilik

- İşçiler ByteTrack ID ile anonim olarak takip edilir — biyometrik yok
- Yüz tanıma yapılmaz
- Anlık görüntüler yalnızca aktif ihlal kanıtı olarak saklanır
- Tüm rapor dışa aktarımları denetim günlüğüne kaydedilir

### 📄 Lisans

**Tescilli — Tüm hakları saklıdır.** Bkz. [LICENSE](LICENSE).

---

<div dir="rtl">

## 🇸🇦 العربية

**İşGözcü** هو نظام رؤية حاسوبية يعمل في الوقت الفعلي لمراقبة مدى التزام عمال مواقع البناء بمعدات الحماية الشخصية (PPE). يكتشف النظام ما إذا كان العمال يرتدون الخوذة، السترة العاكسة، الكمامة، وغيرها من معدات السلامة، ويُنتج تقارير مخالفات قابلة للتدقيق — مع امتثال كامل لقانون حماية البيانات الشخصية التركي (KVKK).

### 🎓 الإشراف الأكاديمي

تم تطوير هذا المشروع تحت الإشراف الأكاديمي للأستاذ  
**د. مراد شيمشيك (Dr. Murat Şimşek)**  
جامعة OSTİM التقنية، قسم هندسة الذكاء الاصطناعي، أنقرة

الإطار المنهجي ومقاربة الرؤية الحاسوبية صيغت بتوجيه خبرة د. شيمشيك.

---

### ✨ المزايا الرئيسية

- **كشف معدات الحماية في الوقت الفعلي** — خوذة، سترة، كمامة، قفازات، أحذية أمان (10 فئات)
- **مراقبة المناطق المحظورة** مع تتبع العمال بشكل مجهول الهوية (ByteTrack)
- **متوافق مع KVKK** — لا تعرف على الوجوه، ولا تخزين لأي بيانات حيوية
- **مرحلة معالجة لاحقة** — حدود ثقة خاصة لكل فئة، تحقق هندسي، تنعيم زمني
- **محرك المخالفات** — مع debouncing وhysteresis لقمع الومضات اللحظية
- **تقارير امتثال ساعية / يومية / أسبوعية / شهرية** + تصدير PDF
- **لوحة تحكم Next.js** — بث مباشر MJPEG + تدفق مخالفات في الزمن الحقيقي
- **حفظ اختياري عبر Supabase** — للأحداث والصور واللوحات

### 🛠 المكدس التقني

| الطبقة | التقنية |
|---|---|
| الكشف | RT-DETR (Apache 2.0) عبر HuggingFace `transformers`، مُدرّب على Construction Site Safety (CC BY 4.0) |
| التتبع | ByteTrack عبر `supervision` (MIT) |
| الواجهة الخلفية | FastAPI + Uvicorn |
| قاعدة البيانات | Supabase (PostgreSQL) |
| الواجهة الأمامية | Next.js 16 (لوحة بألوان زرقاء داكنة) |
| المعالج الرسومي | NVIDIA CUDA 12.1 (PyTorch 2.5) |

### 🚀 البدء السريع

```powershell
# 1) إنشاء بيئة Python
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) إعداد المتغيرات السرية
Copy-Item .env.example .env
# ثم افتح .env واملأ بيانات Supabase

# 3) تشغيل المحاكي المباشر (الكاميرا)
python backend\inference\live_simulator.py --input 0 --detector rtdetr --supabase --stream

# 4) تشغيل لوحة التحكم (في طرفية منفصلة)
cd frontend
npm install
npm run dev   # http://localhost:3000
```

> إذا منع PowerShell التفعيل، نفّذ مرة واحدة:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 📁 هيكل المشروع

```
isgozcu/
├── backend/
│   ├── inference/   # RT-DETR + ByteTrack + محرك المخالفات + المعالجة اللاحقة
│   ├── streaming/   # خادم MJPEG عبر HTTP (FastAPI)
│   ├── storage/     # عميل Supabase
│   ├── api/         # نقاط FastAPI
│   ├── models/      # أوزان النماذج (مستبعد عبر .gitignore)
│   └── utils/
├── frontend/        # لوحة المشغّل بـ Next.js 16
├── docs/            # توثيق المعمارية والقياسات
├── notebooks/       # تجارب Jupyter
├── videos/          # فيديوهات اختبار (مستبعدة)
├── data/snapshots/  # صور إثبات المخالفات (مستبعدة)
├── data/reports/    # تقارير JSON للجلسات (مستبعدة)
├── tests/           # اختبارات وحدة + تكامل
├── .env.example
├── README.md
└── requirements.txt
```

### 📚 التوثيق

| الملف | الوصف |
|---|---|
| [docs/architecture.md](docs/architecture.md) | معمارية النظام |
| [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md) | الانتقال من YOLOv8 إلى RT-DETR وقياسات الأداء |
| [docs/violation_engine.md](docs/violation_engine.md) | آلة حالات المخالفات وضبطها |
| [docs/post_processing.md](docs/post_processing.md) | حدود الثقة والتحقق الهندسي والتنعيم |
| [docs/supabase_integration.md](docs/supabase_integration.md) | طبقة الحفظ وسياسات RLS |
| [docs/detection_benchmark.md](docs/detection_benchmark.md) | أداء الكشف على فيديوهات الاختبار |
| [CHANGELOG.md](CHANGELOG.md) | سجل الإصدارات |
| [CONTRIBUTING.md](CONTRIBUTING.md) | إرشادات المساهمة الداخلية |
| [SECURITY.md](SECURITY.md) | الإبلاغ عن الثغرات الأمنية |

### ⚖️ توافق التراخيص للنشر التجاري

طبقة الاستدلال الإنتاجية نظيفة من الناحية القانونية وصالحة للاستخدام التجاري:

| المكوّن | الترخيص |
|---|---|
| معمارية RT-DETR (Baidu) | Apache 2.0 |
| أوزان النموذج المُدرَّبة | Apache 2.0 |
| `transformers` (HuggingFace) | Apache 2.0 |
| `supervision` (Roboflow) | MIT |
| قاعدة بيانات Construction Site Safety | CC BY 4.0 |
| PyTorch | BSD-style |

أمّا YOLOv8 (AGPL-3.0) فيبقى لأغراض المقارنة فقط، **ولا** يُستخدم في أي مسار إنتاج. التفاصيل في [docs/migration_yolo_to_rtdetr.md](docs/migration_yolo_to_rtdetr.md).

### 🔒 KVKK والخصوصية

- يُتتبَّع العمال بشكل مجهول الهوية عبر معرّفات ByteTrack — لا توجد بيانات حيوية
- لا يُجرى أي تعرف على الوجوه
- تُحفظ صور الإثبات فقط للمخالفات النشطة
- جميع عمليات تصدير التقارير مسجّلة للتدقيق

### 📄 الترخيص

**ملكية خاصة — جميع الحقوق محفوظة.** انظر [LICENSE](LICENSE).

</div>

---

<div align="center">

### 🚧 Development status · Geliştirme durumu · حالة التطوير

| Phase | Status | Highlights |
|---|---|---|
| Phase 1 — Detection MVP | ✅ Done | YOLOv8 PPE detection on test footage |
| Phase 2A — RT-DETR migration | ✅ Done | License-clean inference stack, v1 → v5 checkpoints |
| Phase 2A.7 — Post-processing | ✅ Done | Confidence floors, geometry, vest smoothing |
| Phase 2B — Violation engine | ✅ Done | Debouncing, hysteresis, snapshots, JSON reports |
| Phase 2C — Supabase persistence | ✅ Done | Events + snapshots auto-upload |
| Phase 3.1–3.5 — Dashboard | ✅ Done | Next.js 16, live MJPEG, polling updates |
| Phase 4 — Auth + multi-site | 🟡 Planned | Operator login, per-site dashboards |
| Phase 5 — Mobile alerts | ⚪ Planned | Push notifications for active violations |

</div>
