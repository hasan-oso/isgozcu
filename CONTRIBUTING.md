# Contributing · Katkıda Bulunma · المساهمة

**🇬🇧 [English](#-english)** · **🇹🇷 [Türkçe](#-türkçe)** · **🇸🇦 [العربية](#-العربية)**

> ⚠️ **İşGözcü is proprietary software currently in active development.** External contributions are reviewed on a case-by-case basis and may not be accepted. The guidelines below apply primarily to internal collaborators and authorized partners.

---

## 🇬🇧 English

Thank you for your interest in İşGözcü. This document describes our internal workflow for source-code contributions, bug reports, and feature proposals.

### Before you start

1. **Read the [Code of Conduct](CODE_OF_CONDUCT.md)** — it applies to every interaction.
2. **Check the [LICENSE](LICENSE)** — by contributing, you confirm you have the right to grant the copyright holder a perpetual, royalty-free license over your contribution.
3. **Search existing [Issues](../../issues)** — your topic may already be tracked.
4. **For non-trivial changes, open an issue first** so we can align on scope and approach before you write code.

### Development environment

```powershell
git clone <repository-url>
cd isgozcu

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# fill in your Supabase keys

cd frontend
npm install
```

Pre-flight checks before opening a Pull Request:

```powershell
# Python tests
python tests\test_iou_matcher.py
python tests\test_violation_engine.py
python tests\test_post_processor.py

# Frontend build
cd frontend
npm run build
```

### Branching model

- `main` — protected, production-equivalent. No direct pushes.
- `master` — current integration branch for ongoing phases.
- `phase-N.M-<short-slug>` — feature branches; squash-merge into `master`.
- `hotfix/<short-slug>` — urgent fixes; merge to both `master` and `main`.

### Commit message style

We follow the convention already visible in the git log:

```
Phase 2A.7: post-processing pipeline + violation engine tuning
^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
phase tag   imperative summary (~70 chars max)

<optional body explaining the why, in present tense>
```

- Use imperative mood ("add", "fix", "refactor"), not past tense.
- Reference the phase number when applicable.
- One logical change per commit; squash noise locally before pushing.

### Pull Request checklist

- [ ] Branch name follows the convention
- [ ] Tests pass locally (Python + frontend build)
- [ ] No secrets, no `.env`, no large binaries committed
- [ ] Updated [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]`
- [ ] Updated relevant docs in `docs/`
- [ ] Turkish UI strings remain Turkish; code identifiers remain English
- [ ] No regressions in `tests/test_violation_engine.py` or `tests/test_post_processor.py`

### Reporting bugs

Use the **Bug report** issue template. Include:

- Steps to reproduce
- Expected vs. observed behavior
- Console output (with `--debug` if available)
- Hardware (GPU model, VRAM), Python version, OS
- Detector version (`--model-version v3` etc.)

### Proposing features

Use the **Feature request** issue template. Include the user problem, the proposed solution, and the smallest possible scope ("minimum lovable feature").

### Security

**Do not open public issues for security vulnerabilities.** Follow the disclosure process in [SECURITY.md](SECURITY.md).

---

## 🇹🇷 Türkçe

İşGözcü'ye gösterdiğiniz ilgi için teşekkür ederiz. Bu belge kaynak kodu katkıları, hata bildirimleri ve özellik önerileri için iç akışımızı açıklar.

### Başlamadan önce

1. **[Davranış Kuralları](CODE_OF_CONDUCT.md)**'nı okuyun — her etkileşim için geçerlidir.
2. **[LİSANS](LICENSE)**'ı kontrol edin — katkıda bulunarak, telif hakkı sahibine katkınız üzerinde süresiz ve telifsiz bir lisans verme hakkına sahip olduğunuzu onaylarsınız.
3. **Mevcut [Issue'ları](../../issues)** arayın — konunuz zaten takip ediliyor olabilir.
4. **Önemli değişiklikler için önce issue açın** — kod yazmadan kapsam ve yaklaşım üzerinde anlaşalım.

### Geliştirme ortamı

```powershell
git clone <repository-url>
cd isgozcu

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# Supabase anahtarlarınızı doldurun

cd frontend
npm install
```

Pull Request açmadan önce kontroller:

```powershell
# Python testleri
python tests\test_iou_matcher.py
python tests\test_violation_engine.py
python tests\test_post_processor.py

# Frontend derlemesi
cd frontend
npm run build
```

### Dallanma modeli

- `main` — korumalı, üretim eşdeğeri. Doğrudan push yok.
- `master` — devam eden fazlar için entegrasyon dalı.
- `phase-N.M-<kısa-slug>` — özellik dalları; `master`'a squash-merge.
- `hotfix/<kısa-slug>` — acil düzeltmeler; hem `master`'a hem `main`'e merge.

### Commit mesaj biçimi

Git geçmişinde halihazırda görünür olan kuralı izleriz:

```
Phase 2A.7: post-processing pipeline + violation engine tuning
^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
faz etiketi  emir kipinde özet (~70 karakter)

<isteğe bağlı: nedeni şimdiki zamanda açıklayan gövde>
```

- Emir kipi ("ekle", "düzelt", "yeniden düzenle") — geçmiş zaman değil.
- Mümkünse faz numarasını referans alın.
- Her commit tek bir mantıksal değişiklik; itme öncesi yerelde squash yapın.

### Pull Request kontrol listesi

- [ ] Dal adı kurala uyuyor
- [ ] Testler yerelde geçiyor (Python + frontend build)
- [ ] Sır, `.env` veya büyük ikili dosya commitlenmemiş
- [ ] [CHANGELOG.md](CHANGELOG.md) `## [Unreleased]` altında güncellendi
- [ ] `docs/` altındaki ilgili belgeler güncellendi
- [ ] Türkçe UI metinleri Türkçe kalıyor; kod tanımlayıcıları İngilizce kalıyor
- [ ] `tests/test_violation_engine.py` ve `tests/test_post_processor.py`'de regresyon yok

### Hata bildirimi

**Bug report** şablonunu kullanın. Şunları ekleyin:

- Adım adım yeniden üretim
- Beklenen vs. gözlemlenen davranış
- Konsol çıktısı (mevcutsa `--debug` ile)
- Donanım (GPU modeli, VRAM), Python sürümü, işletim sistemi
- Dedektör sürümü (`--model-version v3` vb.)

### Özellik önerme

**Feature request** şablonunu kullanın. Kullanıcı problemini, önerilen çözümü ve mümkün olan en küçük kapsamı belirtin ("minimum sevilebilir özellik").

### Güvenlik

**Güvenlik açıkları için herkese açık issue açmayın.** [SECURITY.md](SECURITY.md)'deki açıklama sürecini takip edin.

---

<div dir="rtl">

## 🇸🇦 العربية

شكرًا لاهتمامك بـ İşGözcü. تصف هذه الوثيقة سير العمل الداخلي للمساهمات في الشيفرة، والإبلاغ عن الأخطاء، واقتراح الميزات.

### قبل أن تبدأ

1. **اقرأ [قواعد السلوك](CODE_OF_CONDUCT.md)** — تنطبق على كل تفاعل.
2. **راجع [الترخيص](LICENSE)** — بمساهمتك، تؤكد أن لديك الحق في منح صاحب حقوق النشر ترخيصًا دائمًا ومجانيًا على مساهمتك.
3. **ابحث في [القضايا الموجودة](../../issues)** — قد يكون موضوعك مُتابعًا بالفعل.
4. **للتغييرات غير البديهية، افتح قضية أولًا** حتى نتفق على النطاق والمقاربة قبل كتابة الكود.

### بيئة التطوير

```powershell
git clone <repository-url>
cd isgozcu

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# املأ مفاتيح Supabase

cd frontend
npm install
```

تحقق قبل فتح Pull Request:

```powershell
# اختبارات Python
python tests\test_iou_matcher.py
python tests\test_violation_engine.py
python tests\test_post_processor.py

# بناء الواجهة الأمامية
cd frontend
npm run build
```

### نموذج التفرّع

- `main` — محمي، يعادل الإنتاج. لا push مباشر.
- `master` — فرع التكامل للمراحل الجارية.
- `phase-N.M-<وصف-قصير>` — فروع الميزات؛ تُدمج بـ squash إلى `master`.
- `hotfix/<وصف-قصير>` — إصلاحات عاجلة؛ تُدمج إلى `master` و`main`.

### أسلوب رسائل Commit

نتّبع الأسلوب الظاهر في سجل git:

```
Phase 2A.7: post-processing pipeline + violation engine tuning
^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
وسم المرحلة  ملخص بصيغة الأمر (~70 حرفًا)

<اختياري: متن يشرح السبب بصيغة المضارع>
```

- استخدم صيغة الأمر ("أضف"، "أصلِح"، "أعد الهيكلة")، لا صيغة الماضي.
- أشر إلى رقم المرحلة عند الإمكان.
- تغيير منطقي واحد لكل commit؛ نظّف الضوضاء محليًا قبل الـpush.

### قائمة فحص Pull Request

- [ ] اسم الفرع يتبع الاصطلاح
- [ ] الاختبارات تنجح محليًا (Python + بناء الواجهة)
- [ ] لا أسرار، لا `.env`، لا ملفات ثنائية كبيرة في الـcommit
- [ ] تم تحديث [CHANGELOG.md](CHANGELOG.md) تحت `## [Unreleased]`
- [ ] تم تحديث الوثائق ذات الصلة في `docs/`
- [ ] نصوص الواجهة التركية تبقى تركية؛ معرفات الكود تبقى إنكليزية
- [ ] لا انحدار في `tests/test_violation_engine.py` أو `tests/test_post_processor.py`

### الإبلاغ عن الأخطاء

استخدم قالب **Bug report**. أضِف:

- خطوات إعادة الإنتاج
- السلوك المتوقع مقابل المُلاحظ
- مخرجات الكونسول (مع `--debug` إن أمكن)
- العتاد (نوع GPU، حجم VRAM)، إصدار Python، نظام التشغيل
- إصدار الكاشف (`--model-version v3` إلخ)

### اقتراح ميزة

استخدم قالب **Feature request**. اذكر مشكلة المستخدم، الحل المقترح، وأصغر نطاق ممكن ("الحد الأدنى للميزة المحبوبة").

### الأمن

**لا تفتح قضايا علنية للثغرات الأمنية.** اتبع عملية الإفصاح الموضحة في [SECURITY.md](SECURITY.md).

</div>
