# Security Policy · Güvenlik Politikası · سياسة الأمن

**🇬🇧 [English](#-english)** · **🇹🇷 [Türkçe](#-türkçe)** · **🇸🇦 [العربية](#-العربية)**

---

## 🇬🇧 English

### Supported versions

İşGözcü is currently **pre-release** (Phase 3.5). Only the current `master` branch receives security updates.

| Version | Supported |
|---|---|
| `master` (current) | ✅ |
| Tagged pre-releases | ⚠️ Best effort only |
| Anything older | ❌ |

### Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, please send a private report through one of these channels:

1. Open a [GitHub Security Advisory](../../security/advisories/new) for this repository (preferred).
2. Contact the project maintainer through the official channels listed in the repository profile.

Include in your report:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof-of-concept
- The component affected (detector, violation engine, Supabase layer, frontend, etc.)
- Your suggested mitigation, if any
- Whether you wish to be credited publicly when the fix is published

### What to expect

| Timeframe | Action |
|---|---|
| Within 72 hours | Acknowledgement of your report |
| Within 14 days | Initial assessment + severity classification |
| Within 90 days | Patch release for High/Critical issues (best effort for others) |

We follow a **coordinated disclosure** model. Please give us reasonable time to investigate and patch before any public disclosure.

### In scope

- Authentication/authorization issues in the FastAPI surface
- SQL injection or RLS bypass in the Supabase layer
- Path traversal or arbitrary file write in snapshot/report writers
- Exposed secrets (`.env`, service-role keys) accidentally committed
- Frontend XSS, CSRF, or auth bypass
- Privacy regressions that could leak identifiable worker data (KVKK-relevant)

### Out of scope

- Vulnerabilities in third-party dependencies — please report those to the upstream project (we will then update our pin).
- Issues that require physical access to the camera or the simulator host.
- Model evasion (e.g. spoofing a helmet detection with a sticker) — these are model-quality issues, not security vulnerabilities.
- Denial of service from sending pathological input video — interesting, but not a security boundary we currently defend.

### KVKK / personal-data privacy

İşGözcü is designed to be KVKK-compliant. If you discover a way the system could leak personal data — including but not limited to:

- Facial features that survive cropping or could be re-identified
- Worker tracker IDs that could be correlated back to a real identity
- Snapshots exposed via predictable URLs
- Database rows readable by unauthorized roles via misconfigured RLS

please report it with the **highest priority** through the channels above. We treat privacy regressions as critical regardless of CVSS score.

### Hall of fame

Researchers who responsibly disclose valid vulnerabilities will (with their consent) be listed in our security acknowledgements once the fix is released.

---

## 🇹🇷 Türkçe

### Desteklenen sürümler

İşGözcü şu anda **ön sürüm** aşamasındadır (Faz 3.5). Yalnızca mevcut `master` dalı güvenlik güncellemeleri alır.

| Sürüm | Destek |
|---|---|
| `master` (mevcut) | ✅ |
| Etiketli ön sürümler | ⚠️ Sadece en iyi çaba |
| Daha eski sürümler | ❌ |

### Bir güvenlik açığı bildirme

**Güvenlik açıkları için lütfen herkese açık bir GitHub issue açmayın.**

Bunun yerine, lütfen aşağıdaki kanallardan biri aracılığıyla özel bir rapor gönderin:

1. Bu depo için bir [GitHub Security Advisory](../../security/advisories/new) açın (tercih edilen).
2. Depo profilinde listelenen resmi kanallar aracılığıyla proje bakımcısıyla iletişime geçin.

Raporunuza şunları ekleyin:

- Güvenlik açığının ve potansiyel etkisinin açıklaması
- Yeniden üretim adımları veya bir kavram kanıtı
- Etkilenen bileşen (dedektör, ihlal motoru, Supabase katmanı, frontend vb.)
- Varsa önerdiğiniz hafifletme
- Düzeltme yayımlandığında alenen kredi almak isteyip istemediğiniz

### Beklentiler

| Süre | Eylem |
|---|---|
| 72 saat içinde | Raporunuzun alındığına dair bildirim |
| 14 gün içinde | İlk değerlendirme + ciddiyet sınıflandırması |
| 90 gün içinde | Yüksek/Kritik sorunlar için yama (diğerleri için en iyi çaba) |

**Koordineli ifşa** modelini takip ediyoruz. Lütfen kamu ifşasından önce araştırıp yama yapmamız için bize makul zaman tanıyın.

### Kapsam dahilinde

- FastAPI yüzeyinde kimlik doğrulama/yetkilendirme sorunları
- Supabase katmanında SQL enjeksiyonu veya RLS atlatma
- Anlık görüntü/rapor yazıcılarında yol gezme veya rastgele dosya yazma
- Yanlışlıkla commitlenmiş açık sırlar (`.env`, servis-rol anahtarları)
- Frontend XSS, CSRF veya kimlik doğrulama atlatma
- Tanımlanabilir işçi verilerini sızdırabilecek gizlilik regresyonları (KVKK ilgili)

### Kapsam dışı

- Üçüncü taraf bağımlılıklardaki güvenlik açıkları — lütfen bunları yukarı akış projesine bildirin (sonra pin'imizi güncelleriz).
- Kameraya veya simülatör ana bilgisayarına fiziksel erişim gerektiren sorunlar.
- Model atlatma (örn. bir çıkartmayla baret tespitini kandırma) — bunlar model kalitesi sorunlarıdır, güvenlik açıkları değil.
- Patolojik giriş videosu göndererek hizmet reddi — ilginç ama şu anda savunduğumuz bir güvenlik sınırı değil.

### KVKK / kişisel veri gizliliği

İşGözcü KVKK uyumlu olacak şekilde tasarlanmıştır. Sistemin kişisel verileri sızdırabileceği bir yol keşfederseniz — şunları içeren ancak bunlarla sınırlı olmayan:

- Kırpmadan sonra hayatta kalan veya yeniden tanımlanabilen yüz özellikleri
- Gerçek bir kimlikle korelasyon kurulabilecek işçi takip kimlikleri
- Tahmin edilebilir URL'ler aracılığıyla açığa çıkan anlık görüntüler
- Yanlış yapılandırılmış RLS aracılığıyla yetkisiz roller tarafından okunabilen veritabanı satırları

lütfen yukarıdaki kanallar aracılığıyla **en yüksek öncelikle** bildirin. CVSS skoru ne olursa olsun gizlilik regresyonlarını kritik olarak ele alırız.

### Onur listesi

Geçerli güvenlik açıklarını sorumlu bir şekilde ifşa eden araştırmacılar (rızalarıyla), düzeltme yayımlandıktan sonra güvenlik teşekkürlerimizde listelenecektir.

---

<div dir="rtl">

## 🇸🇦 العربية

### الإصدارات المدعومة

İşGözcü حاليًا في **مرحلة ما قبل الإصدار** (المرحلة 3.5). فقط فرع `master` الحالي يتلقى تحديثات أمنية.

| الإصدار | الدعم |
|---|---|
| `master` (الحالي) | ✅ |
| إصدارات ما قبل الإصدار الموسومة | ⚠️ بأفضل جهد |
| أي إصدار أقدم | ❌ |

### الإبلاغ عن ثغرة أمنية

**يُرجى عدم فتح قضية GitHub علنية للثغرات الأمنية.**

بدلاً من ذلك، يُرجى إرسال بلاغ خاص عبر إحدى هذه القنوات:

1. افتح [GitHub Security Advisory](../../security/advisories/new) لهذا المستودع (مُفضّل).
2. تواصل مع مشرف المشروع عبر القنوات الرسمية المدرجة في ملف المستودع.

ضمّن في بلاغك:

- وصفًا للثغرة وأثرها المحتمل
- خطوات إعادة الإنتاج، أو إثبات للمفهوم (PoC)
- المكوّن المتأثر (الكاشف، محرك المخالفات، طبقة Supabase، الواجهة الأمامية، إلخ)
- اقتراحك للتخفيف، إن وُجد
- إن كنت ترغب بأن يُنسب لك الفضل علنيًا عند نشر الإصلاح

### ما يمكن توقعه

| الإطار الزمني | الإجراء |
|---|---|
| خلال 72 ساعة | الإقرار باستلام بلاغك |
| خلال 14 يومًا | تقييم أولي + تصنيف الخطورة |
| خلال 90 يومًا | إصدار رقعة للمشكلات الحرجة/العالية (بأفضل جهد لغيرها) |

نتّبع نموذج **الإفصاح المُنسَّق**. يُرجى منحنا وقتًا معقولًا للتحقيق والترقيع قبل أي إفصاح علني.

### في النطاق

- مشكلات المصادقة/التفويض في طبقة FastAPI
- حقن SQL أو تجاوز RLS في طبقة Supabase
- اجتياز المسارات أو كتابة ملفات عشوائية في كاتب اللقطات/التقارير
- أسرار مكشوفة (`.env`، مفاتيح service-role) عُمدت بالخطأ
- XSS أو CSRF أو تجاوز المصادقة في الواجهة الأمامية
- انحدارات الخصوصية التي قد تكشف بيانات عمال يمكن التعرف عليها (متعلقة بـ KVKK)

### خارج النطاق

- ثغرات في التبعيات الخارجية — يُرجى الإبلاغ عنها للمشروع الأصلي (سنقوم بتحديث الإصدار المُثبّت لاحقًا).
- مشكلات تتطلب وصولًا فعليًا للكاميرا أو لمضيف المحاكي.
- تجاوز النموذج (مثلًا خداع كشف الخوذة بملصق) — هذه مشكلات جودة نموذج، لا ثغرات أمنية.
- حجب الخدمة بإرسال فيديو إدخال مرضي — مثير للاهتمام، لكنه ليس حدًا أمنيًا نُدافع عنه حاليًا.

### KVKK / خصوصية البيانات الشخصية

صُمّم İşGözcü ليكون متوافقًا مع KVKK. إن اكتشفت طريقة قد يسرّب بها النظام بيانات شخصية — تشمل على سبيل المثال لا الحصر:

- ملامح وجه تبقى بعد القصّ أو يمكن إعادة التعرف عليها
- معرّفات تتبع العمال التي يمكن ربطها بهوية حقيقية
- لقطات مكشوفة عبر روابط يمكن تخمينها
- صفوف قاعدة بيانات قابلة للقراءة من قبل أدوار غير مخوَّلة بسبب RLS مُهيَّأ بشكل خاطئ

يُرجى الإبلاغ عنها بـ **الأولوية القصوى** عبر القنوات أعلاه. نتعامل مع انحدارات الخصوصية بوصفها حرجة بصرف النظر عن درجة CVSS.

### قاعة المشاهير

الباحثون الذين يُفصحون عن ثغرات صحيحة بمسؤولية سيُدرَجون (بموافقتهم) في تنويهاتنا الأمنية بمجرد إصدار الإصلاح.

</div>
