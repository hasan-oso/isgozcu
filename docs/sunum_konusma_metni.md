# İşGözcü — Sunum Konuşma Metni (Prova için)

7 ekranlık canlı tanıtım. Her ekran için **ne söyleyeceğin** ve **kısa bir sunum ipucu** var.
Ekrana yazılanı **okuma** — ekran başlık, sen hikâyeyi anlat. **Kahraman, 5. ekrandaki canlı demo.**

Geçiş: `←` `→` ok tuşları, ya da alttaki "Geri / İleri" düğmeleri, ya da noktalara tıkla.

---

## Ekran 1 — AÇILIŞ
**Ne söylemeli:**
"Merhaba, hoş geldiniz. Daha önce size İşGözcü'yü bir sunumda tanıtmıştık — şantiyede baret ve yelek uyumunu izleyen güvenlik sistemi. Bugün farklı bir şey yapacağız: sistemi size **canlı** göstereceğim, slaytta değil, gerçek zamanlı çalışırken. Ben Hasan Oso, projenin uygulayıcı mühendisiyim; yanımda akademik danışmanım Dr. Murat Şimşek."

**İpucu:** Yavaş ve kendinden emin başla. "Canlı" kelimesini vurgula — bütün sunumun vaadi bu. Dr. Şimşek'i tanıtırken ona dön/işaret et (akademik güven).

---

## Ekran 2 — SİSTEM NE YAPAR
**Ne söylemeli:**
"Kısaca İşGözcü ne yapar: işçilerin koruyucu donanımını — bareti ve yeleği — gerçek zamanlı izler ve her an 'bu işçi kurallara uygun mu?' sorusunu yanıtlar. Üç şey yapar: **görür**, yani otomatik tespit eder; **değerlendirir**, yani gerçek ihlali yanlış alarmdan ayırır; ve **kanıtlar**, yani her ihlali görüntüyle belgeler."

**İpucu:** Üç kelimeyi (Görür – Değerlendirir – Kanıtlar) tek tek, ara vererek söyle. "Değerlendirir" kısmı önemli — müşteri yanlış alarmdan korkar, burada güven ver.

---

## Ekran 3 — NASIL ÇALIŞIR
**Ne söylemeli:**
"Teknik olarak üç aşamada çalışır. Birincisi algılama: RF-DETR adında, 2026'nın en güncel yapay zekâ mimarisini kullanıyoruz — bu, yüksek doğruluk ve düşük yanlış alarm demek. İkincisi takip: OC-SORT her işçiyi sahnede sürekli izler. Üçüncüsü değerlendirme: üç katmanlı bir doğrulama — sadece 5 saniyeden uzun süren **gerçek** ihlalleri bildirir, bir anlık paraziti değil."

**İpucu:** Teknik isimleri söyle ama hemen ardından "yani..." diyerek değere çevir. Mühendis olmayan bir dinleyici teknik ismi unutur, değeri hatırlar. 5 saniye detayını vurgula — demoda bunu görecekler.

---

## Ekran 4 — NEDEN GÜVENİLİR
**Ne söylemeli:**
"Peki neden bu sisteme güvenebilirsiniz? Dört nedeni var. Son nesil bir mimari kullanıyor — eski teknoloji değil. Apache 2.0 lisanslı — yani tam ticari kullanım hakkınız var, hiçbir yasal kısıt yok. Gerçek şantiye verisiyle eğitildi, laboratuvar koşullarında değil. Ve ölçülmüş doğruluğu yüzde seksen iki — bu rakam doğrulama setinde gerçekten ölçüldü, tahmin değil."

**İpucu:** Apache 2.0 ve "yasal kısıt yok" kısmında dur — ticari müşteri için bu büyük bir rahatlama. 0.82'yi "ölçülmüş, uydurma değil" diye dürüstçe sun; güven bundan gelir.

---

## Ekran 5 — CANLI DEMO  ★ (Kahraman ekran)
Bu ekranda canlı yayın ve sağda gerçek zamanlı bilgiler (kişi sayısı, durum, ihlal, FPS, yayın) görünür. Aşağıdaki senaryoyu **yavaş** uygula; her hareketten sonra ekrandaki sayıların değiştiğini göster.

**Açılış sözü:**
"Şimdi en önemli kısım. Bu gördüğünüz **canlı** yayın, şu an çalışıyor. Sağdaki bütün rakamlar gerçek — her saniye güncelleniyor."

**Adım adım senaryo:**

1. **Baret + yelek takılı şekilde kameranın önüne geç.**
   Söyle: "Üzerimde baret ve yelek var. Bakın — sistem beni tespit etti: 'Tespit edilen kişi: 1'. Ve durum **Uyumlu**, yeşil."
   *(Ekranda 'Durum: Uyumlu' yeşil, 'Aktif ihlaller: 0'.)*

2. **Bareti yavaşça çıkar.**
   Söyle: "Şimdi baretimi çıkarıyorum. Dikkat edin — sistem **hemen** alarm vermiyor."
   *(Bekle. ~5 saniye.)*
   "Çünkü gerçekten bir ihlal mi, yoksa bir anlık karışıklık mı diye 5 saniye bekliyor. Bu, yanlış alarmı önler."

3. **Beş saniye dolunca durum kırmızıya döner.**
   Söyle: "Ve işte — şimdi durum **İhlal**, kırmızı. 'Aktif ihlaller: 1'. Sistem bunu kaydetti, görüntüsünü sakladı; bu artık kanıt."

4. **Bareti geri tak.**
   Söyle: "Şimdi baretimi geri takıyorum."
   *(Birkaç saniye bekle.)*
   "Ve sistem uyumu yeniden onaylıyor — tekrar yeşil. İşçi tehlikeden çıktı, sistem bunu da gördü."

**İpucu:** Acele etme — 5 saniyelik bekleme **özelliktir**, sus ve ekranı izlet; sessizlik etkili olur. Hareketleri abartılı/net yap (kamera görsün). Eğer ışık/açı yüzünden tespit gecikirse panik yapma: "Sistem temkinli, gerçek ihlalde emin olmak ister" de ve tekrar dene.

---

## Ekran 6 — DEĞER & KVKK
**Ne söylemeli:**
"Bu sistem sizi nasıl korur? Üç açıdan. Yasal uyum: KVKK'ya uygun — takip **anonim**, yüz tanıma **yok**; kimliği değil, güvenliği izliyoruz. Maliyet: önlenen tek bir ciddi kaza bile bu sistemin maliyetini fazlasıyla çıkarır. Ve denetim: 7/24 kesintisiz — yorulmayan, dikkati dağılmayan bir gözcü."

**İpucu:** "Yüz tanıma yok" cümlesini net söyle — hem yasal hem etik bir endişeyi peşinen kapatır. Maliyet argümanını karar vericiye bakarak söyle.

---

## Ekran 7 — KAPANIŞ
**Ne söylemeli:**
"Toparlarsak: İşGözcü bir vaat değil — az önce canlı gördüğünüz, çalışan bir sistem. Bundan sonrası için yol haritamız da net: çok kameralı takip ve yeni güvenlik sınıfları — eldiven, gözlük gibi. İlginiz için teşekkür ederim; sorularınızı memnuniyetle alırım."

**İpucu:** Yol haritasını kısa tut — bu aynı zamanda "sistemin sınırları ne?" sorusunun cevabıdır (henüz tek kamera, 3 sınıf). Güçlü bir "Teşekkürler" ile bitir, soruları davet et.

---

## Genel İpuçları
- **Yavaş konuş.** Acele en sık yapılan hata. Cümleler arasında nefes al.
- **Ekrandan okuma.** Ekran senin notun değil; sen hikâyeyi anlat, ekran başlığı destekler.
- **Demo her şeydir.** Slaytlarda hızlı geç, 5. ekranda zaman harca. Müşteri "çalışıyor mu?" sorusunun cevabını orada görür.
- **5 saniyelik bekleme bir kusur değil, özelliktir** — sessizce izlet, sonra "yanlış alarmı böyle önlüyoruz" de.
- **Sınır sorusuna hazır ol.** "Aynı kıyafetli, çok kalabalık sahnelerde tek tek kişi takibi zorlaşabilir; yol haritamızda çok kamera ve iyileştirmeler var" — dürüst ve hazırlıklı cevap güven verir.
- **Dr. Şimşek = bilimsel güven.** Akademik bir soru gelirse ona yönlendirebilirsin.
- **Teknik isim + 'yani' + değer.** Her teknik terimi hemen faydaya çevir.
- **Yedek plan:** Canlı kamera takılırsa, hazırda bir video kaynağıyla (`--source <video>`) aynı demoyu gösterebilirsin; ya da kameranın ışığını/açısını önceden ayarla.
