# İşGözcü — Sunum Konuşma Metni ve Çalışma Kılavuzu

> Ticari müşteriye **canlı** tanıtım için kapsamlı prova kılavuzu. Ekrandakini **okuma** — sen hikâyeyi anlat, ekran başlık. **Kahraman, 5. ekrandaki canlı demo.**

---

## Genel Bakış

**Akış (7 ekran):** Açılış → Sistem Ne Yapar → Nasıl Çalışır → Neden Güvenilir → **CANLI DEMO** → Değer & KVKK → Kapanış.

**Süre:** Toplam ~10–12 dakika. Slaytlarda hızlı ilerle (her biri ~30–60 sn); **5. ekranda (demo) 3–4 dakika** harca — asıl ikna orada.

**Gezinme (sahnede, görünür düğme yok):**
- Ekranın **sağ yarısına** tıkla → ileri · **sol yarısına** tıkla → geri.
- Ya da klavyede **→** ileri, **←** geri.
- Alttaki küçük noktalar nerede olduğunu gösterir (istersen tıklayarak atlayabilirsin).
- 5. ekranda **canlı yayının/panelin üstüne tıklamak slaytı çevirmez** — rahatça işaret edebilirsin; ilerlemek için **→** tuşunu kullan.

---

## Ekran 1 — AÇILIŞ
**Ne söylemeli:**
"Merhaba, hoş geldiniz. Daha önce size İşGözcü'yü tanıtmıştık — şantiyede baret ve yelek uyumunu izleyen güvenlik sistemi. O zaman bir fikirdi; bugün size onu **canlı**, çalışırken göstereceğim. Ben Hasan Oso, projenin yapay zekâ mühendisiyim; yanımda akademik danışmanım Dr. Murat Şimşek."

**Vurgu:** Bu artık bir vaat değil, **çalışan** bir sistem — ve bugün canlı göreceksiniz.

**İpucu:** Yavaş ve sıcak başla. "Canlı" kelimesini vurgula. Dr. Şimşek'i tanıtırken ona dön — akademik danışman, sisteme bilimsel güven katar.

---

## Ekran 2 — SİSTEM NE YAPAR
**Ne söylemeli:**
"Kısaca İşGözcü ne yapar: işçilerin koruyucu donanımını — bareti ve yeleği — gerçek zamanlı izler ve her an 'bu işçi kurallara uygun mu?' sorusunu yanıtlar. Üç şey yapar. **Görür:** her işçiyi ve donanımı otomatik tespit eder. **Değerlendirir:** gerçek ihlali geçici bir karışıklıktan ayırır — yani boş yere alarm vermez. Ve **kanıtlar:** her ihlali görüntülü kayıtla belgeler, sonra inkâr edilemez."

**Vurgu:** Sistem sadece "görmüyor" — **değerlendiriyor ve kanıtlıyor.** Yanlış alarm vermemesi en az tespit kadar önemli.

**İpucu:** Görür–Değerlendirir–Kanıtlar üçlüsünü ara vererek, tek tek söyle. "Değerlendirir" kısmında dur; müşterinin en büyük korkusu yanlış alarmdır.

---

## Ekran 3 — NASIL ÇALIŞIR
**Ne söylemeli:**
"Teknik olarak üç aşamada çalışır, ama önemli olan ne işe yaradığı. Birincisi algılama: RF-DETR adında, 2026'nın en güncel yapay zekâ tespit mimarisini kullanıyoruz — bu yüksek doğruluk ve düşük yanlış alarm demek. İkincisi takip: OC-SORT, her işçiyi sahnede sürekli izler. Üçüncüsü değerlendirme: üç katmanlı bir doğrulamadan geçirir ve sadece **5 saniyeden uzun süren gerçek ihlalleri** bildirir — bir anlık görüntü parazitini değil."

**Vurgu:** En güncel teknoloji + "5 saniye kuralı" = doğru ve sakin bir sistem. Az sonra demoda bu 5 saniyeyi göreceksiniz.

**İpucu:** Teknik ismi söyle, hemen "yani..." diyerek faydaya çevir. 5 saniye detayını özellikle vurgula — demonun can alıcı noktası bu.

---

## Ekran 4 — NEDEN GÜVENİLİR
**Ne söylemeli:**
"Peki neden güvenebilirsiniz? Dört nedeni var. Bir: son nesil bir mimari kullanıyor, eski teknoloji değil. İki: Apache 2.0 lisanslı — yani tam ticari kullanım hakkınız var, hiçbir yasal kısıt yok. Üç: gerçek şantiye verisiyle eğitildi, laboratuvar koşullarında değil. Dört: ölçülmüş doğruluğu mAP olarak yüzde seksen iki — ve bu rakam doğrulama setinde gerçekten ölçüldü, tahmin değil."

**Vurgu:** Apache 2.0 (yasal kısıt yok) + 0.82 (ölçülmüş, dürüst rakam). Ticari müşteri için "yasal risk yok" çok değerli.

**İpucu:** Apache 2.0'da dur — bu büyük bir rahatlama. 0.82'yi "uydurmadık, ölçtük" diye dürüstçe sun; güven dürüstlükten gelir.

---

## Ekran 5 — CANLI DEMO ★ (Kahraman ekran)
Ekranda **canlı yayın** (büyük, ortada) ve sağda **gerçek zamanlı bilgiler** var: tespit edilen kişi, durum (Uyumlu/İhlal), aktif ihlal, FPS, yayın durumu. Hepsi her saniye güncellenir. Aşağıdaki senaryoyu **yavaş** uygula.

**Açılış sözü:**
"Şimdi en önemli kısım. Bu gördüğünüz **canlı** yayın, şu an çalışıyor. Sağdaki bütün rakamlar gerçek — her saniye güncelleniyor. İzleyelim."

**Adım adım senaryo (jest + söz birlikte):**

**1) Baret + yelek takılı kameranın önüne geç.**
> Söyle: "Üzerimde baret ve yelek var. Bakın — sistem beni tespit etti: 'Tespit edilen kişi: 1'. Durum **Uyumlu**, yeşil. Her şey kuralına uygun."
> *(Sağ paneldeki 'Durum: Uyumlu'yu işaret et.)*

**2) Bareti yavaşça çıkar.**
> Söyle: "Şimdi baretimi çıkarıyorum. Dikkat edin — sistem **hemen** alarm vermiyor."
> *(Sus, bekle. Yaklaşık 5 saniye.)*
> "Çünkü 'gerçek bir ihlal mi, yoksa bir anlık karışıklık mı?' diye 5 saniye doğruluyor. Yanlış alarmı işte böyle önlüyor."

**3) Beş saniye dolunca durum kırmızıya döner.**
> Söyle: "Ve işte — durum şimdi **İhlal**, kırmızı. 'Aktif ihlaller: 1'. Sistem bunu kaydetti, görüntüsünü sakladı. Artık bu bir kanıt; kimse 'baretimi takmıştım' diyemez."
> *(Kırmızı durumu ve 'Aktif ihlaller' sayısını işaret et.)*

**4) Bareti geri tak.**
> Söyle: "Şimdi baretimi geri takıyorum."
> *(Birkaç saniye bekle, yeşile dönsün.)*
> "Sistem uyumu yeniden onayladı — tekrar yeşil. İşçi güvene döndü, sistem bunu da gördü. Sürekli, otomatik, kesintisiz."

**Sistem hemen tepki vermezse (sakin kal):**
- Panik yapma. Şunu söyle: "Sistem temkinli — gerçek ihlalde emin olmak için 5 saniye bekliyor; bu bir kusur değil, tasarımın parçası."
- Hareketi net tekrarla (bareti tam çıkar, kamera açısına dön). Işık/açı önemli — kameraya dönük dur.
- Gerekirse: "Bu, yanlış alarmları sıfıra yakın tutan güvenlik marjı" diyerek beklemeyi bir **özelliğe** çevir.

**Vurgu:** Sistem gerçekten çalışıyor + 5 saniye kuralı = doğru ve güvenilir. Sessizlik (bekleme anı) en güçlü ikna.

**İpucu:** Acele etme; 5 saniyelik bekleme sırasında **sus ve ekranı izlet**. Jestleri abartılı/net yap (kamera görsün). Bu ekranda zamanın çoğunu harca.

---

## Ekran 6 — DEĞER & KVKK
**Ne söylemeli:**
"Bu sistem sizi nasıl korur? Üç açıdan. Yasal uyum: KVKK'ya uygun — takip tamamen **anonim**, yüz tanıma **yok**. Kimliği değil, sadece güvenlik durumunu izliyoruz. Maliyet: önlenen tek bir ciddi iş kazası bile bu sistemin maliyetini fazlasıyla karşılar. Ve denetim: 7 gün 24 saat kesintisiz — yorulmayan, dikkati dağılmayan bir gözcü gibi."

**Vurgu:** "Yüz tanıma yok, anonim" — yasal ve etik endişeyi peşinen kapatır. Maliyet argümanı karar vericiye yöneliktir.

**İpucu:** "Yüz tanıma yok" cümlesini net ve yavaş söyle. Maliyet derken karar veren kişiye bak.

---

## Ekran 7 — KAPANIŞ
**Ne söylemeli:**
"Toparlarsak: İşGözcü bir vaat değil — az önce canlı gördüğünüz, çalışan bir sistem. Bundan sonrası için yol haritamız da net: çok kameralı takip ve yeni güvenlik sınıfları, örneğin eldiven veya gözlük. İlginiz için teşekkür ederim; sorularınızı memnuniyetle alırım."

**Vurgu:** "Çalışan sistem" + net yol haritası = bugün hazır, yarın daha güçlü.

**İpucu:** Yol haritasını kısa tut (aynı zamanda "sınırlar ne?" sorusunun cevabı). Güçlü bir "Teşekkürler" ile bitir, soruları davet et.

---

## Olası Sorular ve Cevapları

**S: Doğruluğu ne kadar? Hata payı?**
C: "Doğrulama setinde ölçülmüş mAP değerimiz 0.82 — sektör için güçlü bir sonuç. Üç sınıfı tanıyor: kişi, baret, yelek. Üstelik 5 saniyelik doğrulama katmanı, anlık hataların alarma dönüşmesini engelliyor."

**S: İşçiler birbirine benzerse ya da üst üste geçerse, numaralar karışmaz mı?**
C: "Doğru tespit — bu, tüm görüntü-takip sistemlerinin bilinen bir sınırı: aynı kıyafetli kişiler tam üst üste geçince kimlik takibi zorlaşabilir. Biz bunu mühendislik olarak ele aldık: ihlal kaydı kişi numarasına değil, sahnedeki konuma da bağlanıyor, böylece numara değişse de ihlal kaybolmuyor. Tam çözüm için yol haritamızda **ikinci, farklı açıdan kamera** var — bu, üst üste binmeyi büyük ölçüde ortadan kaldırır." *(Olgun, hazırlıklı; özür dileme.)*

**S: Gizlilik? KVKK?**
C: "Tamamen KVKK uyumlu. Yüz tanıma **yok**, kişisel veri saklanmıyor. Takip **anonim** — 'işçi 1, işçi 2' gibi; kim olduğunu değil, güvenli olup olmadığını izliyoruz."

**S: Performans, hız, gereken donanım?**
C: "Gerçek zamanlı çalışıyor. Mütevazı bir dizüstü GPU'su (4 GB) yeterli — pahalı sunucu gerekmez. Mevcut kameranızla, kenarda (edge) çalışabilir."

**S: Maliyet / geri dönüş?**
C: "Çoğu şantiyede kameralar zaten var; biz üzerine zekâ ekliyoruz. Önlenen tek bir ciddi kaza — hem insani hem yasal hem mali — sistemin maliyetini kat kat çıkarır. Bu bir gider değil, sigorta."

**S: Gece veya zayıf ışıkta çalışır mı?**
C: "Bu, modelin değil kameranın sınırı: kamera görüyorsa sistem de görür. Düşük ışık için aydınlatma ya da IR kamera ile iyileştirilebilir; istenirse gece koşullarıyla da eğitilebilir."

**S: Kaç şey tespit ediyor? Eldiven, gözlük eklenebilir mi?**
C: "Şu an üç sınıf: kişi, baret, yelek. Mimari genişlemeye uygun — eldiven, gözlük, kask tipi gibi sınıflar yol haritamızda; veri toplayıp ekleyebiliyoruz."

**S: Sıradan bir güvenlik kamerasından farkı ne?**
C: "Güvenlik kamerası sadece **kaydeder** — birinin izlemesini ve fark etmesini bekler. İşGözcü **anlık değerlendirir**: ihlali kendisi görür, doğrular ve kanıtıyla bildirir. Pasif kayıt değil, aktif denetim."

---

## Genel Sunum İpuçları

- **Yavaş ve kendinden emin konuş.** Acele en sık hata. Cümleler arasında nefes al.
- **Ekrandan okuma.** Ekran senin notun değil; sen anlat, ekran destekler.
- **Demo her şeydir.** Slaytlarda hızlı geç, 5. ekranda zaman harca; "çalışıyor mu?" sorusunun cevabı orada.
- **5 saniyelik bekleme bir özelliktir** — sessizce izlet, sonra "yanlış alarmı böyle önlüyoruz" de.
- **Sınır sorusuna olgun cevap ver.** "Bilinen bir sınır; konuma bağlı kayıt + çok kameralı yol haritasıyla çözüyoruz." Özür değil, mühendislik.
- **Dr. Şimşek = bilimsel güven.** Akademik soru gelirse ona yönlendirebilirsin.
- **Teknik isim + 'yani' + değer.** Her terimi hemen faydaya çevir.

**Teknik hazırlık (sunumdan önce):**
- Dizüstü **şarjda** olsun.
- **GPU'yu maksimum performans** moduna al (akıcı FPS için).
- Kamera = **dizüstü kamerası, index 0**. Çalıştırma:
  `python app/server.py --index 0 --backend rfdetr --tracker ocsort`
- Tarayıcıda sunum: **http://localhost:8000/sunum** (5. ekrana gelince canlı yayın otomatik açılır).
- Önceden **dene**: ışık iyi mi, kamera seni net görüyor mu, baret/yelek hazır mı.

**Yedek plan:**
- Canlı kamera takılırsa, hazırda bir şantiye videosuyla aynı demoyu göster:
  `python app/server.py --source <video.mp4> --backend rfdetr --tracker ocsort`
- İnternet/yayın takılırsa sakin kal: "Bir saniye, yayını tazeleyelim" de; 5. ekran yeniden açılınca yayın kendini toparlar.

**Nasıl başla / nasıl bitir:**
- Başla: göz teması + "Bugün size çalışan bir sistemi canlı göstereceğim."
- Bitir: güçlü "Teşekkürler" + "Sorularınızı alayım" + sakin bekle.
