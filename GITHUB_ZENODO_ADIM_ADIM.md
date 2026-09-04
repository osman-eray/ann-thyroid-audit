# GitHub ve Zenodo — adım adım

Hiç yapmamış biri için yazıldı. Yaklaşık 30–40 dakika sürer. Arayüz etiketleri zaman zaman
değişebilir; mantık aynı kalır.

**Baştan bilinmesi gereken iki şey:**

1. **Zenodo entegrasyonunu release'den ÖNCE açmalısınız.** Zenodo yalnızca, anahtar açıldıktan
   sonra oluşturulan release'leri arşivler. Sırayı ters yaparsanız release Zenodo'ya düşmez ve
   ikinci bir release oluşturmanız gerekir.
2. **Depo public olmalı.** Zenodo private depoları arşivlemez, ve hakem zaten inceleme aşamasında
   erişim istedi.

---

# BÖLÜM 1 — GitHub

## 1.1 Hesap

github.com → **Sign up**. Akademik e-postanız (`oeray@akdeniz.edu.tr`) iyi bir tercih; Zenodo
kaydında da aynı kimlik görünür. E-postayı doğrulayın.

Kullanıcı adı kalıcıdır ve DOI'ye giden URL'de yer alır. Kurumsal/profesyonel bir ad seçin.

## 1.2 Depoyu oluştur

Sağ üstte **+** → **New repository**.

- **Repository name:** `ann-thyroid-audit`
- **Description:** `Controlled audit of the UCI ann-thyroid multiclass benchmark: label semantics, method comparison, and reproducibility`
- **Public** seçin
- **"Add a README file", "Add .gitignore", "Choose a license" kutularının ÜÇÜNÜ DE İŞARETSİZ
  bırakın.** Bu dosyalar zaten pakette var; GitHub'ın kendi sürümlerini eklerseniz çakışır.
- **Create repository**

Boş bir depo ve "quick setup" ekranı göreceksiniz. Bir sonraki adıma geçin.

## 1.3 Dosyaları yükle

Git kurmanıza gerek yok. Bu tek seferlik bir yükleme, tarayıcı arayüzü yeterli.

1. `ann_thyroid_audit_repo.zip` dosyasını açın. İçinde bir `pkg` klasörü var.
2. `pkg` klasörünün **içine** girin. Şunları görmelisiniz: `code`, `data`, `results`,
   `README.md`, `LICENSE`, `CITATION.cff`, `CHANGES.md`, `DEPOSIT.md`,
   `SUPPLEMENTARY_S7.3_revised.md`, `requirements.txt`, `environment.yml`, `.gitignore`.
3. GitHub'daki boş depo sayfasında **uploading an existing file** bağlantısına tıklayın.
4. `pkg` klasörünün **içindekilerin tümünü** seçip (klasörün kendisini değil) tarayıcı
   penceresine sürükleyin. Klasörler alt dosyalarıyla birlikte yüklenir.
5. Alttaki kutuya commit mesajı: `Code and data for the ann-thyroid benchmark audit`
6. **Commit changes**

`.gitignore` gizli dosya olduğu için işletim sisteminizde görünmeyebilir. Windows'ta Dosya
Gezgini → Görünüm → "Gizli öğeler"i işaretleyin. Yüklenmezse önemli değil, sonradan da
eklenebilir.

## 1.4 Kontrol

Depo ana sayfasında README'nin biçimlenmiş olarak göründüğünü, sağda "MIT license" yazdığını, ve
`code/` içinde 21 `.py` dosyası, `data/` içinde 8 dosya olduğunu doğrulayın.

`thyroid_ann_7200.csv` **olmamalı** — bilerek dışarıda bırakıldı, `make_dataset.py` onu ham
dosyalardan üretiyor.

## 1.5 Placeholder'ları düzelt

`CITATION.cff` dosyasına tıklayın → kalem ikonu (**Edit this file**). Üç yeri değiştirin:

- `orcid:` satırına gerçek ORCID'inizi yazın (yoksa orcid.org'dan 2 dakikada alınır; MDPI
  makalenizde zaten bir tane var)
- `repository-code:` satırındaki `USERNAME` yerine GitHub kullanıcı adınız
- `date-released:` satırına release tarihi (bugünün tarihi)

**Commit changes** → **Commit directly to the main branch**.

---

# BÖLÜM 2 — Zenodo

## 2.1 Giriş

zenodo.org → **Sign up** → **Sign up with GitHub**. Bu, iki hesabı doğrudan bağlar; ayrı hesap
açıp sonradan bağlamaktan daha az adım.

GitHub izin isteyecek → **Authorize zenodo**.

## 2.2 Entegrasyonu aç — RELEASE'DEN ÖNCE

1. Zenodo'da sağ üstte adınız → **Settings** (bazı sürümlerde **Profile**)
2. Sol menüden **GitHub**
3. Depolarınızın listesini göreceksiniz. Liste boşsa **Sync now** düğmesine basın ve bir dakika
   bekleyin.
4. `ann-thyroid-audit` satırındaki anahtarı **On** konumuna getirin.

Bu adımı atlarsanız release Zenodo'ya düşmez. En sık yapılan hata budur.

## 2.3 GitHub'da release oluştur

GitHub'daki depoya dönün.

1. Sağ sütunda **Releases** → **Create a new release**
   (hiç release yoksa yazı "Create a new release" yerine "Releases" altında küçük görünebilir)
2. **Choose a tag** → kutuya `v1.0.0` yazın → **Create new tag: v1.0.0 on publish**
3. **Release title:** `Initial release for peer review`
4. **Describe this release** kutusuna:
   ```
   Code, raw UCI data and generated results accompanying the manuscript
   "A Controlled Audit of the ann-thyroid Benchmark" (Scientific Reports, under review).
   ```
5. **Publish release**

## 2.4 DOI'yi al

Birkaç dakika içinde Zenodo release'i arşivler.

Zenodo → **Settings → GitHub** → depo satırına tıklayın. Bir DOI rozeti göreceksiniz.

**Burada dikkat: iki DOI var.**

- **Concept DOI** — her zaman en son sürüme çözümlenir. Zenodo kaydında "Cite all versions" ya da
  benzeri bir ifadeyle gösterilir. **Makalede bunu kullanın.**
- **Version DOI** — yalnızca `v1.0.0`'a işaret eder. Kodu revizyondan sonra güncellerseniz bu
  eskir.

İkisi ardışık numaralıdır; concept DOI genelde küçük olanıdır. Emin olmak için Zenodo kaydını
açın: sağ sütunda "Versions" bölümü altında hangisinin tüm sürümlere ait olduğu yazar.

## 2.5 Zenodo kaydını düzelt

Kayıt sayfasında **Edit** düğmesi var. Şunları kontrol edin:

- **Authors:** Osman Eray, ORCID ve kurum bilgisi doğru mu
- **License:** MIT olarak görünüyor mu (görünmüyorsa elle seçin)
- **Resource type:** Software
- **Related identifiers:** makale yayımlandığında DOI'sini buraya "is supplement to" ilişkisiyle
  ekleyin. Şimdilik boş bırakabilirsiniz.

**Publish** ile kaydedin. DOI değişmez.

Zenodo bazı meta verileri `CITATION.cff` dosyasından okur, bazılarını okumaz — bu yüzden gözle
kontrol etmek gerekiyor.

## 2.6 Son kontrol

Zenodo kaydındaki arşiv dosyasını (`.zip`) indirin ve açın. `code/`, `data/`, `results/` ve
README'nin içinde olduğunu doğrulayın. Bu, hakemin göreceği şeydir.

---

# BÖLÜM 3 — Manuscript'e işleme

DOI elinizde. `DEPOSIT.md` içindeki üç metni kullanın, `USERNAME` ve `XXXXXXX` yerlerini doldurun:

- **Data availability statement**
- **Code availability statement**
- **Hakem 1'e erişilebilirlik cevabı**

Yeniden göndermeden önce son bir kontrol: makaledeki GitHub bağlantısını ve DOI'yi **gizli
sekmede** açın. Oturum açmadan erişilebiliyorsa hakem de erişebilir.

---

# Sonradan kod değişirse

Revizyon sırasında bir düzeltme gerekirse: dosyayı GitHub'da düzenleyin, sonra yeni bir release
oluşturun (`v1.0.1`). Zenodo otomatik olarak yeni bir sürüm arşivler ve **concept DOI aynı kalır**
— makaledeki bağlantıyı değiştirmeniz gerekmez. Makalede version DOI kullanırsanız bu işlemez;
concept DOI kullanmanızın sebebi budur.

---

# Sık karşılaşılan sorunlar

**Zenodo'da depo listede yok.** Settings → GitHub → **Sync now**. Depo public değilse hiç
görünmez.

**Release yayınlandı ama Zenodo'da DOI yok.** Anahtar release'den önce açılmamış demektir.
Çözüm: anahtarı açın, GitHub'da `v1.0.1` etiketiyle ikinci bir release oluşturun. İlk release
arşivlenmeden kalır, zararı yok.

**Yükleme sırasında "file too large".** Bizim en büyük dosyamız 772 KB, bu hatayı almamalısınız.
Alırsanız yanlış klasörü sürüklemişsinizdir.

**README GitHub'da düz metin görünüyor.** Dosya adı tam olarak `README.md` olmalı — uzantı
eksikse biçimlenmez.
