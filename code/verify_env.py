"""
verify_env.py — Ortam dogrulama.

Amac: yeniden kurulan `thyroid` conda ortaminin, makalede raporlanan
sonuclari birebir uretip uretmedigini kontrol etmek.

Tam ablasyonu kosmaz (o 2-3 saat surer). Yalnizca uc seed icin AdaBoost
baseline degerini hesaplar ve S7.3'te raporlanan degerlerle karsilastirir.

Kullanim (Anaconda Prompt, thyroid ortami aktifken):
    cd <bu deponun kok klasoru>
    python verify_env.py

REV: sabit kodlanmis Windows yolu kaldirildi; script artik kendi bulundugu
klasorde calisir, boylece depoyu klonlayan herkes calistirabilir.

Not: figur scripti "11.make_figures_1_6_7_8.py" olarak adlandirilmistir.
Dosya adi degisirse asagidaki SCRIPT sabiti guncellenmelidir.

Beklenen: uc seedde de FARK = 0.000000 ve TAMAM isareti.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

FOLDER = os.path.dirname(os.path.abspath(__file__))   # REV: was a hard-coded path
DATA = "thyroid_ann_7200.csv"
SCRIPT = "11.make_figures_1_6_7_8.py"

# S7.3 / ablasyon ciktisindaki AdaBoost A_baseline roc_auc_ovr degerleri
BEKLENEN = {0: 0.990523, 1: 0.988141, 2: 0.992931}
TOLERANS = 0.0005


def main() -> int:
    os.chdir(FOLDER)

    import pandas as pd
    import sklearn
    from sklearn.metrics import roc_auc_score

    print("=" * 60)
    print("ORTAM")
    print("=" * 60)
    print("python     :", sys.version.split()[0])
    print("yorumlayici:", sys.executable)
    print("sklearn    :", sklearn.__version__, "(beklenen 1.3.2)")
    print("konum      :", os.path.dirname(sklearn.__file__))
    print()

    for f in (DATA, SCRIPT):
        if not os.path.exists(f):
            print(f"HATA: {f} bu klasorde yok -> {FOLDER}")
            return 1

    # Figur scriptinin fonksiyonlarini, __main__ blogunu calistirmadan yukle
    src = open(SCRIPT, encoding="utf-8").read().split("if __name__ ==")[0]
    ns = {}
    exec(compile(src, SCRIPT, "exec"), ns)

    if "algorithm=\"SAMME\"" not in src and "algorithm='SAMME'" not in src:
        print("UYARI: scriptte algorithm='SAMME' bulunamadi.")
        print("       Duzeltilmis surumu kullandiginizdan emin olun;")
        print("       aksi halde sklearn 1.3.2 SAMME.R kullanir ve")
        print("       ~0.9965 gibi bir deger cikar.")
        print()

    data = pd.read_csv(DATA)
    print("veri       :", data.shape)
    print()

    print("=" * 60)
    print("ADABOOST BASELINE DOGRULAMASI")
    print("=" * 60)
    print(f"{'seed':>5}  {'hesaplanan':>12}  {'beklenen':>12}  {'fark':>10}  durum")

    tum_tamam = True
    for seed, hedef in BEKLENEN.items():
        Xtr, Xte, ytr, yte, _ = ns["prepare_split"](data, seed)
        est = ns["fit_pipeline"]("AdaBoost", Xtr, ytr, False)
        v = roc_auc_score(
            yte, est.predict_proba(Xte), multi_class="ovr", average="weighted"
        )
        fark = abs(v - hedef)
        ok = fark < TOLERANS
        tum_tamam &= ok
        print(f"{seed:>5}  {v:>12.6f}  {hedef:>12.6f}  {fark:>10.6f}  "
              f"{'TAMAM' if ok else 'FARKLI'}")

    print()
    if tum_tamam:
        print("SONUC: ortam dogrulandi. Revizyona devam edilebilir.")
        return 0

    print("SONUC: degerler tutmuyor.")
    print("  ~0.9965 civari  -> algorithm='SAMME' eksik, script duzeltilmemis")
    print("  baska bir fark  -> sklearn surumunu ve veri dosyasini kontrol edin")
    return 2


if __name__ == "__main__":
    sys.exit(main())
