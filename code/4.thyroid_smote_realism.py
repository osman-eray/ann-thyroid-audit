# ============================================================
# SCRIPT 8 — SMOTE Gerçekçilik / Biyolojik Tutarlılık Kontrolü
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1-34, R1-35, R1-36, R1-37
#   #34: "Klinik veride sentetik oversampling gerçekçi olmayan hasta profilleri
#         üretebilir. Yazarlar bu riski tartışmalı."
#   #35: "SMOTE'un biyolojik olarak imkansız hormon kombinasyonları üretip
#         üretmediği incelenmeli."
#   #36: "Sentetik örneklerin klinik olarak anlamlı aralıklara kısıtlanması
#         düşünülmeli."
#   #37: "Her training split için SMOTE öncesi/sonrası sınıf dağılımı raporlanmalı."
#
# NE YAPAR:
#   1) Her split için SMOTE öncesi/sonrası sınıf dağılımı  (#37)
#   2) Sentetik örneklerin hormon değerlerinin fizyolojik aralık dışına
#      çıkıp çıkmadığı  (#35)
#   3) Biyolojik TUTARSIZLIK kontrolü: HPT ekseni mantığına aykırı
#      kombinasyonlar  (#35) — örn. TSH çok yüksek AMA TT4 da yüksek
#   4) Gerçek vs sentetik dağılım mesafesi (KS testi)  (#34)
#   5) Kısıtlama simülasyonu: klinik aralığa kırpma uygulanırsa kaç örnek
#      etkilenir  (#36)
#
#   NOT: ann-thyroid normalize dağıtılır; tüm kontroller ham (denormalize)
#   klinik birimlerde yapılır, aksi halde "fizyolojik aralık" anlamsızdır.
#
# ÇIKTI:
#   smote_class_distribution.csv   (#37)
#   smote_range_violations.csv     (#35)
#   smote_implausible_combos.csv   (#35)
#   smote_ks_distances.csv         (#34)
#   smote_realism_report.txt
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_smote_realism_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_DIST    = os.path.join(OUTPUT_DIR, "smote_class_distribution.csv")
F_RANGE   = os.path.join(OUTPUT_DIR, "smote_range_violations.csv")
F_COMBO   = os.path.join(OUTPUT_DIR, "smote_implausible_combos.csv")
F_KS      = os.path.join(OUTPUT_DIR, "smote_ks_distances.csv")
F_REPORT  = os.path.join(OUTPUT_DIR, "smote_realism_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'
CLASS_NAMES = {0: "overt_hypothyroid", 1: "subclinical_hypothyroid", 2: "euthyroid"}

# Normalizasyon katsayıları (veriden doğrulandı)
DENORM = {"TSH": 1000.0, "T3": 100.0, "TT4": 1000.0, "T4U": 10.0, "FTI": 1000.0, "age": 100.0}

# Fizyolojik olarak mümkün aralıklar (ham birimler, geniş tutuldu — sadece
# gerçekten imkansız olanları yakalamak için)
PHYSIO_RANGE = {
    "TSH": (0.0, 600.0),     # mIU/L — üst sınır ekstrem miksödem
    "T3":  (0.0, 20.0),      # nmol/L
    "TT4": (0.0, 650.0),     # nmol/L
    "T4U": (0.1, 3.0),       # oran
    "FTI": (0.0, 900.0),     # indeks
    "age": (0.0, 120.0),     # yıl
}

# Binary (0/1) olması gereken kolonlar — SMOTE ara değer üretirse tutarsız
BINARY_COLS = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
               'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
               'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
               'tumor', 'hypopituitary', 'psych']


def denormalize(df):
    out = df.copy()
    for c, k in DENORM.items():
        if c in out.columns:
            out[c] = out[c] * k
    return out


# ============================================================
# HFS (makale ile birebir — SMOTE aynı özellik alt kümesinde çalışsın)
# ============================================================

def information_gain(X_col, y):
    return mutual_info_classif(X_col.reshape(-1, 1), y,
                               discrete_features='auto', random_state=INTERNAL_SEED)[0]

def gain_ratio(X_col, y):
    IG = information_gain(X_col, y)
    n_unique = len(np.unique(X_col))
    if n_unique <= 10:
        _, counts = np.unique(X_col, return_counts=True)
    else:
        n_bins = max(10, int(np.sqrt(len(X_col))))
        counts = np.histogram(X_col, bins=n_bins)[0]; counts = counts[counts > 0]
    probs = counts / counts.sum()
    Hx = -np.sum(probs * np.log2(probs))
    return IG / (Hx + 1e-12) if Hx > 0 else 0.0

def hfs_select(Xtr, ytr, k=K_SELECT, theta=0.90, w=0.5):
    feats = list(Xtr.columns); X = Xtr.values
    redundant = set(); n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                redundant.add(feats[j])
    non_red = [f for f in feats if f not in redundant]
    scores = []
    for f in non_red:
        xf = Xtr[f].values
        scores.append((f, w * information_gain(xf, ytr) + (1 - w) * gain_ratio(xf, ytr)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores[:min(k, len(scores))]]


# ============================================================
# Biyolojik tutarlılık kuralları (#35)
# ============================================================

def implausible_combos(df_raw):
    """
    HPT (hipotalamus-hipofiz-tiroid) ekseni mantığına aykırı kombinasyonlar.
    Negatif geri besleme nedeniyle TSH ve TT4 ters yönde hareket eder.
    Dönüş: her kural için boolean maske.
    """
    checks = {}
    if "TSH" in df_raw and "TT4" in df_raw:
        # TSH çok yüksek AMA TT4 da yüksek → negatif geri besleme ihlali
        checks["TSH_high_AND_TT4_high"] = (df_raw["TSH"] > 10.0) & (df_raw["TT4"] > 140.0)
        # TSH baskılı AMA TT4 düşük → (santral hipotiroidi hariç) beklenmez
        checks["TSH_low_AND_TT4_low"] = (df_raw["TSH"] < 0.4) & (df_raw["TT4"] < 60.0)
    if "TT4" in df_raw and "FTI" in df_raw:
        # TT4 ve FTI birbirinden çok kopuk (FTI = TT4/T4U mantığı)
        checks["TT4_FTI_divergent"] = (
            (df_raw["TT4"] > 120.0) & (df_raw["FTI"] < 40.0)
        ) | ((df_raw["TT4"] < 40.0) & (df_raw["FTI"] > 120.0))
    if "T3" in df_raw and "TT4" in df_raw:
        # T3 yüksek AMA TT4 çok düşük
        checks["T3_high_AND_TT4_verylow"] = (df_raw["T3"] > 3.0) & (df_raw["TT4"] < 40.0)
    return checks


def range_violations(df_raw, feats):
    """Fizyolojik aralık dışına çıkan değerler."""
    out = {}
    for f in feats:
        if f in PHYSIO_RANGE:
            lo, hi = PHYSIO_RANGE[f]
            out[f] = ((df_raw[f] < lo) | (df_raw[f] > hi))
    return out


def binary_violations(df_norm, feats):
    """SMOTE binary kolonlarda ara değer (0/1 dışı) üretmiş mi?"""
    out = {}
    for f in feats:
        if f in BINARY_COLS:
            v = df_norm[f].values
            # 0 veya 1'e yakın olmayan (tolerans dışı) değerler
            out[f] = ~(np.isclose(v, 0.0, atol=1e-6) | np.isclose(v, 1.0, atol=1e-6))
    return out


# ============================================================
# Ana analiz
# ============================================================

def run_all():
    print("=" * 64)
    print("SMOTE GERÇEKÇİLİK KONTROLÜ — ann-thyroid")
    print("=" * 64)
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek")
    print(f"Fizyolojik aralıklar (ham birim): {PHYSIO_RANGE}")
    print("=" * 64)

    dist_rows, range_rows, combo_rows, ks_rows = [], [], [], []

    for run_id in SPLIT_SEEDS:
        X = data[FEATURE_COLS]
        y = data[TARGET_COL] - 1
        Xtr_n, Xte_n, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=run_id, stratify=y)

        # Pipeline'daki gibi: önce ölçekle, sonra HFS, sonra SMOTE
        scaler = StandardScaler()
        Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr_n), columns=FEATURE_COLS)
        feats = hfs_select(Xtr_s, ytr.values)

        # SMOTE'u HFS alt kümesinde uygula (makale pipeline'ı ile aynı)
        XtrH_s = Xtr_s[feats]
        sm = SMOTE(random_state=INTERNAL_SEED, k_neighbors=5)
        Xres_s, yres = sm.fit_resample(XtrH_s, ytr)

        # --- (#37) sınıf dağılımı öncesi/sonrası
        before = ytr.value_counts().sort_index()
        after = pd.Series(yres).value_counts().sort_index()
        row = {"run_id": run_id, "n_before": len(ytr), "n_after": len(yres),
               "n_synthetic": len(yres) - len(ytr), "k_neighbors": 5,
               "n_features_hfs": len(feats)}
        for c in [0, 1, 2]:
            row[f"before_{CLASS_NAMES[c]}"] = int(before.get(c, 0))
            row[f"after_{CLASS_NAMES[c]}"] = int(after.get(c, 0))
            row[f"synthetic_{CLASS_NAMES[c]}"] = int(after.get(c, 0) - before.get(c, 0))
        dist_rows.append(row)

        # --- sentetik örnekleri ayır (orijinaller SMOTE çıktısının başında gelir)
        n_orig = len(XtrH_s)
        Xsyn_s = Xres_s.iloc[n_orig:] if hasattr(Xres_s, "iloc") else pd.DataFrame(
            Xres_s[n_orig:], columns=feats)
        ysyn = np.asarray(yres)[n_orig:]

        if len(Xsyn_s) == 0:
            continue

        # --- ölçeği geri al: standardize → normalize → ham
        idx = [FEATURE_COLS.index(f) for f in feats]
        mu = scaler.mean_[idx]
        sd = scaler.scale_[idx]
        Xsyn_norm = pd.DataFrame(Xsyn_s.values * sd + mu, columns=feats)
        Xorig_norm = pd.DataFrame(XtrH_s.values * sd + mu, columns=feats)
        Xsyn_raw = denormalize(Xsyn_norm)
        Xorig_raw = denormalize(Xorig_norm)

        # --- (#35) fizyolojik aralık ihlalleri
        viol = range_violations(Xsyn_raw, feats)
        for f, mask in viol.items():
            n_v = int(mask.sum())
            range_rows.append({
                "run_id": run_id, "feature": f, "kind": "physiological_range",
                "n_synthetic": len(Xsyn_raw), "n_violations": n_v,
                "pct_violations": round(n_v / len(Xsyn_raw) * 100, 3),
                "range_lo": PHYSIO_RANGE[f][0], "range_hi": PHYSIO_RANGE[f][1],
                "syn_min": round(float(Xsyn_raw[f].min()), 3),
                "syn_max": round(float(Xsyn_raw[f].max()), 3),
                "orig_min": round(float(Xorig_raw[f].min()), 3),
                "orig_max": round(float(Xorig_raw[f].max()), 3),
            })
        # binary ihlalleri
        bviol = binary_violations(Xsyn_norm, feats)
        for f, mask in bviol.items():
            n_v = int(mask.sum())
            range_rows.append({
                "run_id": run_id, "feature": f, "kind": "binary_integrity",
                "n_synthetic": len(Xsyn_norm), "n_violations": n_v,
                "pct_violations": round(n_v / len(Xsyn_norm) * 100, 3),
                "range_lo": 0, "range_hi": 1,
                "syn_min": round(float(Xsyn_norm[f].min()), 4),
                "syn_max": round(float(Xsyn_norm[f].max()), 4),
                "orig_min": 0.0, "orig_max": 1.0,
            })

        # --- (#35) biyolojik tutarsız kombinasyonlar: sentetik VE orijinal
        for label, dfr in [("synthetic", Xsyn_raw), ("original", Xorig_raw)]:
            checks = implausible_combos(dfr)
            for rule, mask in checks.items():
                n_v = int(mask.sum())
                combo_rows.append({
                    "run_id": run_id, "source": label, "rule": rule,
                    "n_samples": len(dfr), "n_violations": n_v,
                    "pct_violations": round(n_v / len(dfr) * 100, 3),
                })

        # --- (#34) gerçek vs sentetik dağılım mesafesi (KS)
        for f in feats:
            if f in DENORM:   # sürekli özellikler
                # sadece azınlık sınıflarda karşılaştır (sentetikler oradan)
                for c in [0, 1]:
                    orig_c = Xorig_raw[f].values[ytr.values == c]
                    syn_c = Xsyn_raw[f].values[ysyn == c]
                    if len(orig_c) > 3 and len(syn_c) > 3:
                        stat, p = ks_2samp(orig_c, syn_c)
                        ks_rows.append({
                            "run_id": run_id, "feature": f,
                            "class": CLASS_NAMES[c],
                            "ks_statistic": round(float(stat), 4),
                            "p_value": round(float(p), 5),
                            "n_orig": len(orig_c), "n_syn": len(syn_c),
                        })

        print(f"  seed={run_id}: {len(ytr)} → {len(yres)} "
              f"(+{len(yres)-len(ytr)} sentetik) | HFS={len(feats)} özellik")

    pd.DataFrame(dist_rows).to_csv(F_DIST, index=False)
    pd.DataFrame(range_rows).to_csv(F_RANGE, index=False)
    pd.DataFrame(combo_rows).to_csv(F_COMBO, index=False)
    pd.DataFrame(ks_rows).to_csv(F_KS, index=False)
    print(f"\n  ✓ {F_DIST}\n  ✓ {F_RANGE}\n  ✓ {F_COMBO}\n  ✓ {F_KS}")

    write_report(dist_rows, range_rows, combo_rows, ks_rows)


def write_report(dist_rows, range_rows, combo_rows, ks_rows):
    d = pd.DataFrame(dist_rows)
    r = pd.DataFrame(range_rows)
    c = pd.DataFrame(combo_rows)
    k = pd.DataFrame(ks_rows)

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\nSMOTE GERÇEKÇİLİK / BİYOLOJİK TUTARLILIK — ann-thyroid\n")
        f.write("R1-34/35/36/37\n" + "=" * 72 + "\n\n")

        # #37
        f.write("[R1-37] SINIF DAĞILIMI — SMOTE ÖNCESİ / SONRASI (10 split ort.)\n")
        f.write("-" * 72 + "\n")
        for cls in CLASS_NAMES.values():
            b = d[f"before_{cls}"].mean()
            a = d[f"after_{cls}"].mean()
            s = d[f"synthetic_{cls}"].mean()
            f.write(f"  {cls:<26} önce={b:7.1f}  sonra={a:7.1f}  sentetik=+{s:7.1f}\n")
        f.write(f"\n  Toplam: {d['n_before'].mean():.0f} → {d['n_after'].mean():.0f} "
                f"(+{d['n_synthetic'].mean():.0f} sentetik örnek)\n")
        f.write(f"  SMOTE k_neighbors = 5 (en küçük sınıf train'de "
                f"≈{d['before_' + CLASS_NAMES[0]].min():.0f} örnek → k=5 uygun)\n")

        # #35 aralık
        f.write("\n\n[R1-35] FİZYOLOJİK ARALIK İHLALLERİ (sentetik örnekler)\n")
        f.write("-" * 72 + "\n")
        rr = r[r.kind == "physiological_range"]
        if len(rr):
            g = rr.groupby("feature").agg(
                mean_pct=("pct_violations", "mean"),
                total_viol=("n_violations", "sum"),
                syn_min=("syn_min", "min"), syn_max=("syn_max", "max"),
                orig_min=("orig_min", "min"), orig_max=("orig_max", "max"),
            ).round(3)
            f.write(f"{'Özellik':<10}{'İhlal %':<12}{'Sentetik aralık':<26}{'Orijinal aralık':<26}\n")
            for feat, row in g.iterrows():
                f.write(f"{feat:<10}{row['mean_pct']:<12.3f}"
                        f"[{row['syn_min']:.2f}, {row['syn_max']:.2f}]".ljust(36) +
                        f"[{row['orig_min']:.2f}, {row['orig_max']:.2f}]\n")
            tot = rr["n_violations"].sum()
            f.write(f"\n  → Toplam fizyolojik aralık ihlali: {tot}\n")
            if tot == 0:
                f.write("  → SMOTE interpolasyon yaptığı için sentetik değerler daima\n")
                f.write("    mevcut örneklerin konveks kabuğu içinde kalır; bu nedenle\n")
                f.write("    aralık dışına çıkmaz. (Beklenen ve doğrulanan davranış.)\n")

        # binary
        f.write("\n\n[R1-35] BINARY BÜTÜNLÜK (SMOTE 0/1 arası ara değer üretti mi?)\n")
        f.write("-" * 72 + "\n")
        rb = r[r.kind == "binary_integrity"]
        if len(rb):
            gb = rb.groupby("feature").agg(
                mean_pct=("pct_violations", "mean"),
                syn_min=("syn_min", "min"), syn_max=("syn_max", "max"),
            ).round(3).sort_values("mean_pct", ascending=False)
            for feat, row in gb.iterrows():
                f.write(f"  {feat:<28} ara-değer %{row['mean_pct']:6.2f}  "
                        f"aralık=[{row['syn_min']:.3f}, {row['syn_max']:.3f}]\n")
            f.write("\n  → SMOTE sürekli interpolasyon yapar; binary özelliklerde\n")
            f.write("    0/1 arası kesirli değerler üretmesi beklenen bir davranıştır.\n")
            f.write("    Bu, klinik olarak 'kısmen hamile' gibi yorumlanamaz durumlara\n")
            f.write("    karşılık gelir ve SMOTE-NC bir alternatif olabilir (bkz. Limitations).\n")
        else:
            f.write("  (HFS binary özellik seçmemiş — kontrol uygulanamadı)\n")

        # #35 kombinasyon
        f.write("\n\n[R1-35] BİYOLOJİK OLARAK TUTARSIZ HORMON KOMBİNASYONLARI\n")
        f.write("-" * 72 + "\n")
        f.write("HPT ekseni: TSH ve TT4 negatif geri besleme ile ters yönde hareket eder.\n\n")
        if len(c):
            gc = c.groupby(["rule", "source"])["pct_violations"].mean().unstack().round(3)
            f.write(f"{'Kural':<30}{'Sentetik %':<16}{'Orijinal %':<16}{'Fark':<10}\n")
            for rule in gc.index:
                syn = gc.loc[rule].get("synthetic", np.nan)
                org = gc.loc[rule].get("original", np.nan)
                diff = syn - org if not (np.isnan(syn) or np.isnan(org)) else np.nan
                f.write(f"{rule:<30}{syn:<16.3f}{org:<16.3f}{diff:+.3f}\n")
            f.write("\n  → 'Fark' pozitifse SMOTE, orijinal veride bulunmayan tutarsız\n")
            f.write("    profiller üretiyor demektir. Sıfıra yakınsa sentetik örnekler\n")
            f.write("    gerçek verinin biyolojik yapısını koruyor.\n")

        # #34 KS
        f.write("\n\n[R1-34] GERÇEK vs SENTETİK DAĞILIM MESAFESİ (Kolmogorov-Smirnov)\n")
        f.write("-" * 72 + "\n")
        if len(k):
            gk = k.groupby(["feature", "class"]).agg(
                ks=("ks_statistic", "mean"), p=("p_value", "mean")
            ).round(4)
            f.write(f"{'Özellik':<10}{'Sınıf':<26}{'KS ist.':<12}{'p (ort)':<12}\n")
            for (feat, cls), row in gk.iterrows():
                flag = "  ← farklı" if row["p"] < 0.05 else ""
                f.write(f"{feat:<10}{cls:<26}{row['ks']:<12.4f}{row['p']:<12.4f}{flag}\n")
            f.write("\n  → Düşük KS + yüksek p: sentetik dağılım gerçeğe yakın.\n")
            f.write("    Yüksek KS + düşük p: SMOTE dağılımı kaydırmış olabilir.\n")

        # #36
        f.write("\n\n[R1-36] KLİNİK ARALIĞA KISITLAMA — DEĞERLENDİRME\n")
        f.write("-" * 72 + "\n")
        n_range_viol = r[r.kind == "physiological_range"]["n_violations"].sum() if len(r) else 0
        f.write(f"  Fizyolojik aralık ihlali sayısı: {n_range_viol}\n")
        if n_range_viol == 0:
            f.write("  → Kırpma (clipping) tabanlı bir kısıtlama HİÇBİR sentetik örneği\n")
            f.write("    etkilemezdi; SMOTE'un interpolasyon doğası gereği üretilen\n")
            f.write("    değerler zaten gözlemlenen aralıkta kalıyor. Dolayısıyla\n")
            f.write("    #36'da önerilen kısıtlama bu veri seti için işlemsiz olurdu.\n")
            f.write("    Asıl risk aralık ihlali değil, KOMBİNASYON gerçekçiliğidir\n")
            f.write("    (yukarıdaki tutarsızlık tablosuna bakınız).\n")

    print(f"  ✓ {F_REPORT}")

    # konsol özeti
    print("\nÖZET:")
    print(f"  Ortalama sentetik örnek: +{d['n_synthetic'].mean():.0f} / split")
    rr = r[r.kind == "physiological_range"]
    print(f"  Fizyolojik aralık ihlali: {rr['n_violations'].sum() if len(rr) else 0}")
    if len(c):
        syn_mean = c[c.source == "synthetic"]["pct_violations"].mean()
        org_mean = c[c.source == "original"]["pct_violations"].mean()
        print(f"  Tutarsız kombinasyon — sentetik: %{syn_mean:.3f}  orijinal: %{org_mean:.3f}")


if __name__ == "__main__":
    run_all()
    print("\n" + "=" * 64 + "\nDONE\n" + "=" * 64)
