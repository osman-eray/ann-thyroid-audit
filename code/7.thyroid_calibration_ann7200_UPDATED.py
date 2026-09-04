# ============================================================
# SCRIPT 5 — Kalibrasyon Analizi
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1-69, R1-70
#   #69: "Klinik sınıflandırıcılar kalibre olasılık ister, yalnızca yüksek AUC değil."
#   #70: "Calibration curves, Brier score, expected calibration error (ECE),
#         reliability diagrams raporlayın."
#
# NE YAPAR:
#   HFS+SMOTE pipeline'ı 10 split üzerinde koşulur; her model için kalibrasyon
#   metrikleri hesaplanır:
#     - Multiclass Brier score (one-vs-rest ortalama)
#     - Expected Calibration Error (ECE, one-vs-rest ortalama, 10 bin)
#     - Maximum Calibration Error (MCE)
#     - Reliability diagram verisi (bin bazında güven vs doğruluk) → CSV
#   Ayrıca kalibrasyon ÖNCESİ ve SONRASI (Platt/sigmoid + isotonic) karşılaştırması,
#   böylece "kalibre edilebilir mi?" sorusu yanıtlanır (R1-69).
#
# ÇIKTI:
#   calibration_all_runs.csv, calibration_summary.csv,
#   reliability_curve_data.csv, calibration_report.txt
#
# RESUME: (model, calibration, seed) bazında.
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8
N_BINS = 10

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_calibration_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_ALL     = os.path.join(OUTPUT_DIR, "calibration_all_runs.csv")
F_SUMMARY = os.path.join(OUTPUT_DIR, "calibration_summary.csv")
F_CURVE   = os.path.join(OUTPUT_DIR, "reliability_curve_data.csv")
F_REPORT  = os.path.join(OUTPUT_DIR, "calibration_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'
# Sınıf adlandırması hormon profiline göre doğrulanmıştır (bkz. makale §2.1):
#   Class 0: TSH medyan 53.0 (↑↑), TT4 31.6 (↓)   → overt hypothyroid
#   Class 1: TSH medyan 9.7 (↑), TT4 90.0         → subclinical hypothyroid
#   Class 2: TSH medyan 1.5, TT4 109.0            → euthyroid
CLASS_NAMES = {0: "overt_hypothyroid", 1: "subclinical_hypothyroid", 2: "euthyroid"}

def get_base_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=INTERNAL_SEED),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, C=1, random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
    }

# Kalibrasyon koşulları: raw (kalibrasyonsuz), sigmoid (Platt), isotonic
CAL_MODES = ["raw", "sigmoid", "isotonic"]

# ============================================================
# HFS
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
# Kalibrasyon metrikleri (multiclass, one-vs-rest)
# ============================================================

def multiclass_brier(y_true_bin, y_prob):
    """OVR ortalama Brier score."""
    return np.mean([brier_score_loss(y_true_bin[:, c], y_prob[:, c])
                    for c in range(y_prob.shape[1])])

def ece_mce_ovr(y_true_bin, y_prob, n_bins=N_BINS):
    """Her sınıf için ECE/MCE, sonra OVR ortalama. Reliability eğrisi verisi de döner."""
    eces, mces, curve = [], [], []
    bins = np.linspace(0, 1, n_bins + 1)
    for c in range(y_prob.shape[1]):
        conf = y_prob[:, c]; acc = y_true_bin[:, c]
        ece = 0.0; mce = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
            if mask.sum() == 0:
                continue
            bin_conf = conf[mask].mean()
            bin_acc = acc[mask].mean()
            gap = abs(bin_conf - bin_acc)
            wgt = mask.sum() / len(conf)
            ece += wgt * gap
            mce = max(mce, gap)
            curve.append({"class": CLASS_NAMES[c], "bin": b,
                          "bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                          "mean_confidence": round(bin_conf, 4),
                          "mean_accuracy": round(bin_acc, 4),
                          "count": int(mask.sum())})
        eces.append(ece); mces.append(mce)
    return np.mean(eces), np.mean(mces), curve

# ============================================================
# Resume
# ============================================================

def load_done():
    if not os.path.exists(F_ALL):
        return set()
    df = pd.read_csv(F_ALL)
    return set(zip(df["model"], df["calibration"], df["run_id"]))

def append_csv(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ============================================================
# Ana döngü
# ============================================================

def run_all():
    print("="*60)
    print("KALİBRASYON ANALİZİ — ann-thyroid")
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek | bin sayısı: {N_BINS}")
    print(f"Modeller: {list(get_base_models().keys())}")
    print(f"Kalibrasyon modları: {CAL_MODES}")
    print("="*60)
    done = load_done()
    if done:
        print(f"[RESUME] {len(done)} (model,cal,seed) tamam")

    for run_id in SPLIT_SEEDS:
        X_raw = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X_raw, y, test_size=0.2, random_state=run_id, stratify=y)
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)
        feats = hfs_select(Xtr, ytr.values)
        XtrH, XteH = Xtr[feats], Xte[feats]
        labels = sorted(yte.unique())
        yte_bin = label_binarize(yte, classes=labels)

        # SMOTE'u train'e uygula (kalibrasyon fit'i için dengeli train)
        sm = SMOTE(random_state=INTERNAL_SEED)
        Xtr_bal, ytr_bal = sm.fit_resample(XtrH, ytr)

        for mname, base in get_base_models().items():
            for cal in CAL_MODES:
                if (mname, cal, run_id) in done:
                    continue
                t0 = time.time()
                try:
                    if cal == "raw":
                        clf = base
                        clf.fit(Xtr_bal, ytr_bal)
                    else:
                        clf = CalibratedClassifierCV(base, method=cal, cv=3)
                        clf.fit(Xtr_bal, ytr_bal)
                    yprob = clf.predict_proba(XteH)
                    brier = multiclass_brier(yte_bin, yprob)
                    ece, mce, curve = ece_mce_ovr(yte_bin, yprob)
                except Exception as e:
                    print(f"  ✗ {mname}/{cal} seed={run_id} HATA: {e}")
                    continue
                append_csv([{"model": mname, "calibration": cal, "run_id": run_id,
                             "brier": brier, "ece": ece, "mce": mce}], F_ALL)
                # reliability eğrisi sadece seed 0'da kaydet (temsili, dosya şişmesin)
                if run_id == 0:
                    for row in curve:
                        row.update({"model": mname, "calibration": cal})
                    append_csv(curve, F_CURVE)
                print(f"  seed={run_id} {mname:<18} {cal:<9} "
                      f"Brier={brier:.4f} ECE={ece:.4f} ({time.time()-t0:.0f}s)")
    aggregate()

def aggregate():
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
    df = pd.read_csv(F_ALL)
    summ = df.groupby(["model", "calibration"]).agg(
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        mce_mean=("mce", "mean"), mce_std=("mce", "std"),
    ).round(4).reset_index()
    summ.to_csv(F_SUMMARY, index=False)
    print(f"  ✓ {F_SUMMARY}")

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("="*70 + "\nKALİBRASYON ANALİZİ — ann-thyroid\n")
        f.write("Brier & ECE: düşük = iyi kalibrasyon. raw vs sigmoid vs isotonic.\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Model':<18}{'Cal':<10}{'Brier':<16}{'ECE':<16}{'MCE':<16}\n" + "-"*70 + "\n")
        for _, r in summ.iterrows():
            f.write(f"{r['model']:<18}{r['calibration']:<10}"
                    f"{r['brier_mean']:.4f}±{r['brier_std']:.4f}  "
                    f"{r['ece_mean']:.4f}±{r['ece_std']:.4f}  "
                    f"{r['mce_mean']:.4f}±{r['mce_std']:.4f}\n")
        # her model için en iyi kalibrasyon modu (en düşük ECE)
        f.write("\n\nEN İYİ KALİBRASYON MODU (en düşük ECE, model başına):\n" + "-"*70 + "\n")
        for m in summ["model"].unique():
            sub = summ[summ.model == m]
            best = sub.loc[sub["ece_mean"].idxmin()]
            raw_ece = sub[sub.calibration == "raw"]["ece_mean"].values
            raw_str = f"{raw_ece[0]:.4f}" if len(raw_ece) else "n/a"
            f.write(f"  {m:<18} en iyi={best['calibration']:<9} ECE={best['ece_mean']:.4f}  "
                    f"(raw ECE={raw_str})\n")
    print(f"  ✓ {F_REPORT}")
    print("\nÖzet (raw ECE, model başına):")
    for m in summ["model"].unique():
        raw = summ[(summ.model == m) & (summ.calibration == "raw")]
        if len(raw):
            print(f"  {m:<18} raw ECE={raw['ece_mean'].values[0]:.4f} "
                  f"Brier={raw['brier_mean'].values[0]:.4f}")

if __name__ == "__main__":
    run_all()
    print("\n" + "="*60 + "\nDONE\n" + "="*60)
