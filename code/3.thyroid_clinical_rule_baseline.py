# ============================================================
# SCRIPT 7 — Klinik Kural Tabanlı Baseline
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1-59, R1-60  (ayrıca R1-9, R1-62'ye dolaylı cevap)
#   #59: "TSH, T3, TT4 ve FTI eşiklerine dayalı basit klinik kural baseline ekleyin."
#   #60: "Klinik tanı kurallarıyla karşılaştırma olmadan, ML'in pratik değer katıp
#         katmadığını değerlendirmek zor."
#
# NE YAPAR:
#   Endokrinoloji kılavuzlarındaki (ATA/ETA) standart tanı mantığını doğrudan
#   kodlar ve ML pipeline'ı ile AYNI test bölünmeleri üzerinde karşılaştırır.
#   Hiçbir öğrenme yok — saf eşik mantığı.
#
#   KURAL (primer hipotiroidi tanı mantığı):
#     TSH ↑ ve TT4 ↓            → overt hypothyroid       (aşikar)
#     TSH ↑ ve TT4 normal       → subclinical hypothyroid (subklinik)
#     aksi                       → euthyroid               (ötiroid)
#
#   ÖNEMLİ — ölçekleme: ann-thyroid normalize edilmiş biçimde dağıtılır
#   (TSH÷1000, T3÷100, TT4÷1000, T4U÷10, FTI÷1000, age÷100). Klinik eşikler
#   ham birimlerde tanımlı olduğundan, kural uygulanmadan önce ham ölçeğe
#   geri çevrilir. Bu, kuralın klinik olarak yorumlanabilir kalmasını sağlar.
#
#   Ayrıca iki kural varyantı denenir:
#     (a) sabit kılavuz eşikleri (TSH>4.0, TT4<60)
#     (b) train-fold'da optimize edilmiş eşikler (leakage-free; kuralın
#         en iyi haliyle bile ML'e göre nerede durduğunu görmek için)
#
# ÇIKTI:
#   clinical_rule_all_runs.csv, clinical_rule_summary.csv,
#   clinical_rule_vs_ml.csv, clinical_rule_report.txt
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_clinical_rule_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_ALL     = os.path.join(OUTPUT_DIR, "clinical_rule_all_runs.csv")
F_SUMMARY = os.path.join(OUTPUT_DIR, "clinical_rule_summary.csv")
F_VS_ML   = os.path.join(OUTPUT_DIR, "clinical_rule_vs_ml.csv")
F_REPORT  = os.path.join(OUTPUT_DIR, "clinical_rule_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

# Sınıf adları — hormon profiline göre doğrulanmış (bkz. makale §2.1)
CLASS_NAMES = {0: "overt_hypothyroid", 1: "subclinical_hypothyroid", 2: "euthyroid"}

# ann-thyroid normalizasyon katsayıları (ham = normalize × katsayı)
# Veriden doğrulandı: TSH max 0.53→530, T3 0.18→18, TT4 0.60→600, T4U 0.233→2.33, FTI 0.642→642
DENORM = {"TSH": 1000.0, "T3": 100.0, "TT4": 1000.0, "T4U": 10.0, "FTI": 1000.0, "age": 100.0}

# Klinik referans eşikleri (ATA/ETA kılavuz mantığı, ham birimler)
TSH_HIGH   = 4.0    # mIU/L üzeri = yüksek
TT4_LOW    = 60.0   # nmol/L altı = düşük
FTI_LOW    = 65.0   # düşük FTI (ikincil destek)

# Eşik optimizasyonu için aday ızgara (varyant b)
GRID_TSH = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
GRID_TT4 = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]


def denormalize(df):
    """Normalize edilmiş ann-thyroid değerlerini ham klinik birimlere çevirir."""
    out = df.copy()
    for c, k in DENORM.items():
        if c in out.columns:
            out[c] = out[c] * k
    return out


# ============================================================
# KLİNİK KURAL
# ============================================================

def clinical_rule_predict(df_raw, tsh_high=TSH_HIGH, tt4_low=TT4_LOW):
    """
    Ham (denormalize) hormon değerleri üzerinde klinik tanı kuralı.
    Dönüş: 0=overt hypothyroid, 1=subclinical hypothyroid, 2=euthyroid
    """
    tsh = df_raw["TSH"].values
    tt4 = df_raw["TT4"].values
    pred = np.full(len(df_raw), 2, dtype=int)          # varsayılan: euthyroid
    tsh_elevated = tsh > tsh_high
    tt4_reduced = tt4 < tt4_low
    pred[tsh_elevated & tt4_reduced] = 0               # aşikar hipotiroidi
    pred[tsh_elevated & ~tt4_reduced] = 1              # subklinik hipotiroidi
    return pred


def optimize_thresholds(df_raw_train, y_train):
    """Train-fold üzerinde macro-F1'i maksimize eden eşikleri arar (leakage-free)."""
    best = (-1.0, TSH_HIGH, TT4_LOW)
    for t in GRID_TSH:
        for q in GRID_TT4:
            pred = clinical_rule_predict(df_raw_train, tsh_high=t, tt4_low=q)
            f1 = f1_score(y_train, pred, average="macro", zero_division=0)
            if f1 > best[0]:
                best = (f1, t, q)
    return best[1], best[2], best[0]


# ============================================================
# Metrikler
# ============================================================

def per_class_metrics(cm):
    """Her sınıf için sens/spec döndürür."""
    total = cm.sum()
    out = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out[f"sens_{CLASS_NAMES[i]}"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[f"spec_{CLASS_NAMES[i]}"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return out


def macro_specificity(cm):
    total = cm.sum(); specs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp; tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


# ============================================================
# HFS (ML kolu için — makale ile birebir)
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
# ML kolu (karşılaştırma için — HFS+SMOTE, temsili modeller)
# ============================================================

def get_ml_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
    }

ML_GRIDS = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}


def append_csv(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_done():
    if not os.path.exists(F_ALL):
        return set()
    df = pd.read_csv(F_ALL)
    return set(zip(df["method"], df["run_id"]))


# ============================================================
# Ana döngü
# ============================================================

def run_all():
    print("=" * 64)
    print("KLİNİK KURAL BASELINE vs ML — ann-thyroid")
    print("=" * 64)
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek")
    print(f"Kural: TSH>{TSH_HIGH} & TT4<{TT4_LOW} → overt | TSH>{TSH_HIGH} & TT4≥{TT4_LOW} → subclinical | else euthyroid")
    print(f"Denormalizasyon: {DENORM}")
    print("=" * 64)

    done = load_done()
    if done:
        print(f"[RESUME] {len(done)} (yöntem,seed) tamam")

    for run_id in SPLIT_SEEDS:
        X_raw_norm = data[FEATURE_COLS]
        y = data[TARGET_COL] - 1
        Xtr_n, Xte_n, ytr, yte = train_test_split(
            X_raw_norm, y, test_size=0.2, random_state=run_id, stratify=y)

        # --- Klinik kural kolu: ham ölçeğe çevir (ölçekleme YOK, kural ham değer ister)
        Xtr_raw = denormalize(Xtr_n)
        Xte_raw = denormalize(Xte_n)

        # (a) sabit kılavuz eşikleri
        if ("ClinicalRule_fixed", run_id) not in done:
            t0 = time.time()
            pred = clinical_rule_predict(Xte_raw)
            cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
            row = {"method": "ClinicalRule_fixed", "run_id": run_id,
                   "accuracy": accuracy_score(yte, pred),
                   "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                   "specificity_macro": macro_specificity(cm),
                   "roc_auc_ovr": np.nan,   # kural olasılık üretmez
                   "tsh_thr": TSH_HIGH, "tt4_thr": TT4_LOW,
                   "elapsed_s": round(time.time() - t0, 3)}
            row.update(per_class_metrics(cm))
            append_csv([row], F_ALL)
            print(f"  seed={run_id} ClinicalRule_fixed      "
                  f"F1m={row['f1_macro']:.4f} Spec={row['specificity_macro']:.4f} "
                  f"({row['elapsed_s']}s)")

        # (b) train-fold'da optimize edilmiş eşikler (leakage-free)
        if ("ClinicalRule_tuned", run_id) not in done:
            t0 = time.time()
            t_opt, q_opt, f1_tr = optimize_thresholds(Xtr_raw, ytr)
            pred = clinical_rule_predict(Xte_raw, tsh_high=t_opt, tt4_low=q_opt)
            cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
            row = {"method": "ClinicalRule_tuned", "run_id": run_id,
                   "accuracy": accuracy_score(yte, pred),
                   "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                   "specificity_macro": macro_specificity(cm),
                   "roc_auc_ovr": np.nan,
                   "tsh_thr": t_opt, "tt4_thr": q_opt,
                   "elapsed_s": round(time.time() - t0, 3)}
            row.update(per_class_metrics(cm))
            append_csv([row], F_ALL)
            print(f"  seed={run_id} ClinicalRule_tuned      "
                  f"F1m={row['f1_macro']:.4f} Spec={row['specificity_macro']:.4f} "
                  f"(TSH>{t_opt}, TT4<{q_opt}) ({row['elapsed_s']}s)")

        # --- ML kolu: standart HFS+SMOTE pipeline (aynı split)
        scaler = StandardScaler()
        Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr_n), columns=FEATURE_COLS)
        Xte_s = pd.DataFrame(scaler.transform(Xte_n), columns=FEATURE_COLS)
        feats = hfs_select(Xtr_s, ytr.values)
        XtrH, XteH = Xtr_s[feats], Xte_s[feats]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

        for mname, model in get_ml_models().items():
            tag = f"ML_{mname}"
            if (tag, run_id) in done:
                continue
            t0 = time.time()
            try:
                pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)),
                                    ("model", model)])
                grid = GridSearchCV(pipe, ML_GRIDS[mname], cv=cv,
                                    scoring="roc_auc_ovr_weighted", n_jobs=-1)
                grid.fit(XtrH, ytr)
                best = grid.best_estimator_
                pred = best.predict(XteH)
                prob = best.predict_proba(XteH)
                cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
                row = {"method": tag, "run_id": run_id,
                       "accuracy": accuracy_score(yte, pred),
                       "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                       "specificity_macro": macro_specificity(cm),
                       "roc_auc_ovr": roc_auc_score(yte, prob, multi_class="ovr",
                                                    average="weighted"),
                       "tsh_thr": np.nan, "tt4_thr": np.nan,
                       "elapsed_s": round(time.time() - t0, 2)}
                row.update(per_class_metrics(cm))
                append_csv([row], F_ALL)
                print(f"  seed={run_id} {tag:<24} "
                      f"F1m={row['f1_macro']:.4f} Spec={row['specificity_macro']:.4f} "
                      f"({row['elapsed_s']}s)")
            except Exception as e:
                print(f"  ✗ {tag} seed={run_id} HATA: {e}")
    aggregate()


# ============================================================
# Aggregate + rapor
# ============================================================

def aggregate():
    print(f"\n{'=' * 64}\nAGGREGATE\n{'=' * 64}")
    df = pd.read_csv(F_ALL)

    metric_cols = ["accuracy", "f1_macro", "specificity_macro", "roc_auc_ovr"]
    per_cls = [c for c in df.columns if c.startswith(("sens_", "spec_"))]

    agg = df.groupby("method")[metric_cols + per_cls].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(F_SUMMARY, index=False)
    print(f"  ✓ {F_SUMMARY}")

    # kural vs ML doğrudan karşılaştırma
    rule = df[df.method.str.startswith("ClinicalRule")]
    ml = df[df.method.str.startswith("ML_")]
    cmp_rows = []
    for rmethod in rule["method"].unique():
        r = rule[rule.method == rmethod]
        for mmethod in ml["method"].unique():
            m = ml[ml.method == mmethod]
            cmp_rows.append({
                "rule": rmethod, "ml_model": mmethod,
                "rule_f1_macro": round(r["f1_macro"].mean(), 4),
                "ml_f1_macro": round(m["f1_macro"].mean(), 4),
                "delta_f1_macro": round(m["f1_macro"].mean() - r["f1_macro"].mean(), 4),
                "rule_spec": round(r["specificity_macro"].mean(), 4),
                "ml_spec": round(m["specificity_macro"].mean(), 4),
                "delta_spec": round(m["specificity_macro"].mean() - r["specificity_macro"].mean(), 4),
            })
    pd.DataFrame(cmp_rows).to_csv(F_VS_ML, index=False)
    print(f"  ✓ {F_VS_ML}")

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\nKLİNİK KURAL BASELINE vs ML — ann-thyroid\n")
        f.write("R1-59: TSH/TT4 eşiklerine dayalı klinik kural baseline\n")
        f.write("R1-60: ML'in klinik kurala göre kattığı pratik değer\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"KURAL: TSH>{TSH_HIGH} mIU/L & TT4<{TT4_LOW} nmol/L → overt hypothyroid\n")
        f.write(f"       TSH>{TSH_HIGH} & TT4≥{TT4_LOW}              → subclinical hypothyroid\n")
        f.write(f"       aksi                                     → euthyroid\n\n")

        f.write(f"{'Yöntem':<26}{'Macro-F1':<18}{'Specificity':<18}{'Accuracy':<16}\n")
        f.write("-" * 72 + "\n")
        for _, r in agg.iterrows():
            f.write(f"{r['method']:<26}"
                    f"{r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}   "
                    f"{r['specificity_macro_mean']:.4f}±{r['specificity_macro_std']:.4f}   "
                    f"{r['accuracy_mean']:.4f}\n")

        f.write("\n\nSINIF BAZLI DUYARLILIK (sensitivity):\n" + "-" * 72 + "\n")
        f.write(f"{'Yöntem':<26}{'Overt hypo':<16}{'Subclin. hypo':<16}{'Euthyroid':<14}\n")
        for _, r in agg.iterrows():
            f.write(f"{r['method']:<26}"
                    f"{r.get('sens_overt_hypothyroid_mean', np.nan):<16.4f}"
                    f"{r.get('sens_subclinical_hypothyroid_mean', np.nan):<16.4f}"
                    f"{r.get('sens_euthyroid_mean', np.nan):<14.4f}\n")

        f.write("\n\nML'İN KURALA GÖRE KATKISI (ΔMacro-F1):\n" + "-" * 72 + "\n")
        for row in cmp_rows:
            f.write(f"  {row['ml_model']:<24} vs {row['rule']:<22} "
                    f"ΔF1={row['delta_f1_macro']:+.4f}  ΔSpec={row['delta_spec']:+.4f}\n")

        # optimize edilmiş eşiklerin dağılımı
        tuned = df[df.method == "ClinicalRule_tuned"]
        if len(tuned):
            f.write("\n\nOPTİMİZE EDİLEN EŞİKLER (train-fold, run başına):\n" + "-" * 72 + "\n")
            f.write(f"  TSH eşiği: medyan={tuned['tsh_thr'].median():.1f}  "
                    f"aralık=[{tuned['tsh_thr'].min():.1f}, {tuned['tsh_thr'].max():.1f}]\n")
            f.write(f"  TT4 eşiği: medyan={tuned['tt4_thr'].median():.1f}  "
                    f"aralık=[{tuned['tt4_thr'].min():.1f}, {tuned['tt4_thr'].max():.1f}]\n")

        f.write("\n\nHESAPLAMA MALİYETİ (ortalama, saniye):\n" + "-" * 72 + "\n")
        for m in df["method"].unique():
            f.write(f"  {m:<26}{df[df.method == m]['elapsed_s'].mean():.3f}s\n")

    print(f"  ✓ {F_REPORT}")

    print("\nÖZET (Macro-F1):")
    for _, r in agg.sort_values("f1_macro_mean", ascending=False).iterrows():
        print(f"  {r['method']:<26} F1m={r['f1_macro_mean']:.4f}  "
              f"Spec={r['specificity_macro_mean']:.4f}")


if __name__ == "__main__":
    run_all()
    print("\n" + "=" * 64 + "\nDONE\n" + "=" * 64)
