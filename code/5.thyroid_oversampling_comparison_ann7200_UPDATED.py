# ============================================================
# SCRIPT 6 — Dengesizlik Ele Alma Stratejisi Karşılaştırması
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumu: R1-33 — "SMOTE'u Borderline-SMOTE, ADASYN, RandomOverSampler,
#   SMOTEENN, class weighting ve cost-sensitive learning ile karşılaştırın."
#
# NE YAPAR:
#   HFS ile seçilen 8 özellik sabit tutulur; yalnızca DENGESİZLİK STRATEJİSİ değişir.
#   Her strateji leakage-free pipeline içinde (train-fold) uygulanır, 10 split üzerinde
#   temsili modellerle (LR, RF, XGB) değerlendirilir.
#
# STRATEJİLER:
#   1. None (baseline, dengeleme yok)
#   2. SMOTE (önerilen)
#   3. Borderline-SMOTE
#   4. ADASYN
#   5. RandomOverSampler
#   6. SMOTEENN (hibrit over+under)
#   7. Class Weighting (class_weight='balanced')
#   8. Cost-Sensitive (sample_weight ile, class-frekans tersi)
#
# NOT: 1-6 resampling; 7-8 algoritma-seviyesi (resampling YOK, ağırlık VAR).
#      Bu ayrım pipeline kurulumunda ele alınır.
#
# ÇIKTI:
#   oversampling_all_runs.csv, oversampling_summary.csv,
#   oversampling_perclass.csv, oversampling_report.txt
#
# RESUME: (strateji, model, seed) bazında.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_oversampling_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_ALL      = os.path.join(OUTPUT_DIR, "oversampling_all_runs.csv")
F_SUMMARY  = os.path.join(OUTPUT_DIR, "oversampling_summary.csv")
F_PERCLASS = os.path.join(OUTPUT_DIR, "oversampling_perclass.csv")
F_REPORT   = os.path.join(OUTPUT_DIR, "oversampling_report.txt")

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

# Stratejiler: (ad, resampler-veya-None, use_class_weight, use_sample_weight)
def make_resampler(name):
    if name == "SMOTE":             return SMOTE(random_state=INTERNAL_SEED)
    if name == "BorderlineSMOTE":   return BorderlineSMOTE(random_state=INTERNAL_SEED)
    if name == "ADASYN":            return ADASYN(random_state=INTERNAL_SEED)
    if name == "RandomOverSampler": return RandomOverSampler(random_state=INTERNAL_SEED)
    if name == "SMOTEENN":          return SMOTEENN(random_state=INTERNAL_SEED)
    return None

STRATEGIES = [
    ("None",             "resample"),
    ("SMOTE",            "resample"),
    ("BorderlineSMOTE",  "resample"),
    ("ADASYN",           "resample"),
    ("RandomOverSampler","resample"),
    ("SMOTEENN",         "resample"),
    ("ClassWeight",      "weight"),
    ("CostSensitive",    "sample_weight"),
]

def get_models(class_weight=None):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=class_weight,
                                                  random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight=class_weight,
                                                     random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
    }

PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}
# ağırlık modunda pipeline'da "model__" prefix yok (resampler yok)
PARAM_GRIDS_NOPREFIX = {k: {p.replace("model__", ""): v for p, v in g.items()}
                        for k, g in PARAM_GRIDS.items()}
SCORING_CV = 'roc_auc_ovr_weighted'

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
# Metrik
# ============================================================

def per_class_recall(cm):
    """Her sınıf için recall (sensitivity)."""
    out = {}
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]; fn = cm[idx, :].sum() - tp
        out[CLASS_NAMES[idx]] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return out

def macro_specificity(cm):
    total = cm.sum(); specs = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]; fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp; tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specs)

# ============================================================
# Değerlendirme (tek strateji, tek model, tek seed)
# ============================================================

def evaluate(strategy, kind, mname, XtrH, ytr, XteH, yte):
    labels = sorted(yte.unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

    if kind == "resample":
        resampler = make_resampler(strategy)
        model = get_models(class_weight=None)[mname]
        if resampler is None:  # "None" baseline
            pipe = ImbPipeline([("model", model)])
        else:
            pipe = ImbPipeline([("resampler", resampler), ("model", model)])
        grid = GridSearchCV(pipe, PARAM_GRIDS[mname], cv=cv, scoring=SCORING_CV, n_jobs=-1)
        grid.fit(XtrH, ytr)
        best = grid.best_estimator_
        yprob = best.predict_proba(XteH); ypred = best.predict(XteH)

    elif kind == "weight":
        model = get_models(class_weight="balanced")[mname]
        grid = GridSearchCV(model, PARAM_GRIDS_NOPREFIX[mname], cv=cv, scoring=SCORING_CV, n_jobs=-1)
        grid.fit(XtrH, ytr)
        best = grid.best_estimator_
        yprob = best.predict_proba(XteH); ypred = best.predict(XteH)

    elif kind == "sample_weight":
        # cost-sensitive: sınıf frekansı tersi ağırlık
        classes, counts = np.unique(ytr, return_counts=True)
        cw = {c: len(ytr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sw = np.array([cw[v] for v in ytr])
        model = get_models(class_weight=None)[mname]
        grid = GridSearchCV(model, PARAM_GRIDS_NOPREFIX[mname], cv=cv, scoring=SCORING_CV, n_jobs=-1)
        grid.fit(XtrH, ytr, sample_weight=sw)
        best = grid.best_estimator_
        yprob = best.predict_proba(XteH); ypred = best.predict(XteH)

    roc = roc_auc_score(yte, yprob, multi_class='ovr', average='weighted')
    f1m = f1_score(yte, ypred, average='macro', zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=labels)
    spec = macro_specificity(cm)
    pcr = per_class_recall(cm)
    return roc, f1m, spec, pcr

# ============================================================
# Resume
# ============================================================

def load_done():
    if not os.path.exists(F_ALL):
        return set()
    df = pd.read_csv(F_ALL)
    return set(zip(df["strategy"], df["model"], df["run_id"]))

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
    print("DENGESİZLİK STRATEJİSİ KARŞILAŞTIRMASI — ann-thyroid")
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek")
    print(f"Stratejiler: {[s[0] for s in STRATEGIES]}")
    print(f"Modeller: {list(get_models().keys())}")
    print("="*60)
    done = load_done()
    if done:
        print(f"[RESUME] {len(done)} (strateji,model,seed) tamam")

    for run_id in SPLIT_SEEDS:
        X_raw = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X_raw, y, test_size=0.2, random_state=run_id, stratify=y)
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)
        feats = hfs_select(Xtr, ytr.values)
        XtrH, XteH = Xtr[feats], Xte[feats]

        for strategy, kind in STRATEGIES:
            for mname in get_models().keys():
                if (strategy, mname, run_id) in done:
                    continue
                t0 = time.time()
                try:
                    roc, f1m, spec, pcr = evaluate(strategy, kind, mname, XtrH, ytr, XteH, yte)
                except Exception as e:
                    print(f"  ✗ {strategy}/{mname} seed={run_id} HATA: {e}")
                    continue
                append_csv([{"strategy": strategy, "model": mname, "run_id": run_id,
                             "roc_auc_ovr": roc, "f1_macro": f1m, "specificity_macro": spec,
                             "recall_overt_hypo": pcr["overt_hypothyroid"],
                             "recall_subclin_hypo": pcr["subclinical_hypothyroid"],
                             "recall_euthyroid": pcr["euthyroid"]}], F_ALL)
                print(f"  seed={run_id} {strategy:<17} {mname:<18} "
                      f"ROC={roc:.4f} F1m={f1m:.4f} ({time.time()-t0:.0f}s)")
    aggregate()

def aggregate():
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
    df = pd.read_csv(F_ALL)
    # strateji × model özet
    summ = df.groupby(["strategy", "model"]).agg(
        roc_mean=("roc_auc_ovr", "mean"), roc_std=("roc_auc_ovr", "std"),
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        spec_mean=("specificity_macro", "mean"), spec_std=("specificity_macro", "std"),
    ).round(4).reset_index()
    summ.to_csv(F_SUMMARY, index=False)
    print(f"  ✓ {F_SUMMARY}")

    # per-class recall özet (azınlık sınıf duyarlılığı — asıl klinik önem)
    pc = df.groupby("strategy").agg(
        recall_overt_hypo_mean=("recall_overt_hypo", "mean"), recall_overt_hypo_std=("recall_overt_hypo", "std"),
        recall_subclin_hypo_mean=("recall_subclin_hypo", "mean"), recall_subclin_hypo_std=("recall_subclin_hypo", "std"),
        recall_euthyroid_mean=("recall_euthyroid", "mean"),
    ).round(4).reset_index()
    pc.to_csv(F_PERCLASS, index=False)
    print(f"  ✓ {F_PERCLASS}")

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("="*70 + "\nDENGESİZLİK STRATEJİSİ KARŞILAŞTIRMASI — ann-thyroid\n")
        f.write("HFS sabit (8 özellik); yalnızca dengeleme stratejisi değişir.\n")
        f.write("="*70 + "\n\n")
        # strateji ortalaması (modeller arası)
        f.write("STRATEJİ ORTALAMASI (tüm modeller):\n" + "-"*70 + "\n")
        f.write(f"{'Strateji':<18}{'ROC-AUC':<12}{'Macro-F1':<12}{'Spec':<12}\n")
        strat_avg = summ.groupby("strategy").agg(
            roc=("roc_mean", "mean"), f1=("f1_mean", "mean"), spec=("spec_mean", "mean")
        ).round(4).sort_values("f1", ascending=False)
        for s, r in strat_avg.iterrows():
            star = "  ← önerilen" if s == "SMOTE" else ""
            f.write(f"{s:<18}{r['roc']:.4f}      {r['f1']:.4f}      {r['spec']:.4f}{star}\n")
        # azınlık sınıf duyarlılığı — R1-33'ün asıl klinik hedefi
        f.write("\n\nAZINLIK SINIF DUYARLILIĞI (recall, strateji başına):\n" + "-"*70 + "\n")
        f.write(f"{'Strateji':<18}{'Overt hypo':<16}{'Subclin. hypo':<16}{'Euthyroid':<12}\n")
        for _, r in pc.iterrows():
            f.write(f"{r['strategy']:<18}"
                    f"{r['recall_overt_hypo_mean']:.4f}±{r['recall_overt_hypo_std']:.4f}  "
                    f"{r['recall_subclin_hypo_mean']:.4f}±{r['recall_subclin_hypo_std']:.4f}  "
                    f"{r['recall_euthyroid_mean']:.4f}\n")
    print(f"  ✓ {F_REPORT}")
    print("\nStrateji ortalaması (Macro-F1):")
    for s, r in strat_avg.iterrows():
        print(f"  {s:<18} F1m={r['f1']:.4f} ROC={r['roc']:.4f} Spec={r['spec']:.4f}")

if __name__ == "__main__":
    run_all()
    print("\n" + "="*60 + "\nDONE\n" + "="*60)
