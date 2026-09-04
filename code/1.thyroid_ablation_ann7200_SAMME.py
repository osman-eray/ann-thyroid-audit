# ============================================================
# SCRIPT 1 — Faktöriyel Ablasyon + Zengin Metrikler
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1-7,8,12,13,14,15,45,50,51,52,55,56,57,58 · R4-ek · R5-1
#
# 4 KOŞUL (faktöriyel ablasyon):
#   A = Baseline        (HFS yok,  SMOTE yok)   — 21 özellik
#   C = HFS only        (HFS var,  SMOTE yok)   —  8 özellik
#   D = SMOTE only      (HFS yok,  SMOTE var)   — 21 özellik
#   B = HFS + SMOTE     (HFS var,  SMOTE var)   —  8 özellik
#
# ZENGİN METRİKLER (her model, her koşul):
#   - Aggregate: accuracy, precision/recall/F1 (weighted), specificity (macro), ROC-AUC (OVR)
#   - Macro-F1  (imbalance için R1-56)
#   - Per-class: sensitivity, specificity, precision, F1, PR-AUC  (class 0/1/2)
#   - Naïve majority-class baseline (DummyClassifier) — R1-58
#
# AGGREGATE:
#   - %95 güven aralıkları (R1-45,50)
#   - Paired Wilcoxon: C-vs-A (saf HFS), D-vs-A (saf SMOTE),
#                      B-vs-D (HFS artımsal), B-vs-A (toplam)
#                      hem ROC-AUC hem Specificity üzerinde
#
# ÖZELLİKLER: Resume mekanizması korundu. Kesintide kaldığı yerden devam.
# Tahmini süre: i5-3210M'de ~gecelik (4 koşul × 10 model × 10 run × tam grid)
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier
)
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon

# ============================================================
# Çoklu-test düzeltmesi (R1-47) — saf numpy, harici bağımlılık yok
# Holm-Bonferroni (FWER) ve Benjamini-Hochberg (FDR)
# ============================================================

def holm_bonferroni(pvals):
    """Holm-Bonferroni düzeltilmiş p-değerleri döndürür. NaN'ları korur."""
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pv[idx]
        running = max(running, val)      # monoton artan zorla
        adj[idx] = min(running, 1.0)
    out[valid] = adj
    return out

def benjamini_hochberg(pvals):
    """Benjamini-Hochberg (FDR) düzeltilmiş p-değerleri. NaN'ları korur."""
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 1.0
    # büyükten küçüğe (step-up)
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = pv[idx] * m / (rank + 1)
        running = min(running, val)      # monoton azalan zorla (tersten)
        adj[idx] = min(running, 1.0)
    out[valid] = adj
    return out


# ============================================================
# CONFIG
# ============================================================
# ============================================================
# DEADLOCK ÖNLEMİ (Windows / Python 3.11 / joblib-loky)
# Sorun: GridSearchCV(n_jobs=-1) + Stacking'in iç CV'si → iç içe process
#        paralelliği → joblib TASK_PENDING kilidi (CPU %100 ama ilerleme yok).
# Çözüm: Stacking için dış paralellik kapalı (kodda otomatik).
# Eğer YİNE takılırsa: USE_THREADING = True yap (process yerine thread kullanır,
#   deadlock olmaz; biraz daha yavaş ama güvenli).
# ============================================================
USE_THREADING = False   # takılma devam ederse True yap

if USE_THREADING:
    from joblib import parallel_backend
    import atexit
    _backend_ctx = parallel_backend('threading')
    _backend_ctx.__enter__()
    atexit.register(lambda: _backend_ctx.__exit__(None, None, None))
    print("[CONFIG] joblib backend = threading (deadlock önlemi aktif)")

N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))     # 0, 1, ..., 9
INTERNAL_SEED = 42

DATA_PATH  = "thyroid_ann_7200.csv"   # ← Veri seti yolu (gerekirse değiştirin)
OUTPUT_DIR = "thyroid_ablation_results"

F_ALL_RUNS  = os.path.join(OUTPUT_DIR, "ablation_all_runs.csv")
F_HFS_LOG   = os.path.join(OUTPUT_DIR, "hfs_feature_log.csv")
F_SUMMARY   = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
F_WILCOXON  = os.path.join(OUTPUT_DIR, "ablation_wilcoxon.csv")
F_PERCLASS  = os.path.join(OUTPUT_DIR, "perclass_summary.csv")
F_REPORT    = os.path.join(OUTPUT_DIR, "ablation_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'
CLASS_NAMES = {0: "hyperthyroid", 1: "subnormal", 2: "normal"}

# HFS parameters (makale ile aynı)
HFS_THETA = 0.9
HFS_W     = 0.5
HFS_K     = 8

# 4 koşulun tanımı: (etiket, hfs_kullan, smote_kullan)
CONDITIONS = [
    ("A_baseline",  False, False),
    ("C_HFSonly",   True,  False),
    ("D_SMOTEonly", False, True),
    ("B_HFS_SMOTE", True,  True),
]

# ============================================================
# HFS — Hybrid Feature Selection (makale ile birebir aynı)
# ============================================================

def pearson_correlation(f1, f2):
    return np.corrcoef(f1, f2)[0, 1]

def information_gain(X_col, y):
    return mutual_info_classif(
        X_col.reshape(-1, 1), y,
        discrete_features='auto',
        random_state=INTERNAL_SEED
    )[0]

def gain_ratio(X_col, y):
    IG = information_gain(X_col, y)
    n_unique = len(np.unique(X_col))
    if n_unique <= 10:
        _, counts = np.unique(X_col, return_counts=True)
    else:
        n_bins = max(10, int(np.sqrt(len(X_col))))
        counts = np.histogram(X_col, bins=n_bins)[0]
        counts = counts[counts > 0]
    probs = counts / counts.sum()
    Hx = -np.sum(probs * np.log2(probs))
    return IG / (Hx + 1e-12) if Hx > 0 else 0.0

def hybrid_feature_selection(df, target_col, theta=HFS_THETA, w=HFS_W, k=HFS_K):
    features = [c for c in df.columns if c != target_col]
    X = df[features].values
    y = df[target_col].values

    # Stage 1: Redundancy Elimination (Pearson)
    redundant = set()
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(pearson_correlation(X[:, i], X[:, j])) > theta:
                redundant.add(features[j])

    non_redundant = [f for f in features if f not in redundant]

    # Stage 2-3: IG, GR, Hybrid Score
    feature_scores = []
    for f in non_redundant:
        X_f = df[f].values
        ig = information_gain(X_f, y)
        gr = gain_ratio(X_f, y)
        hs = w * ig + (1 - w) * gr
        feature_scores.append((f, ig, gr, hs))

    # Stage 4: Top-k selection
    feature_scores.sort(key=lambda x: x[3], reverse=True)
    selected = [f for f, ig, gr, hs in feature_scores[:k]]
    return selected, feature_scores

# ============================================================
# Models & grids (makale ile birebir aynı)
# ============================================================

# ============================================================
# SÜRÜM NOTU — AdaBoost algoritması (ÖNEMLİ)
# ------------------------------------------------------------
# scikit-learn 1.6, AdaBoost'un varsayılan algoritmasını SAMME.R'den
# SAMME'ye çevirdi ve SAMME.R'yi tamamen kaldırdı. Bu, aynı kodun
# sklearn<1.6 ve >=1.6 ortamlarında FARKLI sonuç vermesine yol açar
# (bu veri setinde AdaBoost ROC-AUC farkı ~0.006).
#
# Bu nedenle algorithm="SAMME" AÇIKÇA belirtilir:
#   - sklearn 1.3'te de 1.9'da da aynı sonucu verir
#   - yayınlanan kod her sürümde reprodüksiyon yapılabilir kalır (R1-87)
#   - figür üretim scripti de aynı ayarı kullanır → tablo-figür tutarlılığı
# ============================================================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=INTERNAL_SEED),
        "KNN":                 KNeighborsClassifier(),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, C=1, max_iter=200000,
                                   decision_function_shape='ovr', random_state=INTERNAL_SEED),
        "SVM (Linear)":        SVC(kernel="linear", probability=True, C=1, max_iter=200000,
                                   decision_function_shape='ovr', random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, algorithm="SAMME",
                                                random_state=INTERNAL_SEED),
        "Bagging":             BaggingClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
        "Stacking (RF+SVM)":   StackingClassifier(
            estimators=[
                ('rf',  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=INTERNAL_SEED)),
                ('svm', SVC(kernel="rbf", probability=True, C=1, max_iter=200000,
                            random_state=INTERNAL_SEED))
            ],
            final_estimator=LogisticRegression(random_state=INTERNAL_SEED),
            cv=3,        # PERF: iç CV 5→3
            n_jobs=1     # PERF: iç içe paralelliği kır
        ),
    }

PARAM_GRIDS_PLAIN = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"]},
    "Decision Tree":       {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "KNN":                 {"n_neighbors": [3, 5, 7, 9, 11], "metric": ["euclidean", "manhattan"]},
    "SVM (RBF)":           {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "SVM (Linear)":        {"C": [0.01, 0.1, 1, 10]},
    "Random Forest":       {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None]},
    "AdaBoost":            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
    "Bagging":             {"n_estimators": [50, 100, 200], "max_samples": [0.5, 0.7, 1.0]},
    "XGBoost":             {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
                            "learning_rate": [0.01, 0.1, 0.3]},
    "Stacking (RF+SVM)":   {"rf__n_estimators": [100], "rf__max_depth": [5, 10]},  # PERF 4→2
}
# SMOTE'lu pipeline için model__ prefix'li grid
PARAM_GRIDS_SMOTE = {k: {f"model__{p}": v for p, v in grid.items()}
                     for k, grid in PARAM_GRIDS_PLAIN.items()}

SCORING_CV = 'roc_auc_ovr_weighted'

# ============================================================
# GridSearch — koşula göre (SMOTE var/yok)
# ============================================================

def run_gridsearch(models, X_train, y_train, use_smote):
    best_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
    for name, model in models.items():
        print(f"      · GS {name} ...", end="", flush=True)
        _t = time.time()
        # DEADLOCK FIX: Stacking kendi iç CV'sini koşar (iç içe paralellik).
        # Windows/loky'de GridSearchCV(n_jobs=-1) + iç paralellik → TASK_PENDING kilidi.
        # Stacking için dış paralelliği kapat; diğer modeller paralel kalsın.
        gs_jobs = 1 if "Stacking" in name else -1
        if use_smote:
            pipe = ImbPipeline([
                ("smote", SMOTE(random_state=INTERNAL_SEED)),
                ("model", model)
            ])
            grid = GridSearchCV(pipe, PARAM_GRIDS_SMOTE[name],
                                cv=cv, scoring=SCORING_CV, n_jobs=gs_jobs)
        else:
            grid = GridSearchCV(model, PARAM_GRIDS_PLAIN[name],
                                cv=cv, scoring=SCORING_CV, n_jobs=gs_jobs)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f" {time.time()-_t:.0f}s", flush=True)
    return best_models

def run_cv(best_models, X_train, y_train):
    cv_results = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=INTERNAL_SEED)
    for name, model in best_models.items():
        print(f"      · CV {name} ...", end="", flush=True)
        _t = time.time()
        cv_jobs = 1 if "Stacking" in name else -1   # DEADLOCK FIX (aynı gerekçe)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring=SCORING_CV, n_jobs=cv_jobs)
        cv_results[name] = scores
        print(f" {time.time()-_t:.0f}s", flush=True)
    return cv_results

# ============================================================
# Metrik yardımcıları
# ============================================================

def per_class_metrics(cm, labels):
    """3×3 confusion matrix'ten her sınıf için sens/spec/prec/f1 döndür."""
    out = {}
    total = cm.sum()
    for idx, c in enumerate(labels):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall/sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        out[c] = {"sens": sens, "spec": spec, "prec": prec, "f1": f1}
    return out

def macro_specificity_from_cm(cm, labels):
    return np.mean([per_class_metrics(cm, labels)[c]["spec"] for c in labels])

# ============================================================
# Değerlendirme — zengin metrikler
# ============================================================

def evaluate_models(best_models, X_train, y_train, X_test, y_test,
                    cv_results, condition_label, run_id):
    rows = []
    labels = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=labels)  # (n, 3)

    for name, model in best_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)  # R1-56

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        spec_macro = macro_specificity_from_cm(cm, labels)
        pcm = per_class_metrics(cm, labels)

        # ROC-AUC (OVR weighted) + per-class PR-AUC
        try:
            y_prob = model.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        except Exception:
            y_prob = None
            roc = np.nan

        # PR-AUC (average precision) per class — R1-8
        pr_auc = {}
        for idx, c in enumerate(labels):
            if y_prob is not None:
                try:
                    pr_auc[c] = average_precision_score(y_test_bin[:, idx], y_prob[:, idx])
                except Exception:
                    pr_auc[c] = np.nan
            else:
                pr_auc[c] = np.nan

        row = {
            "run_id": run_id,
            "condition": condition_label,
            "model": name,
            "n_features": X_train.shape[1],
            "accuracy": acc,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "specificity_macro": spec_macro,
            "f1_weighted": f1_w,
            "f1_macro": f1_macro,
            "roc_auc_ovr": roc,
            "cv_mean": cv_results[name].mean(),
            "cv_std": cv_results[name].std(),
            # Confusion matrix flatten
            "cm_00": int(cm[0,0]), "cm_01": int(cm[0,1]), "cm_02": int(cm[0,2]),
            "cm_10": int(cm[1,0]), "cm_11": int(cm[1,1]), "cm_12": int(cm[1,2]),
            "cm_20": int(cm[2,0]), "cm_21": int(cm[2,1]), "cm_22": int(cm[2,2]),
        }
        # Per-class metrikler (0=hyper, 1=subnormal, 2=normal)
        for c in labels:
            cn = CLASS_NAMES[c]
            row[f"sens_{cn}"]   = pcm[c]["sens"]
            row[f"spec_{cn}"]   = pcm[c]["spec"]
            row[f"prec_{cn}"]   = pcm[c]["prec"]
            row[f"f1_{cn}"]     = pcm[c]["f1"]
            row[f"prauc_{cn}"]  = pr_auc[c]

        rows.append(row)
    return rows

def evaluate_naive_baseline(X_train, y_train, X_test, y_test, run_id):
    """Majority-class DummyClassifier — R1-58."""
    labels = sorted(y_test.unique())
    dummy = DummyClassifier(strategy="most_frequent", random_state=INTERNAL_SEED)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pcm = per_class_metrics(cm, labels)
    row = {
        "run_id": run_id, "condition": "NAIVE_majority", "model": "DummyClassifier",
        "n_features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "specificity_macro": macro_specificity_from_cm(cm, labels),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "roc_auc_ovr": np.nan, "cv_mean": np.nan, "cv_std": np.nan,
        "cm_00": int(cm[0,0]), "cm_01": int(cm[0,1]), "cm_02": int(cm[0,2]),
        "cm_10": int(cm[1,0]), "cm_11": int(cm[1,1]), "cm_12": int(cm[1,2]),
        "cm_20": int(cm[2,0]), "cm_21": int(cm[2,1]), "cm_22": int(cm[2,2]),
    }
    for c in labels:
        cn = CLASS_NAMES[c]
        row[f"sens_{cn}"] = pcm[c]["sens"]; row[f"spec_{cn}"] = pcm[c]["spec"]
        row[f"prec_{cn}"] = pcm[c]["prec"]; row[f"f1_{cn}"] = pcm[c]["f1"]
        row[f"prauc_{cn}"] = np.nan
    return [row]

# ============================================================
# Tek run — 4 koşul
# ============================================================

def execute_single_run(run_id, data):
    print(f"\n{'█'*60}")
    print(f"  RUN {run_id+1}/{N_RUNS}  (split_seed={run_id})")
    print(f"{'█'*60}")
    t0 = time.time()

    X_raw = data[FEATURE_COLS]
    y = data[TARGET_COL] - 1  # 1,2,3 → 0,1,2

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=run_id, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test_raw), columns=FEATURE_COLS)

    # HFS (train-split içinde — leakage-free)
    df_for_hfs = X_train_scaled.copy()
    df_for_hfs["target"] = y_train.values
    features_B, _ = hybrid_feature_selection(df_for_hfs, "target",
                                             theta=HFS_THETA, w=HFS_W, k=HFS_K)
    print(f"  HFS seçilen {HFS_K}: {features_B}")

    X_train_H = X_train_scaled[features_B]   # HFS özellikleri
    X_test_H  = X_test_scaled[features_B]

    all_rows = []

    # Her koşulu sırayla çalıştır
    for label, use_hfs, use_smote in CONDITIONS:
        feat_tag = f"{'8 HFS' if use_hfs else '21 all'} feat, SMOTE={'on' if use_smote else 'off'}"
        print(f"  [{label}] {feat_tag} ...")

        Xtr = X_train_H if use_hfs else X_train_scaled
        Xte = X_test_H  if use_hfs else X_test_scaled

        best = run_gridsearch(get_models(), Xtr, y_train, use_smote=use_smote)
        cvr  = run_cv(best, Xtr, y_train)
        rows = evaluate_models(best, Xtr, y_train, Xte, y_test, cvr, label, run_id)
        all_rows.extend(rows)

    # Naïve baseline (bir kez, tüm özelliklerle)
    all_rows.extend(evaluate_naive_baseline(X_train_scaled, y_train,
                                            X_test_scaled, y_test, run_id))

    elapsed = time.time() - t0
    print(f"  → Run {run_id+1} tamamlandı: {elapsed/60:.1f} dk")
    return all_rows, features_B, elapsed

# ============================================================
# Resume helpers
# ============================================================

def get_completed_runs():
    if not os.path.exists(F_ALL_RUNS):
        return set()
    df = pd.read_csv(F_ALL_RUNS)
    return set(df["run_id"].unique())

def append_to_csv(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ============================================================
# PHASE 1 — RUN LOOP
# ============================================================

def run_all_experiments():
    print("=" * 60)
    print("THYROID — FAKTÖRİYEL ABLASYON + ZENGİN METRİKLER")
    print(f"Data: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    print(f"Total: {len(data)} samples, {len(FEATURE_COLS)} features")
    print(f"Classes: {data[TARGET_COL].value_counts().sort_index().to_dict()}")
    print(f"Koşullar: {[c[0] for c in CONDITIONS]} + NAIVE")
    print("=" * 60)

    completed = get_completed_runs()
    if completed:
        print(f"\n[RESUME] {len(completed)} run zaten tamamlandı: {sorted(completed)}")

    total_elapsed = 0
    done_this_session = 0
    for run_id in SPLIT_SEEDS:
        if run_id in completed:
            print(f"\n  Run {run_id+1} atlanıyor (zaten var)")
            continue
        try:
            rows, features_B, elapsed = execute_single_run(run_id, data)
        except Exception as e:
            print(f"\n  ✗ Run {run_id} HATA: {e}")
            import traceback; traceback.print_exc()
            continue

        append_to_csv(rows, F_ALL_RUNS)
        append_to_csv(
            [{"run_id": run_id, "rank": i+1, "feature": f}
             for i, f in enumerate(features_B)],
            F_HFS_LOG
        )
        total_elapsed += elapsed
        done_this_session += 1
        runs_left = sum(1 for r in SPLIT_SEEDS if r > run_id and r not in completed)
        if runs_left > 0 and done_this_session > 0:
            avg = total_elapsed / done_this_session
            print(f"  → Tahmini kalan: {avg * runs_left / 60:.1f} dk")

    print(f"\n{'='*60}\nTüm runlar bitti. Bu oturum: {total_elapsed/60:.1f} dk\n{'='*60}")

# ============================================================
# PHASE 2 — AGGREGATE
# ============================================================

def mean_ci95(series):
    """Ortalama ve %95 güven aralığı yarı-genişliği (n=10)."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return np.nan, np.nan
    m = s.mean()
    half = 1.96 * s.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, half

def aggregate_results():
    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)

    df = pd.read_csv(F_ALL_RUNS)
    n_runs = df["run_id"].nunique()
    print(f"Data'da run sayısı: {n_runs}")

    # Aggregate metrik listesi
    agg_metrics = ["accuracy", "precision_weighted", "recall_weighted",
                   "specificity_macro", "f1_weighted", "f1_macro", "roc_auc_ovr",
                   "cv_mean"]

    # 1. Özet: koşul × model, mean ± std + %95 CI
    summ_rows = []
    for (cond, model), g in df.groupby(["condition", "model"]):
        rec = {"condition": cond, "model": model}
        for m in agg_metrics:
            mean, ci = mean_ci95(g[m])
            rec[f"{m}_mean"] = round(mean, 4) if pd.notna(mean) else np.nan
            rec[f"{m}_ci95"] = round(ci, 4) if pd.notna(ci) else np.nan
        summ_rows.append(rec)
    pd.DataFrame(summ_rows).to_csv(F_SUMMARY, index=False)
    print(f"  ✓ Özet → {F_SUMMARY}")

    # 2. Per-class özet (sens/spec/prec/f1/prauc for each class)
    pc_metrics = []
    for cn in CLASS_NAMES.values():
        pc_metrics += [f"sens_{cn}", f"spec_{cn}", f"prec_{cn}", f"f1_{cn}", f"prauc_{cn}"]
    pc_rows = []
    for (cond, model), g in df.groupby(["condition", "model"]):
        rec = {"condition": cond, "model": model}
        for m in pc_metrics:
            if m in g.columns:
                mean, ci = mean_ci95(g[m])
                rec[f"{m}_mean"] = round(mean, 4) if pd.notna(mean) else np.nan
                rec[f"{m}_ci95"] = round(ci, 4) if pd.notna(ci) else np.nan
        pc_rows.append(rec)
    pd.DataFrame(pc_rows).to_csv(F_PERCLASS, index=False)
    print(f"  ✓ Per-class özet → {F_PERCLASS}")

    # 3. HFS frekans
    if os.path.exists(F_HFS_LOG):
        hfs = pd.read_csv(F_HFS_LOG)
        ff = hfs["feature"].value_counts().reset_index()
        ff.columns = ["feature", "selection_count"]
        ff["selection_pct"] = (ff["selection_count"] / n_runs * 100).round(1)
        ff.to_csv(os.path.join(OUTPUT_DIR, "hfs_feature_frequency.csv"), index=False)
        print(f"  ✓ HFS frekans → hfs_feature_frequency.csv")

    # 4. Paired Wilcoxon — doğru çiftler, ROC-AUC ve Specificity üzerinde
    #    C-vs-A: saf HFS | D-vs-A: saf SMOTE | B-vs-D: HFS artımsal | B-vs-A: toplam
    pairs = [
        ("C_HFSonly",   "A_baseline",  "HFS_only_effect"),
        ("D_SMOTEonly", "A_baseline",  "SMOTE_only_effect"),
        ("B_HFS_SMOTE", "D_SMOTEonly", "HFS_incremental_over_SMOTE"),
        ("B_HFS_SMOTE", "A_baseline",  "total_effect"),
    ]
    metrics_to_test = ["roc_auc_ovr", "specificity_macro", "f1_macro"]

    wil_rows = []
    for cond_b, cond_a, pair_name in pairs:
        for metric in metrics_to_test:
            for model in df["model"].unique():
                if model == "DummyClassifier":
                    continue
                a = df[(df.condition == cond_a) & (df.model == model)] \
                    .sort_values("run_id")[metric].values
                b = df[(df.condition == cond_b) & (df.model == model)] \
                    .sort_values("run_id")[metric].values
                if len(a) < 3 or len(b) < 3 or len(a) != len(b):
                    continue
                # Tüm farklar sıfırsa Wilcoxon hata verir → guard
                if np.allclose(a, b):
                    stat, p = np.nan, 1.0
                else:
                    try:
                        stat, p = wilcoxon(b, a, alternative="two-sided")
                    except Exception:
                        stat, p = np.nan, np.nan
                wil_rows.append({
                    "pair": pair_name, "metric": metric, "model": model,
                    "mean_ref": round(a.mean(), 4), "mean_test": round(b.mean(), 4),
                    "delta": round(b.mean() - a.mean(), 4),
                    "wilcoxon_stat": stat, "p_value": p,
                    "significant": "yes" if (pd.notna(p) and p < 0.05) else "no",
                })
    # --- R1-47: Çoklu-test düzeltmesi ---
    # Her (çift × metrik) bir hipotez ailesi; düzeltme modeller ARASINDA uygulanır.
    wdf_corr = pd.DataFrame(wil_rows)
    if len(wdf_corr) > 0:
        wdf_corr["p_holm"] = np.nan
        wdf_corr["p_bh"] = np.nan
        wdf_corr["sig_holm"] = "no"
        wdf_corr["sig_bh"] = "no"
        for (pair_name, metric), grp in wdf_corr.groupby(["pair", "metric"]):
            idx = grp.index
            praw = grp["p_value"].values
            ph = holm_bonferroni(praw)
            pb = benjamini_hochberg(praw)
            wdf_corr.loc[idx, "p_holm"] = np.round(ph, 5)
            wdf_corr.loc[idx, "p_bh"] = np.round(pb, 5)
            wdf_corr.loc[idx, "sig_holm"] = ["yes" if (pd.notna(x) and x < 0.05) else "no" for x in ph]
            wdf_corr.loc[idx, "sig_bh"]   = ["yes" if (pd.notna(x) and x < 0.05) else "no" for x in pb]
        wdf_corr.to_csv(F_WILCOXON, index=False)
        n_raw = (wdf_corr["p_value"] < 0.05).sum()
        n_holm = (wdf_corr["sig_holm"] == "yes").sum()
        n_bh = (wdf_corr["sig_bh"] == "yes").sum()
        print(f"  ✓ Wilcoxon + düzeltme → {F_WILCOXON}")
        print(f"    Anlamlı (ham p<0.05): {n_raw} | Holm sonrası: {n_holm} | BH sonrası: {n_bh}")
        wil_rows = wdf_corr.to_dict("records")  # rapor için güncelle
    else:
        pd.DataFrame(wil_rows).to_csv(F_WILCOXON, index=False)
        print(f"  ✓ Wilcoxon → {F_WILCOXON} (düzeltme için satır yok)")

    # 5. Okunur rapor
    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"ABLASYON RAPORU — THYROID — {n_runs} run\n")
        f.write("Koşullar: A=baseline, C=HFS-only, D=SMOTE-only, B=HFS+SMOTE\n")
        f.write("=" * 70 + "\n\n")

        # Naïve baseline
        naive = df[df.condition == "NAIVE_majority"]
        if len(naive):
            f.write("NAÏVE MAJORITY-CLASS BASELINE (R1-58)\n" + "-"*70 + "\n")
            f.write(f"  Accuracy: {naive['accuracy'].mean():.4f}  "
                    f"(çoğunluk sınıfı %92.6 → yüksek accuracy ama işe yaramaz)\n")
            f.write(f"  Macro-F1: {naive['f1_macro'].mean():.4f}  "
                    f"(azınlık sınıflar tamamen kaçırılıyor)\n\n")

        # Ablasyon özeti: her koşulda ortalama specificity ve macro-F1
        f.write("KOŞUL BAZLI ORTALAMALAR (tüm modeller)\n" + "-"*70 + "\n")
        for cond, _, _ in [(c[0], None, None) for c in CONDITIONS]:
            g = df[df.condition == cond]
            f.write(f"  {cond:<14} ROC-AUC={g['roc_auc_ovr'].mean():.4f}  "
                    f"Spec(macro)={g['specificity_macro'].mean():.4f}  "
                    f"Macro-F1={g['f1_macro'].mean():.4f}\n")

        # Ablasyon anahtar sorusu: specificity nereden geliyor?
        f.write("\n\nABLASYON — SPECIFICITY KAYNAĞI (anahtar soru)\n" + "-"*70 + "\n")
        f.write("Aşağıdaki modeller baseline'da düşük specificity gösteriyordu.\n")
        f.write("HFS-only vs SMOTE-only karşılaştırması iyileşmenin kaynağını gösterir:\n\n")
        for model in ["SVM (Linear)", "SVM (RBF)", "KNN", "Logistic Regression"]:
            def cond_spec(c):
                v = df[(df.condition==c) & (df.model==model)]["specificity_macro"]
                return v.mean() if len(v) else np.nan
            f.write(f"  {model:<20} "
                    f"A={cond_spec('A_baseline'):.3f} → "
                    f"C(HFS)={cond_spec('C_HFSonly'):.3f} | "
                    f"D(SMOTE)={cond_spec('D_SMOTEonly'):.3f} | "
                    f"B(both)={cond_spec('B_HFS_SMOTE'):.3f}\n")

        # Wilcoxon özeti
        f.write("\n\nWILCOXON — anlamlı sonuçlar (p<0.05)\n" + "-"*70 + "\n")
        wdf = pd.DataFrame(wil_rows)
        if len(wdf) == 0 or "significant" not in wdf.columns:
            f.write("  (Wilcoxon için yeterli run yok — en az 3 run gerekli)\n")
        else:
            sig = wdf[wdf.significant == "yes"]
            for pair_name in [p[2] for p in pairs]:
                sub = sig[sig.pair == pair_name]
                f.write(f"\n  [{pair_name}] — {len(sub)} anlamlı:\n")
                for _, r in sub.iterrows():
                    f.write(f"    {r['model']:<20} {r['metric']:<18} "
                            f"Δ={r['delta']:+.4f}  p={r['p_value']:.4f}\n")

    print(f"  ✓ Rapor → {F_REPORT}")
    print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)

    # Konsol özeti
    print("\nAnahtar bulgular:")
    for cond, _, _ in [(c[0], None, None) for c in CONDITIONS]:
        g = df[df.condition == cond]
        print(f"  {cond:<14} Spec(macro)={g['specificity_macro'].mean():.4f}  "
              f"Macro-F1={g['f1_macro'].mean():.4f}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_all_experiments()
    aggregate_results()
