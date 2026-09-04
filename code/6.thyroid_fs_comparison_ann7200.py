# ============================================================
# SCRIPT 3 — Özellik Seçim Yöntemi Karşılaştırması
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumu: R1-16 — "HFS'yi SelectKBest, Mutual Information, ReliefF,
#   RFE, LASSO, Boruta, mRMR, tree-based importance ile karşılaştırın."
#
# NE YAPAR:
#   Önerilen HFS'yi 8 yerleşik FS yöntemiyle kıyaslar. Her yöntem k=8 özellik seçer,
#   ardından AYNI downstream pipeline (SMOTE + tuned classifier) ile değerlendirilir.
#   Böylece fark yalnızca seçim yönteminden gelir (adil karşılaştırma).
#
# YÖNTEMLER (harici paket YOK — hepsi sklearn/numpy tabanlı):
#   1. HFS (önerilen)
#   2. SelectKBest (ANOVA F)
#   3. Mutual Information (MI)
#   4. ReliefF            (saf numpy implementasyon)
#   5. RFE (LogReg tabanlı)
#   6. LASSO (L1 LogReg katsayı büyüklüğü)
#   7. Boruta-benzeri     (gölge-özellik + RF importance; saf implementasyon)
#   8. mRMR               (saf numpy: MI relevance − redundancy)
#   9. Tree-based importance (RandomForest)
#
# ÇIKTI:
#   fs_comparison_all_runs.csv, fs_comparison_summary.csv,
#   fs_selected_features.csv, fs_overlap_with_hfs.csv, fs_comparison_report.txt
#
# RESUME: (yöntem, seed) bazında.
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, f_classif, RFE
)
from sklearn.linear_model import LogisticRegression, LogisticRegression as LR2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
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
OUTPUT_DIR = "thyroid_fs_comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_ALL      = os.path.join(OUTPUT_DIR, "fs_comparison_all_runs.csv")
F_SELECTED = os.path.join(OUTPUT_DIR, "fs_selected_features.csv")
F_SUMMARY  = os.path.join(OUTPUT_DIR, "fs_comparison_summary.csv")
F_OVERLAP  = os.path.join(OUTPUT_DIR, "fs_overlap_with_hfs.csv")
F_REPORT   = os.path.join(OUTPUT_DIR, "fs_comparison_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

# Downstream değerlendirme modeli (tüm FS yöntemleri için AYNI — adil karşılaştırma)
# RandomForest güçlü ve stabil bir temsilci; tek model ile FS yöntemleri izole edilir.
def get_eval_model():
    return RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED)

EVAL_GRID = {"model__n_estimators": [100, 200], "model__max_depth": [10, None]}
SCORING_CV = 'roc_auc_ovr_weighted'

# ============================================================
# HFS (önerilen — makale ile birebir)
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
        counts = np.histogram(X_col, bins=n_bins)[0]
        counts = counts[counts > 0]
    probs = counts / counts.sum()
    Hx = -np.sum(probs * np.log2(probs))
    return IG / (Hx + 1e-12) if Hx > 0 else 0.0

def fs_hfs(X, y, feat_names, k=K_SELECT, theta=0.90, w=0.5):
    df = pd.DataFrame(X, columns=feat_names)
    redundant = set()
    n = len(feat_names)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                redundant.add(feat_names[j])
    non_red = [f for f in feat_names if f not in redundant]
    scores = []
    for f in non_red:
        xf = df[f].values
        hs = w * information_gain(xf, y) + (1 - w) * gain_ratio(xf, y)
        scores.append((f, hs))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores[:min(k, len(scores))]]

# ============================================================
# Yerleşik FS yöntemleri (hepsi saf sklearn/numpy)
# ============================================================

def fs_selectkbest(X, y, feat_names, k=K_SELECT):
    sel = SelectKBest(f_classif, k=k).fit(X, y)
    idx = np.argsort(sel.scores_)[::-1][:k]
    return [feat_names[i] for i in idx]

def fs_mutual_info(X, y, feat_names, k=K_SELECT):
    mi = mutual_info_classif(X, y, random_state=INTERNAL_SEED)
    idx = np.argsort(mi)[::-1][:k]
    return [feat_names[i] for i in idx]

def fs_relieff(X, y, feat_names, k=K_SELECT, n_neighbors=10, sample_size=500):
    """Saf numpy ReliefF (çok-sınıf). Örneklem üzerinden komşu farkları."""
    rng = np.random.RandomState(INTERNAL_SEED)
    n, d = X.shape
    # ölçek [0,1] (Relief mesafe için)
    Xn = (X - X.min(0)) / (np.ptp(X, axis=0) + 1e-12)
    classes, counts = np.unique(y, return_counts=True)
    prior = dict(zip(classes, counts / n))
    m = min(sample_size, n)
    idxs = rng.choice(n, m, replace=False)
    W = np.zeros(d)
    for i in idxs:
        xi, yi = Xn[i], y[i]
        dist = np.abs(Xn - xi).sum(1)
        # aynı sınıf (hit)
        same = np.where(y == yi)[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        hits = same[np.argsort(dist[same])[:n_neighbors]]
        W -= np.abs(Xn[hits] - xi).mean(0)
        # diğer sınıflar (miss) — prior ağırlıklı
        for c in classes:
            if c == yi:
                continue
            oth = np.where(y == c)[0]
            if len(oth) == 0:
                continue
            miss = oth[np.argsort(dist[oth])[:n_neighbors]]
            W += (prior[c] / (1 - prior[yi] + 1e-12)) * np.abs(Xn[miss] - xi).mean(0)
    idx = np.argsort(W)[::-1][:k]
    return [feat_names[i] for i in idx]

def fs_rfe(X, y, feat_names, k=K_SELECT):
    est = LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED)
    sel = RFE(est, n_features_to_select=k, step=1).fit(X, y)
    return [f for f, s in zip(feat_names, sel.support_) if s]

def fs_lasso(X, y, feat_names, k=K_SELECT):
    # L1 multinomial LogReg — katsayı büyüklüğü (sınıflar arası ortalama |coef|)
    est = LogisticRegression(penalty='l1', solver='saga', max_iter=2000,
                             C=0.5, random_state=INTERNAL_SEED)
    est.fit(X, y)
    importance = np.abs(est.coef_).mean(0)
    idx = np.argsort(importance)[::-1][:k]
    return [feat_names[i] for i in idx]

def fs_boruta_like(X, y, feat_names, k=K_SELECT, n_iter=20):
    """Gölge-özellik + RF importance; Boruta mantığının hafif implementasyonu.
       Her özelliğin, gölgelerin maksimum importance'ını kaç kez geçtiğini sayar."""
    rng = np.random.RandomState(INTERNAL_SEED)
    n, d = X.shape
    hits = np.zeros(d)
    for it in range(n_iter):
        Xs = X.copy()
        shadow = X.copy()
        for j in range(d):
            shadow[:, j] = rng.permutation(shadow[:, j])
        Xaug = np.hstack([Xs, shadow])
        rf = RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED + it, n_jobs=-1)
        rf.fit(Xaug, y)
        imp = rf.feature_importances_
        real_imp, shadow_imp = imp[:d], imp[d:]
        shadow_max = shadow_imp.max()
        hits += (real_imp > shadow_max).astype(float)
    idx = np.argsort(hits)[::-1][:k]  # en çok "gölgeyi geçen" k özellik
    return [feat_names[i] for i in idx]

def fs_mrmr(X, y, feat_names, k=K_SELECT):
    """Saf numpy mRMR: max relevance (MI with y) − min redundancy (mean MI with selected)."""
    d = X.shape[1]
    rel = mutual_info_classif(X, y, random_state=INTERNAL_SEED)
    # özellikler arası MI (redundancy) — sürekli MI yaklaşık
    from sklearn.feature_selection import mutual_info_regression
    selected, remaining = [], list(range(d))
    # ilk: en yüksek relevance
    first = int(np.argmax(rel))
    selected.append(first); remaining.remove(first)
    # redundancy cache
    red_cache = {}
    while len(selected) < min(k, d):
        best_score, best_f = -np.inf, None
        for f in remaining:
            reds = []
            for s in selected:
                key = tuple(sorted((f, s)))
                if key not in red_cache:
                    mi_fs = mutual_info_regression(
                        X[:, f].reshape(-1, 1), X[:, s], random_state=INTERNAL_SEED)[0]
                    red_cache[key] = mi_fs
                reds.append(red_cache[key])
            score = rel[f] - np.mean(reds) if reds else rel[f]
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f); remaining.remove(best_f)
    return [feat_names[i] for i in selected]

def fs_tree(X, y, feat_names, k=K_SELECT):
    rf = RandomForestClassifier(n_estimators=200, random_state=INTERNAL_SEED, n_jobs=-1)
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:k]
    return [feat_names[i] for i in idx]

FS_METHODS = {
    "HFS (proposed)":   fs_hfs,
    "SelectKBest":      fs_selectkbest,
    "MutualInfo":       fs_mutual_info,
    "ReliefF":          fs_relieff,
    "RFE":              fs_rfe,
    "LASSO":            fs_lasso,
    "Boruta-like":      fs_boruta_like,
    "mRMR":             fs_mrmr,
    "TreeImportance":   fs_tree,
}

# ============================================================
# Değerlendirme
# ============================================================

def macro_specificity(cm):
    total = cm.sum(); specs = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]; fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp; tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specs)

def evaluate_subset(XtrH, ytr, XteH, yte):
    labels = sorted(pd.Series(yte).unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
    pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)), ("model", get_eval_model())])
    grid = GridSearchCV(pipe, EVAL_GRID, cv=cv, scoring=SCORING_CV, n_jobs=-1)
    grid.fit(XtrH, ytr)
    best = grid.best_estimator_
    yprob = best.predict_proba(XteH); ypred = best.predict(XteH)
    roc = roc_auc_score(yte, yprob, multi_class='ovr', average='weighted')
    f1m = f1_score(yte, ypred, average='macro', zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=labels)
    return roc, f1m, macro_specificity(cm)

# ============================================================
# Resume
# ============================================================

def load_done():
    if not os.path.exists(F_ALL):
        return set()
    df = pd.read_csv(F_ALL)
    return set(zip(df["method"], df["run_id"]))

def append_csv(rows, path):
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
    print("FS YÖNTEM KARŞILAŞTIRMASI — ann-thyroid")
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek, {len(FEATURE_COLS)} özellik")
    print(f"Yöntemler: {list(FS_METHODS.keys())}")
    print("="*60)
    done = load_done()
    if done:
        print(f"[RESUME] {len(done)} (yöntem,seed) tamam")

    for run_id in SPLIT_SEEDS:
        X_raw = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X_raw, y, test_size=0.2, random_state=run_id, stratify=y)
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)
        Xtr_np, Xte_np = Xtr.values, Xte.values
        ytr_np = ytr.values

        for mname, mfunc in FS_METHODS.items():
            if (mname, run_id) in done:
                continue
            t0 = time.time()
            try:
                feats = mfunc(Xtr_np, ytr_np, FEATURE_COLS, k=K_SELECT)
                roc, f1m, spec = evaluate_subset(Xtr[feats], ytr, Xte[feats], yte)
            except Exception as e:
                print(f"  ✗ {mname} seed={run_id} HATA: {e}")
                import traceback; traceback.print_exc()
                continue
            append_csv([{"method": mname, "run_id": run_id, "n_selected": len(feats),
                         "roc_auc_ovr": roc, "f1_macro": f1m, "specificity_macro": spec}], F_ALL)
            append_csv([{"method": mname, "run_id": run_id, "rank": i+1, "feature": f}
                        for i, f in enumerate(feats)], F_SELECTED)
            print(f"  seed={run_id} {mname:<16} ROC={roc:.4f} F1m={f1m:.4f} "
                  f"Spec={spec:.4f} ({time.time()-t0:.0f}s)")
    aggregate()

def aggregate():
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
    df = pd.read_csv(F_ALL)
    summ = df.groupby("method").agg(
        roc_mean=("roc_auc_ovr", "mean"), roc_std=("roc_auc_ovr", "std"),
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        spec_mean=("specificity_macro", "mean"), spec_std=("specificity_macro", "std"),
    ).round(4).sort_values("roc_mean", ascending=False).reset_index()
    summ.to_csv(F_SUMMARY, index=False)
    print(f"  ✓ {F_SUMMARY}")

    # HFS ile örtüşme analizi
    if os.path.exists(F_SELECTED):
        sel = pd.read_csv(F_SELECTED)
        hfs_by_run = {r: set(g[g.method == "HFS (proposed)"]["feature"])
                      for r, g in sel.groupby("run_id")}
        overlap_rows = []
        for m in FS_METHODS.keys():
            if m == "HFS (proposed)":
                continue
            jacc = []
            for r, g in sel.groupby("run_id"):
                mset = set(g[g.method == m]["feature"])
                hset = hfs_by_run.get(r, set())
                if hset or mset:
                    jacc.append(len(mset & hset) / len(mset | hset))
            overlap_rows.append({"method": m,
                                 "mean_jaccard_with_HFS": round(np.mean(jacc), 3) if jacc else np.nan,
                                 "mean_overlap_count": round(np.mean(
                                     [len(set(g[g.method==m]['feature']) & hfs_by_run.get(r,set()))
                                      for r,g in sel.groupby('run_id')]), 2)})
        pd.DataFrame(overlap_rows).to_csv(F_OVERLAP, index=False)
        print(f"  ✓ {F_OVERLAP}")

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("="*70 + "\nFS YÖNTEM KARŞILAŞTIRMASI — ann-thyroid\n")
        f.write(f"Her yöntem k={K_SELECT} özellik seçer, aynı SMOTE+RF pipeline ile değerlendirilir.\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Yöntem':<18}{'ROC-AUC':<18}{'Macro-F1':<18}{'Specificity':<18}\n" + "-"*70 + "\n")
        for _, r in summ.iterrows():
            star = "  ← önerilen" if r["method"] == "HFS (proposed)" else ""
            f.write(f"{r['method']:<18}{r['roc_mean']:.4f}±{r['roc_std']:.4f}   "
                    f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}   "
                    f"{r['spec_mean']:.4f}±{r['spec_std']:.4f}{star}\n")
    print(f"  ✓ {F_REPORT}")
    print("\nSıralama (ROC-AUC):")
    for _, r in summ.iterrows():
        print(f"  {r['method']:<18} ROC={r['roc_mean']:.4f}")

if __name__ == "__main__":
    run_all()
    print("\n" + "="*60 + "\nDONE\n" + "="*60)
