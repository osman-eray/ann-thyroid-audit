# ============================================================
# FIGURE ÜRETİM SCRIPTI — Figures 1, 6, 7, 8
# "What Actually Drives Performance in Multiclass Thyroid
#  Classification? A Controlled Audit of the ann-thyroid Benchmark"
#
# Tek script, tek koşum, makaleyle birebir örtüşen figürler.
#
#   Figure 1 : Sınıf dağılımı (veri sabiti — koşum gerektirmez)
#   Figure 6 : Confusion matrix — 10 SPLIT TOPLAMI (aggregate)
#   Figure 7 : ROC eğrileri — 10 SPLIT ORTALAMASI ± 1 SD
#   Figure 8 : Sınıf-bazlı SHAP — 4 model × 3 sınıf
#
# Sınıf adlandırması hormon profiline göre doğrulanmıştır (makale §3.1):
#   Class 0: TSH medyan 53.0 (↑↑), TT4 31.6 (↓)  → overt hypothyroid
#   Class 1: TSH medyan 9.7 (↑),  TT4 90.0       → subclinical hypothyroid
#   Class 2: TSH medyan 1.5,      TT4 109.0      → euthyroid
#
# ÇIKTI (300 DPI):
#   Figure1_class_distribution.png
#   Figure6_confusion_matrix_aggregate.png
#   Figure7_roc_curves_mean.png
#   Figure8_shap_per_class.png
#   figure_values.csv   ← figürlerdeki sayılar (tablolarla çapraz kontrol için)
#
# GEREKLİ: shap  (pip install shap)
# SÜRE: ~3.5-5 saat (Erazer). Figure 1 saniyeler; 6/7/8 model eğitimi gerektirir.
#   Grid'ler ablasyon scriptiyle BİREBİR aynıdır — figürlerin Tablo 2/3 ile
#   örtüşmesi için bu şarttır. Grid'i küçültmek figürleri tablolardan ayırır.
#
# KESİNTİ OLURSA: script resume içermez; baştan koşar. Uzun süreceği için
#   ekranı kapatma / uyku modunu devre dışı bırak.
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
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier, StackingClassifier)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None
    print("[UYARI] shap kurulu değil — Figure 8 atlanacak. Kurulum: pip install shap")

# ============================================================
# CONFIG
# ============================================================
N_RUNS        = 10
SPLIT_SEEDS   = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT      = 8
HFS_THETA     = 0.90
HFS_W         = 0.50
SHAP_SAMPLE   = 300

DATA_PATH = "thyroid_ann_7200.csv"
OUT_DIR   = "."

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

CLASS_SHORT = {0: "Overt\nhypo",  1: "Subclin.\nhypo", 2: "Euthyroid"}
CLASS_LONG  = {0: "Overt hypothyroid", 1: "Subclinical hypothyroid", 2: "Euthyroid"}
CLASS_BAR   = {0: "Overt\nhypothyroid", 1: "Subclinical\nhypothyroid", 2: "Euthyroid"}
COLORS      = ["#c44e52", "#dd8452", "#4c72b0"]

# Figure 6/7'de gösterilecek modeller
CM_MODELS  = ["SVM (RBF)", "SVM (Linear)", "Logistic Regression", "KNN"]
ROC_MODELS = ["Logistic Regression", "Decision Tree", "KNN", "SVM (RBF)", "SVM (Linear)",
              "Random Forest", "AdaBoost", "Bagging", "XGBoost", "Stacking (RF+SVM)"]
SHAP_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]

_values_log = []   # figure_values.csv için


# ============================================================
# MODELLER (makale Tablo 2/3 ile birebir aynı)
# ============================================================

# SÜRÜM NOTU — AdaBoost algoritması:
#   scikit-learn 1.6'da AdaBoost'un varsayılanı SAMME.R'den SAMME'ye değişti
#   (SAMME.R kaldırıldı). Ablasyon sonuçları (Tablo 2/3) SAMME ile üretildiği
#   için, figürlerin onlarla örtüşmesi adına algorithm="SAMME" AÇIKÇA belirtilir.
#   Aksi halde sklearn<1.6 olan bir makinede SAMME.R kullanılır ve AdaBoost
#   ~0.006 daha yüksek AUC verir (0.9965 vs 0.9905) — figür tablodan ayrışır.
#   Bu parametre sklearn>=1.6'da da geçerlidir (tek seçenek zaten SAMME).

def get_model(name):
    M = {
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
            estimators=[('rf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                                      random_state=INTERNAL_SEED, n_jobs=1)),
                        ('svm', SVC(kernel="rbf", probability=True, C=1, max_iter=200000,
                                    random_state=INTERNAL_SEED))],
            final_estimator=LogisticRegression(random_state=INTERNAL_SEED),
            cv=3, n_jobs=1),
    }
    return M[name]

# NOT: Bu grid'ler ablasyon scriptindekiyle (Tablo 2/3'ün kaynağı) BİREBİR
# aynı olmalıdır; aksi halde figürler tablolardan farklı modeller gösterir.
GRIDS_A = {
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
  "Stacking (RF+SVM)":   {"rf__n_estimators": [100], "rf__max_depth": [5, 10]},   # ablasyonla aynı (PERF 4→2)
}
GRIDS_B = {k: {f"model__{p}": v for p, v in g.items()} for k, g in GRIDS_A.items()}


# ============================================================
# HFS (makale §2.3 ile birebir)
# ============================================================

def information_gain(x, y):
    return mutual_info_classif(x.reshape(-1, 1), y, discrete_features='auto',
                               random_state=INTERNAL_SEED)[0]

def gain_ratio(x, y):
    IG = information_gain(x, y)
    nu = len(np.unique(x))
    if nu <= 10:
        _, counts = np.unique(x, return_counts=True)
    else:
        nb = max(10, int(np.sqrt(len(x))))
        counts = np.histogram(x, bins=nb)[0]; counts = counts[counts > 0]
    p = counts / counts.sum()
    Hx = -np.sum(p * np.log2(p))
    return IG / (Hx + 1e-12) if Hx > 0 else 0.0

def hfs_select(Xtr, ytr, k=K_SELECT, theta=HFS_THETA, w=HFS_W):
    feats = list(Xtr.columns); X = Xtr.values
    red = set()
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                red.add(feats[j])
    nonred = [f for f in feats if f not in red]
    sc = [(f, w * information_gain(Xtr[f].values, ytr) + (1 - w) * gain_ratio(Xtr[f].values, ytr))
          for f in nonred]
    sc.sort(key=lambda t: t[1], reverse=True)
    return [f for f, _ in sc[:min(k, len(sc))]]


def prepare_split(data, seed):
    X = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
    Xtr_r, Xte_r, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=seed, stratify=y)
    sc = StandardScaler()
    Xtr = pd.DataFrame(sc.fit_transform(Xtr_r), columns=FEATURE_COLS)
    Xte = pd.DataFrame(sc.transform(Xte_r), columns=FEATURE_COLS)
    feats = hfs_select(Xtr, ytr.values)
    return Xtr, Xte, ytr, yte, feats


def fit_pipeline(name, Xtr, ytr, use_hfs_smote):
    """A koşulu: use_hfs_smote=False (21 özellik, SMOTE yok)
       B koşulu: use_hfs_smote=True  (8 özellik, SMOTE)"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
    jobs = 1 if "Stacking" in name else -1
    if use_hfs_smote:
        pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)),
                            ("model", get_model(name))])
        g = GridSearchCV(pipe, GRIDS_B[name], cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=jobs)
    else:
        g = GridSearchCV(get_model(name), GRIDS_A[name], cv=cv,
                         scoring='roc_auc_ovr_weighted', n_jobs=jobs)
    g.fit(Xtr, ytr)
    return g.best_estimator_


# ============================================================
# FIGURE 1 — Sınıf dağılımı
# ============================================================

def figure1(data):
    print("\n[Figure 1] Sınıf dağılımı ...")
    y = data[TARGET_COL] - 1
    counts = y.value_counts().sort_index()
    total = counts.sum()
    pcts = (counts / total * 100).round(1)
    ratio = counts.max() / counts.min()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color=COLORS, edgecolor="black",
                  linewidth=0.8, width=0.62, zorder=3)
    for b, n, p in zip(bars, counts.values, pcts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + total*0.012,
                f"n = {n:,}\n({p}%)", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_BAR[c] for c in sorted(counts.index)], fontsize=10.5)
    ax.set_ylabel("Number of instances", fontsize=11)
    ax.set_ylim(0, counts.max()*1.18)
    ax.set_title("Class Distribution of the ann-thyroid Dataset\n"
                 f"(N = {total:,};  imbalance ratio {ratio:.1f}:1)",
                 fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure1_class_distribution.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    for c in sorted(counts.index):
        _values_log.append({"figure": 1, "item": CLASS_LONG[c],
                            "metric": "n", "value": int(counts[c])})
    print(f"  ✓ Figure1_class_distribution.png  ({dict(counts)}, ratio {ratio:.1f}:1)")


# ============================================================
# FIGURE 6 — Confusion matrix, 10 SPLIT TOPLAMI
# ============================================================

def figure6(data):
    print("\n[Figure 6] Confusion matrix (10 split toplamı) ...")
    agg = {m: {"A": np.zeros((3, 3), dtype=int), "B": np.zeros((3, 3), dtype=int)}
           for m in CM_MODELS}
    for seed in SPLIT_SEEDS:
        Xtr, Xte, ytr, yte, feats = prepare_split(data, seed)
        t0 = time.time()
        for m in CM_MODELS:
            for cond, use in [("A", False), ("B", True)]:
                Xa, Xb = (Xtr[feats], Xte[feats]) if use else (Xtr, Xte)
                est = fit_pipeline(m, Xa, ytr, use)
                pred = est.predict(Xb)
                agg[m][cond] += confusion_matrix(yte, pred, labels=[0, 1, 2])
        print(f"  seed={seed} bitti ({time.time()-t0:.0f}s)")

    def spec_sens(cm):
        tot = cm.sum(); sp, se = [], []
        for i in range(3):
            tp = cm[i, i]; fn = cm[i, :].sum()-tp; fp = cm[:, i].sum()-tp
            tn = tot-tp-fn-fp
            sp.append(tn/(tn+fp) if tn+fp else 0); se.append(tp/(tp+fn) if tp+fn else 0)
        return np.mean(se), np.mean(sp)

    n = len(CM_MODELS)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.8*n),
                             gridspec_kw={'wspace': 0.30, 'hspace': 0.50})
    for i, m in enumerate(CM_MODELS):
        for j, (cond, title) in enumerate([("A", "without HFS"), ("B", "with HFS + SMOTE")]):
            cm = agg[m][cond]
            se, sp = spec_sens(cm)
            acc = np.trace(cm)/cm.sum()
            ax = axes[i, j]
            rs = cm.sum(axis=1, keepdims=True).astype(float); rs[rs == 0] = 1
            pct = cm/rs*100
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
            for r in range(3):
                for c in range(3):
                    ax.text(c, r, f"{cm[r,c]:,}\n({pct[r,c]:.1f}%)", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if pct[r, c] > 50 else "black")
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels([CLASS_SHORT[c] for c in range(3)], fontsize=9)
            ax.set_yticklabels([CLASS_SHORT[c] for c in range(3)], fontsize=9, rotation=90, va="center")
            ax.set_title(f"{title}\nSensitivity={se:.3f}  Specificity={sp:.3f}  Acc={acc:.3f}",
                         fontsize=10, pad=8)
            if i == n-1: ax.set_xlabel("Predicted", fontsize=10)
            if j == 0:   ax.set_ylabel(f"{m}\n\nTrue", fontsize=11, fontweight="bold", labelpad=8)
            for e in ["top","right","bottom","left"]: ax.spines[e].set_visible(False)
            ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
            ax.grid(which="minor", color="gray", linewidth=1)
            ax.tick_params(which="minor", length=0)
            _values_log.append({"figure": 6, "item": f"{m} / {cond}",
                                "metric": "sensitivity_macro", "value": round(se, 4)})
            _values_log.append({"figure": 6, "item": f"{m} / {cond}",
                                "metric": "specificity_macro", "value": round(sp, 4)})
    fig.suptitle("Confusion Matrices Aggregated over Ten Splits\nwithout HFS vs. with HFS + SMOTE",
                 fontsize=13, fontweight="bold", y=0.998)
    plt.savefig(os.path.join(OUT_DIR, "Figure6_confusion_matrix_aggregate.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Figure6_confusion_matrix_aggregate.png")


# ============================================================
# FIGURE 7 — ROC, 10 SPLIT ORTALAMASI ± 1 SD
# ============================================================

def figure7(data):
    print("\n[Figure 7] ROC (10 split ortalaması ± 1 SD) ...")
    grid = np.linspace(0, 1, 200)
    store = {m: {"A": [], "B": []} for m in ROC_MODELS}
    aucs  = {m: {"A": [], "B": []} for m in ROC_MODELS}
    wauc  = {m: {"A": [], "B": []} for m in ROC_MODELS}   # weighted OVR (Tablo 2/3 metriği)

    for seed in SPLIT_SEEDS:
        Xtr, Xte, ytr, yte, feats = prepare_split(data, seed)
        yb = label_binarize(yte, classes=[0, 1, 2])
        t0 = time.time()
        for m in ROC_MODELS:
            for cond, use in [("A", False), ("B", True)]:
                Xa, Xb = (Xtr[feats], Xte[feats]) if use else (Xtr, Xte)
                est = fit_pipeline(m, Xa, ytr, use)
                prob = est.predict_proba(Xb)
                # macro-average OVR ROC
                tprs = []
                for c in range(3):
                    fpr_c, tpr_c, _ = roc_curve(yb[:, c], prob[:, c])
                    tprs.append(np.interp(grid, fpr_c, tpr_c))
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[0] = 0.0
                store[m][cond].append(mean_tpr)
                aucs[m][cond].append(auc(grid, mean_tpr))
                # tablolarla birebir karşılaştırma için weighted OVR AUC de sakla
                wauc[m][cond].append(roc_auc_score(yte, prob, multi_class='ovr',
                                                   average='weighted'))
        print(f"  seed={seed} bitti ({time.time()-t0:.0f}s)")

    fig, axes = plt.subplots(5, 2, figsize=(11, 17),
                             gridspec_kw={'wspace': 0.24, 'hspace': 0.45})
    for idx, m in enumerate(ROC_MODELS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        for cond, col, style, lab in [("A", "#888888", "--", "without HFS"),
                                      ("B", "#c44e52", "-",  "HFS + SMOTE")]:
            arr = np.array(store[m][cond])
            mu, sd = arr.mean(0), arr.std(0)
            a_mu, a_sd = np.mean(aucs[m][cond]), np.std(aucs[m][cond])
            w_mu = np.mean(wauc[m][cond])
            ax.plot(grid, mu, style, color=col, lw=2.0, zorder=3,
                    label=f"{lab}\n  macro AUC={a_mu:.4f}±{a_sd:.4f}\n  weighted AUC={w_mu:.4f}")
            ax.fill_between(grid, np.clip(mu-sd, 0, 1), np.clip(mu+sd, 0, 1),
                            color=col, alpha=0.18, zorder=2, linewidth=0)
            _values_log.append({"figure": 7, "item": f"{m} / {cond}",
                                "metric": "auc_macro_mean", "value": round(a_mu, 4)})
            _values_log.append({"figure": 7, "item": f"{m} / {cond}",
                                "metric": "auc_weighted_mean", "value": round(w_mu, 4)})
        ax.plot([0, 1], [0, 1], ":", color="black", lw=1, alpha=0.5, zorder=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(m, fontsize=11, fontweight="bold", pad=6)
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
        ax.grid(alpha=0.25, linestyle="--", zorder=0); ax.set_axisbelow(True)
        if r == 4: ax.set_xlabel("False Positive Rate", fontsize=10)
        if c == 0: ax.set_ylabel("True Positive Rate", fontsize=10)
    fig.suptitle("ROC Curves (one-vs-rest), mean over ten splits\n"
                 "Curve and shaded band: macro-average ±1 SD. Weighted AUC (the metric of Tables 2-3) also shown.",
                 fontsize=13, fontweight="bold", y=0.997)
    plt.savefig(os.path.join(OUT_DIR, "Figure7_roc_curves_mean.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Figure7_roc_curves_mean.png")


# ============================================================
# FIGURE 8 — Sınıf-bazlı SHAP, 4 model × 3 sınıf
# ============================================================

def shap_per_class(model, X_bg, X_ex, feats, name):
    if name == "Logistic Regression":
        sv = shap.LinearExplainer(model, X_bg).shap_values(X_ex)
    else:
        sv = shap.TreeExplainer(model).shap_values(X_ex)
    out = {}
    nf = len(feats)
    if isinstance(sv, list):
        for c, arr in enumerate(sv): out[c] = np.asarray(arr)
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            if sv.shape[-1] == nf:
                for c in range(sv.shape[0]): out[c] = sv[c]
            else:
                for c in range(sv.shape[2]): out[c] = sv[:, :, c]
        else:
            out[0] = sv
    return out

def figure8(data):
    if shap is None:
        print("\n[Figure 8] ATLANDI — shap kurulu değil"); return
    print("\n[Figure 8] Sınıf-bazlı SHAP (10 split ortalaması) ...")
    acc = {m: {c: {} for c in range(3)} for m in SHAP_MODELS}
    for seed in SPLIT_SEEDS:
        Xtr, Xte, ytr, yte, feats = prepare_split(data, seed)
        rng = np.random.RandomState(INTERNAL_SEED + seed)
        idx = rng.choice(len(Xte), min(SHAP_SAMPLE, len(Xte)), replace=False)
        X_ex = Xte[feats].iloc[idx]
        X_bg = Xtr[feats].sample(min(100, len(Xtr)), random_state=INTERNAL_SEED)
        t0 = time.time()
        for m in SHAP_MODELS:
            est = fit_pipeline(m, Xtr[feats], ytr, True)
            inner = est.named_steps["model"]
            try:
                sv = shap_per_class(inner, X_bg, X_ex, feats, m)
            except Exception as e:
                print(f"    ✗ {m}: {e}"); continue
            for c, mat in sv.items():
                if c not in range(3): continue
                imp = np.abs(mat).mean(axis=0)
                for f, v in zip(feats, imp):
                    acc[m][c].setdefault(f, []).append(float(v))
        print(f"  seed={seed} bitti ({time.time()-t0:.0f}s)")

    fig, axes = plt.subplots(len(SHAP_MODELS), 3,
                             figsize=(13, 3.6*len(SHAP_MODELS)),
                             gridspec_kw={'wspace': 0.42, 'hspace': 0.55})
    for i, m in enumerate(SHAP_MODELS):
        for c in range(3):
            ax = axes[i, c]
            d = acc[m][c]
            if not d:
                ax.axis("off"); continue
            means = {f: np.mean(v) for f, v in d.items()}
            order = sorted(means, key=means.get)
            vals  = [means[f] for f in order]
            errs  = [np.std(d[f]) for f in order]
            ax.barh(range(len(order)), vals, xerr=errs, color=COLORS[c],
                    edgecolor="black", linewidth=0.5, alpha=0.85,
                    error_kw={"linewidth": 0.8, "ecolor": "gray"})
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=8.5)
            ax.set_xlabel("mean |SHAP|", fontsize=9)
            if c == 0:
                ax.set_ylabel(m, fontsize=10.5, fontweight="bold", labelpad=8)
            if i == 0:
                ax.set_title(CLASS_LONG[c], fontsize=11, fontweight="bold", pad=8)
            ax.grid(axis="x", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
            top = order[-1]
            _values_log.append({"figure": 8, "item": f"{m} / {CLASS_LONG[c]}",
                                "metric": "top_feature", "value": top})
    fig.suptitle("Class-specific SHAP Importance\n"
                 "Mean |SHAP| over ten splits; error bars = ±1 SD across splits",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.savefig(os.path.join(OUT_DIR, "Figure8_shap_per_class.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Figure8_shap_per_class.png")
    print("\n  Sınıf başına en önemli özellik (§3.10 iddiasının kontrolü):")
    for m in SHAP_MODELS:
        tops = {}
        for c in range(3):
            if acc[m][c]:
                means = {f: np.mean(v) for f, v in acc[m][c].items()}
                tops[CLASS_LONG[c]] = max(means, key=means.get)
        uniq = len(set(tops.values()))
        print(f"    {m:<22} {tops}  → {'FARKLI' if uniq>1 else 'aynı'}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print("FIGURE ÜRETİMİ — Figures 1, 6, 7, 8")
    print("=" * 64)
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {DATA_PATH} | {len(data)} örnek, {len(FEATURE_COLS)} özellik")
    print(f"Sınıflar: {data[TARGET_COL].value_counts().sort_index().to_dict()}")
    print(f"Split sayısı: {N_RUNS} | HFS: k={K_SELECT}, w={HFS_W}, theta={HFS_THETA}")

    t_all = time.time()
    figure1(data)
    figure6(data)
    figure7(data)
    figure8(data)

    if _values_log:
        pd.DataFrame(_values_log).to_csv(os.path.join(OUT_DIR, "figure_values.csv"), index=False)
        print(f"\n  ✓ figure_values.csv  ({len(_values_log)} satır — tablolarla çapraz kontrol için)")

    print("\n" + "=" * 64)
    print(f"TAMAMLANDI — toplam {(time.time()-t_all)/60:.1f} dk")
    print("=" * 64)
