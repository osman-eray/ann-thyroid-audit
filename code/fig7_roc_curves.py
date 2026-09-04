# ============================================================
# Figure 7 — ROC Curves, all ten classifiers, both pipelines
# Median-Performance Run (split_seed=5)
#
# Pipeline A (without HFS, no SMOTE)  vs  Pipeline B (HFS + SMOTE)
# Çok-sınıf ROC: one-vs-rest, macro-average + sınıf bazlı eğriler
#
# Sınıf adlandırması hormon profiline göre doğrulanmıştır (bkz. makale §2.1):
#   Class 0 → overt hypothyroid, Class 1 → subclinical hypothyroid,
#   Class 2 → euthyroid
#
# Çıktı: Figure7_roc_curves.png (300 DPI)
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
SPLIT_SEED    = 5        # Median-performance run
INTERNAL_SEED = 42

DATA_PATH   = "thyroid_ann_7200.csv"        # <- Veri seti yolunu güncelle
OUTPUT_PATH = "Figure7_roc_curves.png"

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

HFS_THETA, HFS_W, HFS_K = 0.9, 0.5, 8

CLASS_NAMES = {
    0: "Overt hypothyroid",
    1: "Subclinical hypothyroid",
    2: "Euthyroid",
}

MODEL_ORDER = [
    "Logistic Regression", "Decision Tree", "KNN",
    "SVM (RBF)", "SVM (Linear)", "Random Forest",
    "AdaBoost", "Bagging", "XGBoost", "Stacking (RF+SVM)",
]

# ============================================================
# HFS (makale ile birebir)
# ============================================================

def pearson_correlation(f1, f2):
    return np.corrcoef(f1, f2)[0, 1]

def information_gain(X_col, y):
    return mutual_info_classif(X_col.reshape(-1, 1), y,
                               discrete_features='auto',
                               random_state=INTERNAL_SEED)[0]

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
    redundant = set()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if abs(pearson_correlation(X[:, i], X[:, j])) > theta:
                redundant.add(features[j])
    non_redundant = [f for f in features if f not in redundant]
    scores = []
    for f in non_redundant:
        ig = information_gain(df[f].values, y)
        gr = gain_ratio(df[f].values, y)
        scores.append((f, w * ig + (1 - w) * gr))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores[:k]]

# ============================================================
# MODELLER
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
                                                  random_state=INTERNAL_SEED),   # REV: pinned
        "Bagging":             BaggingClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
        "Stacking (RF+SVM)":   StackingClassifier(
            estimators=[
                ('rf',  RandomForestClassifier(n_estimators=200, max_depth=10,
                                               random_state=INTERNAL_SEED, n_jobs=1)),
                ('svm', SVC(kernel="rbf", probability=True, C=1, max_iter=200000,
                            random_state=INTERNAL_SEED))
            ],
            final_estimator=LogisticRegression(random_state=INTERNAL_SEED),
            cv=3, n_jobs=1),
    }

PARAM_GRIDS_A = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Decision Tree":       {"max_depth": [5, 10, None]},
    "KNN":                 {"n_neighbors": [3, 5, 9]},
    "SVM (RBF)":           {"C": [1, 10], "gamma": ["scale"]},
    "SVM (Linear)":        {"C": [0.1, 1, 10]},
    "Random Forest":       {"n_estimators": [100, 200], "max_depth": [10, None]},
    "AdaBoost":            {"n_estimators": [50, 100]},
    "Bagging":             {"n_estimators": [50, 100]},
    "XGBoost":             {"n_estimators": [100, 200], "max_depth": [3, 5]},
    "Stacking (RF+SVM)":   {"rf__n_estimators": [100], "rf__max_depth": [5, 10]},
}
PARAM_GRIDS_B = {k: {f"model__{p}": v for p, v in g.items()}
                 for k, g in PARAM_GRIDS_A.items()}

# ============================================================
# VERİ & HFS
# ============================================================
print("=" * 60)
print(f"Figure 7 — ROC Curves (split_seed={SPLIT_SEED})")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
X_raw = data[FEATURE_COLS]
y = data[TARGET_COL] - 1

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y)

scaler = StandardScaler()
X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=FEATURE_COLS)
X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=FEATURE_COLS)

df_hfs = X_tr.copy(); df_hfs["target"] = y_tr.values
feats_B = hybrid_feature_selection(df_hfs, "target")
print(f"HFS features ({len(feats_B)}): {feats_B}\n")

X_tr_B, X_te_B = X_tr[feats_B], X_te[feats_B]

classes = sorted(y.unique())
y_te_bin = label_binarize(y_te, classes=classes)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

# ============================================================
# EĞİTİM + ROC
# ============================================================

roc_data = {}   # {model: {"A": (fpr, tpr, auc_macro, per_class), "B": ...}}

for name in MODEL_ORDER:
    print(f"  {name}...", end=" ", flush=True)
    roc_data[name] = {}
    for pipeline in ["A", "B"]:
        gs_jobs = 1 if "Stacking" in name else -1
        if pipeline == "A":
            model = get_models()[name]
            grid = GridSearchCV(model, PARAM_GRIDS_A[name], cv=cv,
                                scoring='roc_auc_ovr_weighted', n_jobs=gs_jobs)
            grid.fit(X_tr, y_tr)
            y_prob = grid.best_estimator_.predict_proba(X_te)
        else:
            pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)),
                                ("model", get_models()[name])])
            grid = GridSearchCV(pipe, PARAM_GRIDS_B[name], cv=cv,
                                scoring='roc_auc_ovr_weighted', n_jobs=gs_jobs)
            grid.fit(X_tr_B, y_tr)
            y_prob = grid.best_estimator_.predict_proba(X_te_B)

        # sınıf bazlı ROC
        per_class = {}
        for i, c in enumerate(classes):
            fpr_c, tpr_c, _ = roc_curve(y_te_bin[:, i], y_prob[:, i])
            per_class[c] = (fpr_c, tpr_c, auc(fpr_c, tpr_c))

        # macro-average ROC
        all_fpr = np.unique(np.concatenate([per_class[c][0] for c in classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in classes:
            mean_tpr += np.interp(all_fpr, per_class[c][0], per_class[c][1])
        mean_tpr /= len(classes)
        roc_data[name][pipeline] = (all_fpr, mean_tpr, auc(all_fpr, mean_tpr), per_class)
    print("done")

print("\nGenerating figure...")

# ============================================================
# PLOT — 2 sütun (A vs B) × 5 satır = 10 model
# ============================================================

n_models = len(MODEL_ORDER)
n_rows = 5
fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.4 * n_rows),
                         gridspec_kw={'wspace': 0.24, 'hspace': 0.45})

CLASS_COLORS = ["#c44e52", "#dd8452", "#4c72b0"]

for idx, name in enumerate(MODEL_ORDER):
    r, c = divmod(idx, 2)
    ax = axes[r, c]

    for pipeline, style, base_lw in [("A", "--", 1.6), ("B", "-", 2.1)]:
        fpr, tpr, auc_val, _ = roc_data[name][pipeline]
        label = ("without HFS" if pipeline == "A" else "HFS + SMOTE")
        color = "#888888" if pipeline == "A" else "#c44e52"
        ax.plot(fpr, tpr, style, color=color, lw=base_lw,
                label=f"{label} (AUC={auc_val:.4f})", zorder=3)

    ax.plot([0, 1], [0, 1], ":", color="black", lw=1, alpha=0.5, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(name, fontsize=11, fontweight="bold", pad=6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    if r == n_rows - 1:
        ax.set_xlabel("False Positive Rate", fontsize=10)
    if c == 0:
        ax.set_ylabel("True Positive Rate", fontsize=10)

fig.suptitle(
    "ROC Curves (macro-average, one-vs-rest)\n"
    "without HFS vs. HFS + SMOTE",
    fontsize=13, fontweight="bold", y=0.997)

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nFigure saved -> {OUTPUT_PATH}")

# ============================================================
# CSV özet
# ============================================================
rows = []
for name in MODEL_ORDER:
    for pipeline in ["A", "B"]:
        _, _, auc_macro, per_class = roc_data[name][pipeline]
        row = {"model": name, "pipeline": pipeline, "auc_macro": round(auc_macro, 5)}
        for c in classes:
            key = CLASS_NAMES[c].lower().replace(" ", "_")
            row[f"auc_{key}"] = round(per_class[c][2], 5)
        rows.append(row)
pd.DataFrame(rows).to_csv("Figure7_roc_auc_values.csv", index=False)
print("AUC values saved -> Figure7_roc_auc_values.csv")
