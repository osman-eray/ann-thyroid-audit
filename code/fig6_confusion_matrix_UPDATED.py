# ============================================================
# Confusion Matrix Grid — Median-Performance Run (split_seed=5)
# Pipeline A (without HFS) vs Pipeline B (HFS+SMOTE)
# 4 model: SVM RBF, SVM Linear, Logistic Regression, KNN
#
# Cikti: Figure6_confusion_matrix_grid.png (300 DPI)
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
SPLIT_SEED    = 5       # Median-performance run
INTERNAL_SEED = 42

DATA_PATH   = "thyroid_ann_7200.csv"                        # <- Veri seti yolunu guncelle
OUTPUT_PATH = "Figure6_confusion_matrix_grid.png"

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

HFS_THETA = 0.9
HFS_W     = 0.5
HFS_K     = 8

# Sınıf etiketleri — hormon profiline göre doğrulanmış adlandırma
#   Class 0: TSH↑↑ (medyan 53.0), TT4↓ (31.6)  → aşikar hipotiroidi
#   Class 1: TSH↑  (medyan 9.7),  TT4 normal-alt → subklinik hipotiroidi
#   Class 2: ötiroid
CLASS_LABELS = ["Overt\nhypo", "Subclin.\nhypo", "Euthyroid"]

SELECTED_MODELS = [
    "SVM (RBF)",
    "SVM (Linear)",
    "Logistic Regression",
    "KNN",
]


# ============================================================
# HFS ALGORITMASI
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

    redundant = set()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if abs(pearson_correlation(X[:, i], X[:, j])) > theta:
                redundant.add(features[j])

    non_redundant = [f for f in features if f not in redundant]
    feature_scores = []
    for f in non_redundant:
        ig = information_gain(df[f].values, y)
        gr = gain_ratio(df[f].values, y)
        hs = w * ig + (1 - w) * gr
        feature_scores.append((f, hs))

    feature_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in feature_scores[:k]]
    return selected


# ============================================================
# MODEL TANIMLARI
# ============================================================

def get_model(name):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=INTERNAL_SEED),
        "KNN": KNeighborsClassifier(),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, C=1,
            decision_function_shape='ovr', random_state=INTERNAL_SEED),
        "SVM (Linear)": SVC(
            kernel="linear", probability=True, C=1,
            decision_function_shape='ovr', random_state=INTERNAL_SEED),
    }
    return models[name]

PARAM_GRIDS_A = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "KNN": {"n_neighbors": [3, 5, 9], "metric": ["euclidean", "manhattan"]},
    "SVM (RBF)": {"C": [1, 10], "gamma": ["scale", "auto"]},
    "SVM (Linear)": {"C": [0.1, 1, 10]},
}

PARAM_GRIDS_B = {
    k: {f"model__{p}": v for p, v in g.items()}
    for k, g in PARAM_GRIDS_A.items()
}


# ============================================================
# METRIC FONKSIYONLARI
# ============================================================

def macro_specificity(cm):
    specs = []
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return np.mean(specs)

def macro_sensitivity(cm):
    sens = []
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    return np.mean(sens)


# ============================================================
# VERI YUKLEME & ON ISLEME
# ============================================================

print("=" * 60)
print(f"Confusion Matrix Grid — Median-Performance Run (split_seed={SPLIT_SEED})")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
X_raw = data[FEATURE_COLS]
y = data[TARGET_COL] - 1   # 1,2,3 -> 0,1,2

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
)

scaler = StandardScaler()
X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=FEATURE_COLS)
X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=FEATURE_COLS)

# HFS
df_hfs = X_tr.copy()
df_hfs["target"] = y_tr.values
feats_B = hybrid_feature_selection(df_hfs, "target")
print(f"HFS features: {feats_B}")

X_tr_B = X_tr[feats_B]
X_te_B = X_te[feats_B]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
labels = sorted(y.unique())


# ============================================================
# PIPELINE A & B — GRIDSEARCH + CONFUSION MATRIX
# ============================================================

cm_results = {}

for name in SELECTED_MODELS:
    print(f"  {name}...", end=" ", flush=True)

    # Pipeline A: without HFS, no SMOTE
    grid_a = GridSearchCV(
        get_model(name), PARAM_GRIDS_A[name],
        cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=-1
    )
    grid_a.fit(X_tr, y_tr)
    y_pred_a = grid_a.best_estimator_.predict(X_te)
    cm_a = confusion_matrix(y_te, y_pred_a, labels=labels)

    # Pipeline B: HFS + SMOTE
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=INTERNAL_SEED)),
        ("model", get_model(name))
    ])
    grid_b = GridSearchCV(
        pipe, PARAM_GRIDS_B[name],
        cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=-1
    )
    grid_b.fit(X_tr_B, y_tr)
    y_pred_b = grid_b.best_estimator_.predict(X_te_B)
    cm_b = confusion_matrix(y_te, y_pred_b, labels=labels)

    cm_results[name] = {"A": cm_a, "B": cm_b}
    print("done")

print("\nAll models trained. Generating figure...")


# ============================================================
# PLOT
# ============================================================

n_models = len(SELECTED_MODELS)
fig, axes = plt.subplots(
    n_models, 2,
    figsize=(11, 3.8 * n_models),
    gridspec_kw={'wspace': 0.30, 'hspace': 0.50}
)

for i, model in enumerate(SELECTED_MODELS):
    for j, (pipeline, condition) in enumerate([
        ("A", "without HFS"),
        ("B", "with HFS + SMOTE"),
    ]):
        cm = cm_results[model][pipeline]
        spec = macro_specificity(cm)
        sens = macro_sensitivity(cm)
        acc = np.trace(cm) / cm.sum()
        ax = axes[i, j]

        # Satir bazli yuzde normalize (renk skalasi icin)
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        sns.heatmap(
            cm_pct,
            annot=False,
            cmap="Blues",
            vmin=0, vmax=100,
            cbar=False,
            linewidths=1.0, linecolor="gray",
            ax=ax,
            square=True,
        )

        # Manuel annotation: sayi + yuzde
        for ri in range(3):
            for ci in range(3):
                count = cm[ri, ci]
                pct = cm_pct[ri, ci]
                color = "white" if pct > 50 else "black"
                ax.text(
                    ci + 0.5, ri + 0.5,
                    f"{count}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color=color
                )

        ax.set_title(
            f"{condition}\n"
            f"Sensitivity={sens:.3f}  Specificity={spec:.3f}  Acc={acc:.3f}",
            fontsize=10, pad=8
        )

        if i == n_models - 1:
            ax.set_xlabel("Predicted", fontsize=10)
        else:
            ax.set_xlabel("")

        if j == 0:
            ax.set_ylabel(
                f"{model}\n\nTrue",
                fontsize=11, fontweight='bold', labelpad=8
            )
        else:
            ax.set_ylabel("")

        ax.set_xticklabels(CLASS_LABELS, rotation=0, fontsize=9)
        ax.set_yticklabels(CLASS_LABELS, rotation=90, fontsize=9, va='center')

fig.suptitle(
    "Confusion Matrix Comparison\n"
    "without HFS vs. with HFS + SMOTE",
    fontsize=13, fontweight='bold', y=0.998
)

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

print(f"\nFigure saved -> {OUTPUT_PATH}")
