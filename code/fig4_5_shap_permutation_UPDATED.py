# ============================================================
# SHAP & Permutation Importance — Thyroid Disease (Multiclass)
# Median-Performance Run (Run #5, Pipeline B mean ROC-AUC = 0.9929)
#
# Çıktılar:
#   - 4 SHAP plots:    LR, DT, RF, XGBoost
#   - 6 Permutation plots: KNN, SVM RBF, SVM Linear,
#                          AdaBoost, Bagging, Stacking
#   - 2 CSV files: shap_importance.csv, permutation_importance.csv
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
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
from sklearn.inspection import permutation_importance

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
SPLIT_SEED    = 5       # Median-performance run from 10-run aggregate report
INTERNAL_SEED = 42

DATA_PATH  = "thyroid_ann_7200.csv"                      # ← Veri seti yolunu güncelle
OUTPUT_DIR = "shap_permutation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL  = 'target'
# NOT: SHAP değerleri sınıflar arası ortalandığı için bu sözlük grafiklerde
# görünmez; yalnızca tutarlılık ve olası sınıf-bazlı analizler için tutulur.
# Adlandırma hormon profiline göre doğrulanmıştır (bkz. makale, Section 2.1).
CLASS_NAMES = {0: "Overt hypothyroid", 1: "Subclinical hypothyroid", 2: "Euthyroid"}

HFS_THETA = 0.9
HFS_W     = 0.5
HFS_K     = 8


# ============================================================
# HFS ALGORİTMASI
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
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(pearson_correlation(X[:, i], X[:, j])) > theta:
                redundant.add(features[j])

    non_redundant = [f for f in features if f not in redundant]
    print(f"  Pearson elenen: {list(redundant)}, Kalan: {len(non_redundant)}")

    feature_scores = []
    for f in non_redundant:
        X_f = df[f].values
        ig  = information_gain(X_f, y)
        gr  = gain_ratio(X_f, y)
        hs  = w * ig + (1 - w) * gr
        feature_scores.append((f, ig, gr, hs))

    feature_scores.sort(key=lambda x: x[3], reverse=True)
    selected = [f for f, ig, gr, hs in feature_scores[:k]]
    print(f"  Seçilen {k}: {selected}")
    return selected, feature_scores


# ============================================================
# MODEL TANIMLARI
# ============================================================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=INTERNAL_SEED),
        "KNN":                 KNeighborsClassifier(),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, C=1,
                                   decision_function_shape='ovr', random_state=INTERNAL_SEED),
        "SVM (Linear)":        SVC(kernel="linear", probability=True, C=1,
                                   decision_function_shape='ovr', random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, algorithm="SAMME",
                                                  random_state=INTERNAL_SEED),   # REV: pinned
        "Bagging":             BaggingClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
        "Stacking (RF+SVM)":   StackingClassifier(
            estimators=[
                ('rf',  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=INTERNAL_SEED)),
                ('svm', SVC(kernel="rbf", probability=True, C=1, random_state=INTERNAL_SEED))
            ],
            final_estimator=LogisticRegression(random_state=INTERNAL_SEED)
        ),
    }

PARAM_GRIDS_B = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10], "model__solver": ["lbfgs"]},
    "Decision Tree":       {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
    "KNN":                 {"model__n_neighbors": [3, 5, 7, 9, 11], "model__metric": ["euclidean", "manhattan"]},
    "SVM (RBF)":           {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto"]},
    "SVM (Linear)":        {"model__C": [0.01, 0.1, 1, 10]},
    "Random Forest":       {"model__n_estimators": [100, 200, 300], "model__max_depth": [5, 10, None]},
    "AdaBoost":            {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 1.0]},
    "Bagging":             {"model__n_estimators": [50, 100, 200], "model__max_samples": [0.5, 0.7, 1.0]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7],
                            "model__learning_rate": [0.01, 0.1, 0.3]},
    "Stacking (RF+SVM)":   {"model__rf__n_estimators": [100, 200], "model__rf__max_depth": [5, 10]},
}


# ============================================================
# Pipeline B reproduction — median-performance run
# ============================================================

print("=" * 60)
print(f"Reproducing Pipeline B with split_seed={SPLIT_SEED}")
print(f"(Median-performance run, Pipeline B mean ROC-AUC = 0.9929)")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
X_raw = data[FEATURE_COLS]
y     = data[TARGET_COL] - 1  # 1,2,3 → 0,1,2

print(f"Data: {len(data)} samples, {len(FEATURE_COLS)} features")
print(f"Classes: {y.value_counts().sort_index().to_dict()}")

# Outer split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
)

# Scale (fit on train only — no imputation needed)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURE_COLS)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test_raw), columns=FEATURE_COLS)

# HFS
df_for_hfs = X_train_scaled.copy()
df_for_hfs["target"] = y_train.values
features_B, _ = hybrid_feature_selection(df_for_hfs, "target")

X_train_B = X_train_scaled[features_B]
X_test_B  = X_test_scaled[features_B]

# GridSearch + fit all 10 models
print(f"\nTraining 10 models with GridSearchCV (SMOTE in ImbPipeline)...")
models = get_models()
best_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

for name, model in models.items():
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=INTERNAL_SEED)),
        ("model", model)
    ])
    grid = GridSearchCV(pipe, PARAM_GRIDS_B[name],
                        cv=cv, scoring="roc_auc_ovr_weighted", n_jobs=-1)
    grid.fit(X_train_B, y_train)
    best_models[name] = grid.best_estimator_
    print(f"  ✓ {name} → {grid.best_params_}")

print("\nAll models trained.")


# ============================================================
# SHAP için SMOTE uygulanmış veri hazırla
# ============================================================

_smote = SMOTE(random_state=INTERNAL_SEED)
X_train_res_np, y_train_res = _smote.fit_resample(X_train_B, y_train)
X_train_exp = pd.DataFrame(X_train_res_np, columns=features_B)
X_test_exp  = pd.DataFrame(X_test_B.values, columns=features_B)

# İç modelleri yeniden eğit (SMOTE'lu veri üzerinde)
def get_inner(pipe):
    return pipe.named_steps['model']

fitted_inner = {}
for name, pipe in best_models.items():
    inner = get_inner(pipe)
    inner.fit(X_train_exp, y_train_res)
    fitted_inner[name] = inner


# ============================================================
# SHAP & Permutation Importance
# ============================================================

print("\n" + "=" * 60)
print("Generating SHAP & Permutation Importance plots")
print("=" * 60)

shap_records = []
perm_records = []

for name, model in fitted_inner.items():
    print(f"\n→ {name}")

    try:
        # ── SHAP: Logistic Regression ──
        if name == "Logistic Regression":
            print("  → SHAP (LinearExplainer)")
            explainer = shap.LinearExplainer(model, X_train_exp)
            sv = explainer(X_test_exp)

            # Multiclass: (n_samples, n_features, n_classes) → average across classes
            if len(sv.values.shape) == 3:
                vals = np.abs(sv.values).mean(axis=2)
            else:
                vals = sv.values

            plt.figure(figsize=(8, 5))
            shap.summary_plot(vals, X_test_exp, show=False)
            plt.title(f"{name} — SHAP (Thyroid)", fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"shap_{name.replace(' ', '_')}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            for f, v in zip(features_B, np.abs(vals).mean(axis=0)):
                shap_records.append({"model": name, "feature": f, "shap_mean": v})

        # ── SHAP: Decision Tree / XGBoost ──
        elif name in ["Decision Tree", "XGBoost"]:
            print(f"  → SHAP (Explainer)")
            explainer = shap.Explainer(model, X_train_exp)
            sv = explainer(X_test_exp)

            if len(sv.values.shape) == 3:
                vals = np.abs(sv.values).mean(axis=2)
            else:
                vals = sv.values

            plt.figure(figsize=(8, 5))
            shap.summary_plot(vals, X_test_exp, show=False)
            plt.title(f"{name} — SHAP (Thyroid)", fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"shap_{name.replace(' ', '_')}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            for f, v in zip(features_B, np.abs(vals).mean(axis=0)):
                shap_records.append({"model": name, "feature": f, "shap_mean": v})

        # ── SHAP: Random Forest (TreeExplainer) ──
        elif name == "Random Forest":
            print("  → SHAP (TreeExplainer)")
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test_exp, check_additivity=False)

            # Multiclass: list of arrays [class0, class1, class2]
            if isinstance(sv, list):
                vals = np.mean([np.abs(s) for s in sv], axis=0)
            elif len(np.array(sv).shape) == 3:
                vals = np.abs(np.array(sv)).mean(axis=2)
            else:
                vals = np.abs(sv)

            plt.figure(figsize=(8, 5))
            shap.summary_plot(vals, X_test_exp, show=False)
            plt.title(f"{name} — SHAP (Thyroid)", fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"shap_{name.replace(' ', '_')}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            for f, v in zip(features_B, np.abs(vals).mean(axis=0).flatten()):
                shap_records.append({"model": name, "feature": f, "shap_mean": v})

        # ── Permutation: KNN, SVM, AdaBoost, Bagging, Stacking ──
        elif name in ["KNN", "SVM (RBF)", "SVM (Linear)",
                      "AdaBoost", "Bagging", "Stacking (RF+SVM)"]:
            print(f"  → Permutation Importance")
            result = permutation_importance(
                model, X_test_exp, y_test,
                n_repeats=10, random_state=INTERNAL_SEED, n_jobs=-1,
                scoring='f1_macro'   # REV: was 'accuracy' (R2 c.4)
            )

            sorted_idx = result.importances_mean.argsort()
            plt.figure(figsize=(8, 5))
            plt.barh(range(len(sorted_idx)),
                     result.importances_mean[sorted_idx],
                     xerr=result.importances_std[sorted_idx],
                     color="steelblue", alpha=0.8)
            plt.yticks(range(len(sorted_idx)),
                       np.array(features_B)[sorted_idx])
            plt.xlabel("Importance Mean")
            plt.title(f"Permutation Importance — {name} (Thyroid)",
                      fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
            plt.savefig(os.path.join(OUTPUT_DIR, f"perm_{safe_name}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            for f, m, s in zip(features_B,
                                result.importances_mean,
                                result.importances_std):
                perm_records.append({
                    "model": name, "feature": f,
                    "importance_mean": m, "importance_std": s
                })

        print(f"  ✓ Done")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback; traceback.print_exc()


# ============================================================
# Save CSVs
# ============================================================

shap_df = pd.DataFrame(shap_records)
shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)

perm_df = pd.DataFrame(perm_records)
perm_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)

print(f"\n{'=' * 60}")
print(f"All results saved to: {OUTPUT_DIR}/")
print(f"{'=' * 60}")
print(f"\nFiles produced:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")
