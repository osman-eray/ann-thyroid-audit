#!/usr/bin/env python3
# ============================================================
# hfs_discrete_estimation.py
#
# Reviewer 1, round three, condition of acceptance.
#
# The released implementation of the hybrid selection scheme estimates the
# information gain of Equation (2) with scikit-learn's mutual_info_classif and
# discrete_features='auto'. On a dense array that argument resolves to False,
# so every attribute — including the fifteen binary indicators — is treated as
# continuous and handled by the k-nearest-neighbour estimator, which perturbs
# its input with a small amount of noise.
#
# This script runs the scheme twice on identical splits and seeds, changing
# only that one argument:
#
#   "auto"      exactly as released  (discrete_features='auto')
#   "discrete"  as Equation (2) specifies (Boolean mask: True for the binary
#               indicators, False for the six continuous assays)
#
# and evaluates both under the downstream pipeline of Table 7 (SMOTE inside
# the folds + tuned random forest) and the three-classifier setup of Table 9.
# It reproduces Table 9b of the manuscript.
#
# Nothing else in the repository is modified: the released scripts keep the
# released behaviour, so the deposited per-split results stand unchanged.
#
# Usage:
#   python hfs_discrete_estimation.py
#   python hfs_discrete_estimation.py --data ../data/thyroid_ann_7200.csv \
#                                     --out-dir ../results/discrete_estimation
# ============================================================

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG — identical to the released scripts
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8
HFS_THETA = 0.90
TEST_SIZE = 0.20
SCORING_CV = "roc_auc_ovr_weighted"

# Table 7 downstream pipeline
EVAL_GRID_RF = {"model__n_estimators": [100, 200], "model__max_depth": [10, None]}

# Table 9 setup: three classifiers averaged
GRIDS_T9 = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}


def models_t9():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric="mlogloss",
                                             objective="multi:softprob",
                                             random_state=INTERNAL_SEED),
    }


# ============================================================
# The hybrid selection scheme (Equations 1-7)
# ============================================================

def pearson_correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def information_gain(X_col, y, discrete):
    """Equation (2).

    `discrete` is the value passed to scikit-learn's discrete_features.
    The released code passes 'auto'; passing True for a binary indicator
    computes the discrete mutual information the equation defines.

    NOTE ON UNITS: mutual_info_classif returns nats. The feature entropy
    below is computed in bits. Every feature is scaled by the same ln 2,
    so no ranking is affected, but the two quantities are reported in
    consistent units wherever they are printed.
    """
    return mutual_info_classif(
        X_col.reshape(-1, 1), y,
        discrete_features=discrete,
        random_state=INTERNAL_SEED,
    )[0]


def feature_entropy_bits(X_col):
    """H(f) of Equation (6), in bits."""
    n_unique = len(np.unique(X_col))
    if n_unique <= 10:
        _, counts = np.unique(X_col, return_counts=True)
    else:
        n_bins = max(10, int(np.sqrt(len(X_col))))
        counts = np.histogram(X_col, bins=n_bins)[0]
        counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def hybrid_feature_selection(X_scaled, y, binary_cols, mode,
                             theta=HFS_THETA, w=0.5, k=K_SELECT):
    """Returns (selected features, per-feature score table).

    mode = "auto"     -> discrete_features='auto'  (released behaviour)
    mode = "discrete" -> discrete_features=True for the binary indicators
    """
    features = list(X_scaled.columns)

    # Stage 1 - redundancy elimination (Equation 1)
    redundant = set()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if abs(pearson_correlation(X_scaled.iloc[:, i], X_scaled.iloc[:, j])) > theta:
                redundant.add(features[j])
    non_redundant = [f for f in features if f not in redundant]

    # Stages 2-3 - relevance and hybrid score (Equations 2-7)
    rows = []
    for f in non_redundant:
        col = X_scaled[f].values
        discrete = (f in binary_cols) if mode == "discrete" else "auto"
        ig_nats = information_gain(col, y, discrete)
        ig_bits = ig_nats / np.log(2)
        h_bits = feature_entropy_bits(col)
        gr_bits = ig_bits / (h_bits + 1e-12) if h_bits > 0 else 0.0
        # the released score divides the nats-valued IG by the bits-valued
        # entropy; kept as released so that the two modes differ in one
        # argument only, and reported in consistent units below
        gr_released = ig_nats / (h_bits + 1e-12) if h_bits > 0 else 0.0
        rows.append({
            "feature": f,
            "ig_nats": ig_nats,
            "ig_bits": ig_bits,
            "entropy_bits": h_bits,
            "gain_ratio_bits_per_bit": gr_bits,
            "hybrid_score": w * ig_nats + (1 - w) * gr_released,
        })

    table = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.head(k)["feature"].tolist(), table


# ============================================================
# Evaluation
# ============================================================

def macro_specificity(cm):
    out = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(out))


def evaluate(Xtr, ytr, Xte, yte, model, grid):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
    pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)), ("model", model)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring=SCORING_CV, n_jobs=-1)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    proba = best.predict_proba(Xte)
    pred = best.predict(Xte)
    return (
        roc_auc_score(yte, proba, multi_class="ovr", average="weighted"),
        f1_score(yte, pred, average="macro", zero_division=0),
        macro_specificity(confusion_matrix(yte, pred, labels=sorted(pd.Series(yte).unique()))),
    )


def split_data(df, features, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        df[features], y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return (pd.DataFrame(scaler.transform(Xtr), columns=features), ytr,
            pd.DataFrame(scaler.transform(Xte), columns=features), yte)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="thyroid_ann_7200.csv")
    ap.add_argument("--out-dir", default="discrete_estimation_results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    features = [c for c in df.columns if c not in ("target", "target_label")]
    y = df["target"].values - 1
    binary_cols = [c for c in features if df[c].nunique() <= 2]

    print("=" * 74)
    print("HYBRID SELECTION SCHEME — released estimator vs Equation (2)")
    print("=" * 74)
    print(f"records {len(df)} | features {len(features)} | binary indicators {len(binary_cols)}")
    print(f"classes {dict(pd.Series(y + 1).value_counts().sort_index())}")

    # ---- selection: per-split ranks and scores -------------------------
    sel_rows, rank_rows = [], []
    for mode in ("auto", "discrete"):
        for seed in SPLIT_SEEDS:
            Xs, ytr, _, _ = split_data(df, features, y, seed)
            chosen, table = hybrid_feature_selection(Xs, ytr, binary_cols, mode)
            table.insert(0, "mode", mode)
            table.insert(1, "run_id", seed)
            rank_rows.append(table)
            for r, f in enumerate(chosen, start=1):
                sel_rows.append({"mode": mode, "run_id": seed, "rank": r, "feature": f})

    pd.concat(rank_rows, ignore_index=True).to_csv(
        os.path.join(args.out_dir, "hfs_scores_by_mode.csv"), index=False)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(os.path.join(args.out_dir, "hfs_selected_by_mode.csv"), index=False)

    print("\n--- selection frequency of each attribute (of 10 splits) ---")
    freq = sel.pivot_table(index="feature", columns="mode", values="run_id",
                           aggfunc="count").fillna(0).astype(int)
    freq = freq.reindex(freq.sum(axis=1).sort_values(ascending=False).index)
    print(freq.to_string())

    # ---- Table 7 pipeline ---------------------------------------------
    print("\n--- Table 7 downstream pipeline (SMOTE + tuned random forest) ---")
    t7_rows = []
    for mode in ("auto", "discrete"):
        per_split = []
        t0 = time.time()
        for seed in SPLIT_SEEDS:
            Xs, ytr, Xts, yte = split_data(df, features, y, seed)
            chosen, _ = hybrid_feature_selection(Xs, ytr, binary_cols, mode)
            roc, f1m, spec = evaluate(
                Xs[chosen].values, ytr, Xts[chosen].values, yte,
                RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
                EVAL_GRID_RF)
            per_split.append((roc, f1m, spec))
            t7_rows.append({"mode": mode, "run_id": seed, "roc_auc_ovr": roc,
                            "f1_macro": f1m, "specificity_macro": spec})
        a = np.array(per_split)
        print(f"  {mode:9s} ROC-AUC {a[:,0].mean():.4f} +/- {a[:,0].std(ddof=1):.4f}"
              f"   macro-F1 {a[:,1].mean():.4f} +/- {a[:,1].std(ddof=1):.4f}"
              f"   specificity {a[:,2].mean():.4f} +/- {a[:,2].std(ddof=1):.4f}"
              f"   ({time.time()-t0:.0f}s)")
    pd.DataFrame(t7_rows).to_csv(
        os.path.join(args.out_dir, "table7_row_by_mode.csv"), index=False)

    # ---- Table 9 weight sweep -----------------------------------------
    print("\n--- Table 9 setup (three classifiers averaged), weight sweep ---")
    t9_rows = []
    for mode in ("auto", "discrete"):
        for w in (0.5, 1.0):
            per = []
            t0 = time.time()
            for seed in SPLIT_SEEDS:
                Xs, ytr, Xts, yte = split_data(df, features, y, seed)
                chosen, _ = hybrid_feature_selection(Xs, ytr, binary_cols, mode, w=w)
                for name, model in models_t9().items():
                    roc, f1m, spec = evaluate(Xs[chosen].values, ytr,
                                              Xts[chosen].values, yte,
                                              model, GRIDS_T9[name])
                    per.append((roc, f1m, spec))
                    t9_rows.append({"mode": mode, "w": w, "run_id": seed, "model": name,
                                    "roc_auc_ovr": roc, "f1_macro": f1m,
                                    "specificity_macro": spec})
            a = np.array(per)
            print(f"  {mode:9s} w={w:<4} ROC-AUC {a[:,0].mean():.4f}"
                  f"   macro-F1 {a[:,1].mean():.4f}"
                  f"   specificity {a[:,2].mean():.4f}   ({time.time()-t0:.0f}s)")
    pd.DataFrame(t9_rows).to_csv(
        os.path.join(args.out_dir, "table9_weight_sweep_by_mode.csv"), index=False)

    print("\n" + "=" * 74)
    print(f"written to {args.out_dir}/")
    print("  hfs_scores_by_mode.csv          per-feature IG, entropy and gain ratio")
    print("  hfs_selected_by_mode.csv        the selected subset in each split")
    print("  table7_row_by_mode.csv          per-split Table 7 values")
    print("  table9_weight_sweep_by_mode.csv per-split weight sweep")
    print("=" * 74)


if __name__ == "__main__":
    main()
