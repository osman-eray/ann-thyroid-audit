#!/usr/bin/env python3
# ============================================================
# explainability_by_mode.py
#
# Reviewer 1, round four, point 3.
#
# Section 3.10 reports the agreement between the hybrid selection score and
# what the fitted models actually use (SHAP and permutation importance). That
# agreement was measured on the subsets the RELEASED implementation selects,
# in which a mis-estimated binary indicator occupies a selection slot in five
# of ten splits. The reviewer asks for the same analysis on the subsets the
# scheme selects when the information gain of Equation (2) is estimated as
# specified, so that the general caution about filter scores rests on a figure
# the implementation does not contaminate.
#
# This script runs the ranking-agreement analysis of script 9 twice on
# identical splits and seeds, changing only the discrete_features argument of
# the information-gain call:
#
#   "auto"      exactly as released  -> must reproduce ranking_agreement.csv
#   "discrete"  as Equation (2) specifies (Boolean mask for the binary
#               indicators)
#
# Everything downstream is the released pipeline: SMOTE inside the folds,
# per-split grid search, LinearExplainer for logistic regression and
# TreeExplainer for the tree models, permutation importance scored on
# macro-F1 (the scoring the second revision adopted).
#
# Output:
#   ranking_agreement_by_mode.csv   per split, model and ranking pair
#   agreement_summary_by_mode.csv   means, SDs and the hypopituitary split
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

import shap

# ---- config, identical to script 9 -------------------------------------
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8
HFS_THETA = 0.90
HFS_W = 0.5
TEST_SIZE = 0.20
SHAP_SAMPLE = 300
SCORING_CV = "roc_auc_ovr_weighted"
PERM_SCORING = "f1_macro"          # the second revision's correction
CLASS_NAMES = {0: "overt_hypothyroid", 1: "subclinical_hypothyroid", 2: "euthyroid"}

PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Decision Tree":       {"model__max_depth": [5, 10, None]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric="mlogloss",
                                             objective="multi:softprob",
                                             random_state=INTERNAL_SEED),
    }


# ---- the selection scheme ----------------------------------------------

def information_gain(X_col, y, discrete):
    return mutual_info_classif(
        X_col.reshape(-1, 1), y,
        discrete_features=discrete,
        random_state=INTERNAL_SEED,
    )[0]


def feature_entropy_bits(X_col):
    n_unique = len(np.unique(X_col))
    if n_unique <= 10:
        _, counts = np.unique(X_col, return_counts=True)
    else:
        n_bins = max(10, int(np.sqrt(len(X_col))))
        counts = np.histogram(X_col, bins=n_bins)[0]
        counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def hfs_select_with_scores(Xtr, ytr, binary_cols, mode,
                           theta=HFS_THETA, w=HFS_W, k=K_SELECT):
    feats = list(Xtr.columns)
    X = Xtr.values
    redundant = set()
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = X[:, i], X[:, j]
            if a.std() == 0 or b.std() == 0:
                continue
            if abs(np.corrcoef(a, b)[0, 1]) > theta:
                redundant.add(feats[j])
    non_red = [f for f in feats if f not in redundant]

    scores = {}
    for f in non_red:
        xf = Xtr[f].values
        discrete = (f in binary_cols) if mode == "discrete" else "auto"
        ig = information_gain(xf, ytr, discrete)
        h = feature_entropy_bits(xf)
        gr = ig / (h + 1e-12) if h > 0 else 0.0
        scores[f] = w * ig + (1 - w) * gr
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ordered[:min(k, len(ordered))]]
    return selected, scores


# ---- per-class SHAP, as script 9 ---------------------------------------

def shap_values_per_class(model, X_background, X_explain, feat_names, model_name):
    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_background)
        sv = explainer.shap_values(X_explain)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain)

    out = {}
    nf = len(feat_names)
    if isinstance(sv, list):
        for c, arr in enumerate(sv):
            out[c] = np.asarray(arr)
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            if sv.shape[-1] == nf:
                for c in range(sv.shape[0]):
                    out[c] = sv[c]
            else:
                for c in range(sv.shape[2]):
                    out[c] = sv[:, :, c]
        elif sv.ndim == 2:
            out[0] = sv
    return out


# ---- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="thyroid_ann_7200.csv")
    ap.add_argument("--out-dir", default="explainability_by_mode_results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    FEATURE_COLS = [c for c in df.columns if c not in ("target", "target_label")]
    binary_cols = [c for c in FEATURE_COLS if df[c].nunique() <= 2]
    print("=" * 74)
    print("RANKING AGREEMENT — released estimator vs Equation (2)")
    print("=" * 74)
    print(f"records {len(df)} | features {len(FEATURE_COLS)} | binary {len(binary_cols)}")
    print(f"permutation scoring: {PERM_SCORING}")

    rows, sel_rows = [], []

    for mode in ("auto", "discrete"):
        for run_id in SPLIT_SEEDS:
            t0 = time.time()
            X = df[FEATURE_COLS]
            y = df["target"] - 1
            Xtr_raw, Xte_raw, ytr, yte = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=run_id, stratify=y)
            scaler = StandardScaler()
            Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
            Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)

            feats, hfs_scores = hfs_select_with_scores(
                Xtr, ytr.values, binary_cols, mode)
            for r, f in enumerate(feats, 1):
                sel_rows.append({"mode": mode, "run_id": run_id, "rank": r, "feature": f})

            XtrH, XteH = Xtr[feats], Xte[feats]
            n_ex = min(SHAP_SAMPLE, len(XteH))
            rng = np.random.RandomState(INTERNAL_SEED + run_id)
            ex_idx = rng.choice(len(XteH), n_ex, replace=False)
            X_explain = XteH.iloc[ex_idx]
            X_bg = XtrH.sample(min(100, len(XtrH)), random_state=INTERNAL_SEED)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

            for mname, model in get_models().items():
                pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)),
                                    ("model", model)])
                grid = GridSearchCV(pipe, PARAM_GRIDS[mname], cv=cv,
                                    scoring=SCORING_CV, n_jobs=-1)
                grid.fit(XtrH, ytr)
                fitted = grid.best_estimator_.named_steps["model"]

                sv_by_class = shap_values_per_class(fitted, X_bg, X_explain, feats, mname)

                perm = permutation_importance(
                    grid.best_estimator_, XteH, yte,
                    n_repeats=10, random_state=INTERNAL_SEED, n_jobs=-1,
                    scoring=PERM_SCORING)
                perm_imp = dict(zip(feats, perm.importances_mean))

                shap_overall = {}
                for f_i, f in enumerate(feats):
                    vals = [np.abs(mat[:, f_i]).mean()
                            for c, mat in sv_by_class.items() if c in CLASS_NAMES]
                    shap_overall[f] = float(np.mean(vals)) if vals else 0.0

                hfs_v = [hfs_scores.get(f, 0.0) for f in feats]
                shap_v = [shap_overall[f] for f in feats]
                perm_v = [perm_imp[f] for f in feats]

                for a_name, a_v, b_name, b_v in [
                    ("HFS", hfs_v, "SHAP", shap_v),
                    ("HFS", hfs_v, "Permutation", perm_v),
                    ("SHAP", shap_v, "Permutation", perm_v),
                ]:
                    rho, p = spearmanr(a_v, b_v)
                    rows.append({
                        "mode": mode, "run_id": run_id, "model": mname,
                        "ranking_a": a_name, "ranking_b": b_name,
                        "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else np.nan,
                        "p_value": round(float(p), 5) if not np.isnan(p) else np.nan,
                        "n_features": len(feats),
                    })
            print(f"  {mode:8s} run {run_id}  {time.time()-t0:6.1f}s  "
                  f"hypopituitary {'IN ' if 'hypopituitary' in feats else 'out'}",
                  flush=True)

    ag = pd.DataFrame(rows)
    ag.to_csv(os.path.join(args.out_dir, "ranking_agreement_by_mode.csv"), index=False)
    pd.DataFrame(sel_rows).to_csv(
        os.path.join(args.out_dir, "selected_by_mode.csv"), index=False)

    srows = []
    for mode in ("auto", "discrete"):
        for a_, b_ in [("HFS", "SHAP"), ("HFS", "Permutation"), ("SHAP", "Permutation")]:
            x = ag[(ag["mode"] == mode) & (ag.ranking_a == a_) & (ag.ranking_b == b_)]
            srows.append({
                "mode": mode, "pair": f"{a_} vs {b_}",
                "mean_rho": round(x.spearman_rho.mean(), 4),
                "sd_rho": round(x.spearman_rho.std(), 4),
                "mean_rho_squared": round((x.spearman_rho ** 2).mean(), 4),
                "squared_mean_rho": round(x.spearman_rho.mean() ** 2, 4),
                "n": len(x),
            })
    s = pd.DataFrame(srows)
    s.to_csv(os.path.join(args.out_dir, "agreement_summary_by_mode.csv"), index=False)
    print()
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
