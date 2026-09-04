#!/usr/bin/env python3
"""
perclass_and_dispersion.py
Scientific Reports 7c9aaee8 — Reviewer 2 comment 6 (item 5) and Reviewer 1 MAJOR 3/4 (item 6).

Derives everything from the per-split CSVs already produced by
1.thyroid_ablation_ann7200_SAMME.py and 5.thyroid_oversampling_comparison_ann7200_UPDATED.py.
No model is refitted.

  A. Class-wise sensitivity, specificity and PR-AUC under the full pipeline and at baseline
  B. Paired baseline-vs-pipeline test on minority-class PR-AUC (Holm-corrected)
  C. Table 8 rebuilt with dispersion, plus the variance decomposition of that dispersion
  D. Friedman and TOST across the seven balancing strategies
  E. Verification of the Table 7 confidence-interval sentence flagged by Reviewer 1

Usage:
    python perclass_and_dispersion.py --results-dir . --out-dir ./perclass_out
"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

# CSV column suffixes carry the original (incorrect) class names; map to the
# laboratory-phenotype designations used in the manuscript.
CLASSES = {"hyperthyroid": "Overt hypothyroid",
           "subnormal":    "Subclinical hypothyroid",
           "normal":       "Euthyroid"}
MINORITY = "subnormal"          # the stratum where ROC-AUC is most optimistic
MARGINS = {"f1_macro": 0.02, "roc_auc_ovr": 0.005, "specificity_macro": 0.01}


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p); m = len(p)
    adj = np.clip(np.maximum.accumulate((m - np.arange(m)) * p[o]), 0, 1)
    out = np.empty(m); out[o] = adj
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./perclass_out"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    abl = pd.read_csv(a.results_dir / "thyroid_ablation_results" / "ablation_all_runs.csv")
    abl = abl[abl.model != "DummyClassifier"]

    # ---------- A. class-wise metrics --------------------------------------
    for cond, label in (("B_HFS_SMOTE", "full pipeline"), ("A_baseline", "baseline")):
        d = abl[abl.condition == cond]
        rows = []
        for m, g in d.groupby("model", sort=False):
            r = {"model": m, "ROC_AUC_ovr": g.roc_auc_ovr.mean(), "ROC_AUC_sd": g.roc_auc_ovr.std()}
            for key, name in CLASSES.items():
                r[f"sens_{name}"]   = g[f"sens_{key}"].mean()
                r[f"spec_{name}"]   = g[f"spec_{key}"].mean()
                r[f"prauc_{name}"]  = g[f"prauc_{key}"].mean()
                r[f"prauc_sd_{name}"] = g[f"prauc_{key}"].std()
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(a.out_dir / f"perclass_{cond}.csv", index=False)
        say("=" * 78)
        say(f"A. CLASS-WISE METRICS — {label} ({cond}), mean over 10 splits")
        say("=" * 78)
        say(f"{'Model':22s}{'ROC-AUC':>9s}" +
            "".join(f"{n[:11]+' sens':>18s}{n[:11]+' PR-AUC':>20s}" for n in CLASSES.values()))
        for _, r in t.iterrows():
            say(f"{r.model:22s}{r.ROC_AUC_ovr:9.4f}" +
                "".join(f"{r['sens_'+n]:18.3f}{r['prauc_'+n]:20.3f}" for n in CLASSES.values()))
        say(f"\n  pooled: macro ROC-AUC {d.roc_auc_ovr.mean():.4f}  vs  "
            + ", ".join(f"{n} PR-AUC {d['prauc_'+k].mean():.3f}" for k, n in CLASSES.items()))
        say()

    # ---------- B. does the pipeline hurt the minority PR-AUC? -------------
    say("=" * 78)
    say(f"B. BASELINE vs FULL PIPELINE on {CLASSES[MINORITY]} PR-AUC (paired, 10 splits)")
    say("=" * 78)
    rows = []
    for m in abl.model.unique():
        A = abl[(abl.condition == "A_baseline") & (abl.model == m)].sort_values("run_id")
        B = abl[(abl.condition == "B_HFS_SMOTE") & (abl.model == m)].sort_values("run_id")
        d = B[f"prauc_{MINORITY}"].values - A[f"prauc_{MINORITY}"].values
        rows.append({"model": m, "auc_base": A.roc_auc_ovr.mean(), "auc_pipe": B.roc_auc_ovr.mean(),
                     "prauc_base": A[f"prauc_{MINORITY}"].mean(),
                     "prauc_pipe": B[f"prauc_{MINORITY}"].mean(),
                     "p": 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue})
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p.to_numpy())
    r["auc_delta"] = r.auc_pipe - r.auc_base
    r["prauc_delta"] = r.prauc_pipe - r.prauc_base
    r.to_csv(a.out_dir / "prauc_baseline_vs_pipeline.csv", index=False)
    say(f"{'Model':22s}{'ROC-AUC Δ':>12s}{'PR-AUC Δ':>12s}{'p_holm':>10s}")
    for _, x in r.iterrows():
        say(f"{x.model:22s}{x.auc_delta:+12.4f}{x.prauc_delta:+12.4f}{x.p_holm:10.4f}")
    sig = r[(r.prauc_delta < 0) & (r.p_holm < 0.05)]
    say(f"\n  significant decreases after Holm: {len(sig)} of {len(r)} classifiers "
        f"({', '.join(sig.model)})")
    if len(sig):
        say(f"  mean ROC-AUC change in those classifiers : {sig.auc_delta.mean():+.4f}")
        say(f"  mean PR-AUC change in those classifiers  : {sig.prauc_delta.mean():+.4f}  "
            f"({abs(sig.prauc_delta.mean() / sig.auc_delta.mean()):.0f}x larger)")
    say()

    # ---------- C/D. balancing strategies ---------------------------------
    ov = pd.read_csv(a.results_dir / "thyroid_oversampling_results" / "oversampling_all_runs.csv")
    ov["strategy"] = ov.strategy.fillna("None (no balancing)")
    say("=" * 78)
    say("C. TABLE 8 WITH DISPERSION (mean ± SD over 3 classifiers x 10 splits)")
    say("=" * 78)
    g = ov.groupby("strategy")[["roc_auc_ovr", "f1_macro", "specificity_macro"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.sort_values("f1_macro_mean", ascending=False)
    g.to_csv(a.out_dir / "table8_with_dispersion.csv")
    for s, x in g.iterrows():
        say(f"  {s:22s} ROC-AUC {x.roc_auc_ovr_mean:.4f}±{x.roc_auc_ovr_std:.4f}   "
            f"macro-F1 {x.f1_macro_mean:.4f}±{x.f1_macro_std:.4f}   "
            f"spec {x.specificity_macro_mean:.4f}±{x.specificity_macro_std:.4f}")
    b = ov[ov.strategy != "None (no balancing)"]
    say(f"\n  pooled SD of macro-F1 over the 30 runs : {b.f1_macro.std():.4f}")
    say("  SD within each classifier              : " +
        ", ".join(f"{k} {v:.4f}" for k, v in b.groupby('model').f1_macro.std().items()))
    say("  -> most of the dispersion is between classifiers, not between strategies or splits;\n"
        "     say so in the caption or the SD will be read as strategy-level noise.")

    say("\n" + "=" * 78)
    say("D. FRIEDMAN AND EQUIVALENCE ACROSS THE SEVEN BALANCING STRATEGIES")
    say("=" * 78)
    b = b.copy(); b["block"] = b.model + "|" + b.run_id.astype(str)
    for metric, margin in MARGINS.items():
        w = b.pivot(index="block", columns="strategy", values=metric)
        mat = w.to_numpy()
        chi2, p = stats.friedmanchisquare(*mat.T)
        ranks = np.apply_along_axis(stats.rankdata, 1, -mat).mean(axis=0)
        eq = sum(max(stats.wilcoxon(mat[:, i] - mat[:, j] + margin, alternative="greater").pvalue,
                     stats.wilcoxon(mat[:, i] - mat[:, j] - margin, alternative="less").pvalue) < 0.05
                 for i, j in itertools.combinations(range(mat.shape[1]), 2))
        npairs = mat.shape[1] * (mat.shape[1] - 1) // 2
        say(f"  {metric:18s} chi2(6)={chi2:7.3f}  p={p:.3e}  "
            f"spread={mat.mean(0).max()-mat.mean(0).min():.4f}  "
            f"equivalent within ±{margin}: {eq}/{npairs}")
        say("     ranks (1=best): " + ", ".join(f"{w.columns[i]} {ranks[i]:.2f}" for i in np.argsort(ranks)))

    # ---------- E. the Table 7 CI sentence --------------------------------
    say("\n" + "=" * 78)
    say("E. THE TABLE 7 SENTENCE FLAGGED BY REVIEWER 1")
    say("=" * 78)
    sds = np.array([0.0003, 0.0003, 0.0004, 0.0005, 0.0005, 0.0006, 0.0008, 0.0017, 0.0022])
    names = ["RFE", "LASSO", "TreeImportance", "MutualInfo", "SelectKBest",
             "Boruta-like", "ReliefF", "Hybrid", "mRMR"]
    tcrit = stats.t.ppf(0.975, 9)
    widths = 2 * tcrit * sds / np.sqrt(10)
    sep = 0.0018
    say(f"  95% CI full widths (t(9)={tcrit:.3f}, n=10): {widths.min():.4f} to {widths.max():.4f}")
    say(f"  separation between first and last: {sep}")
    for n, s, wd in zip(names, sds, widths):
        say(f"    {n:15s} SD {s:.4f}  CI width {wd:.4f}  "
            f"{'separation is LARGER' if wd < sep else 'separation is smaller'}")
    say(f"  -> the published sentence is false for {int((widths < sep).sum())} of 9 entries; "
        "Reviewer 1's arithmetic is confirmed.")

    (a.out_dir / "perclass_and_dispersion_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {a.out_dir/'perclass_and_dispersion_report.txt'}")


if __name__ == "__main__":
    main()
