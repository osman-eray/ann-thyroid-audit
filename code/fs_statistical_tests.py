#!/usr/bin/env python3
"""
fs_statistical_tests.py
Scientific Reports 7c9aaee8 — Reviewer 1 MAJOR 3, Reviewer 2 comment 3.

Supplies the overall statistical test the null claim currently lacks. Operates on the
per-split values already produced by 6.thyroid_fs_comparison_ann7200.py; no model is
refitted and no new experiment is run.

  1. Friedman test across the nine feature-selection methods (blocks = the ten splits)
  2. Kendall's W as the effect size
  3. Post-hoc Nemenyi with the critical difference (rank-based; the standard companion
     to Friedman, and not subject to the resolution floor described below)
  4. Post-hoc pairwise Wilcoxon signed-rank, Holm-corrected over 36 pairs
  5. Two one-sided tests (TOST) against a pre-declared equivalence margin

RESOLUTION FLOOR. With ten splits the smallest attainable two-sided exact Wilcoxon
p-value is 2/2^10 = 0.00195. Holm-corrected over 36 pairs that becomes 0.0703, so NO
pair can reach 0.05 however large its effect. The Holm-corrected Wilcoxon post-hoc is
therefore uninformative by construction here and must not be reported as evidence of
absence. The script prints the floor alongside the results; the Nemenyi test and the
equivalence tests carry the inference.

Usage: python fs_statistical_tests.py --runs fs_comparison_all_runs.csv --out-dir ./stats_out
"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

# metric -> (higher is better, pre-declared equivalence margin)
METRICS = {"roc_auc_ovr": (True, 0.005), "f1_macro": (True, 0.02), "specificity_macro": (True, 0.01)}

# Studentised range statistic / sqrt(2) at alpha = 0.05, indexed by number of treatments
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def kendalls_w(ranks: np.ndarray) -> float:
    n, k = ranks.shape
    R = ranks.sum(axis=0)
    return 12 * ((R - R.mean()) ** 2).sum() / (n ** 2 * (k ** 3 - k))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.clip(np.maximum.accumulate((m - np.arange(m)) * p[order]), 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out


def tost_p(diff: np.ndarray, margin: float) -> float:
    """Paired equivalence: both one-sided Wilcoxon nulls must be rejected."""
    lo = stats.wilcoxon(diff + margin, alternative="greater").pvalue
    hi = stats.wilcoxon(diff - margin, alternative="less").pvalue
    return max(lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./stats_out"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.runs)
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    for metric, (higher_better, margin) in METRICS.items():
        wide = df.pivot(index="run_id", columns="method", values=metric)
        methods, mat = list(wide.columns), wide.to_numpy()
        n, k = mat.shape
        sign = -1.0 if higher_better else 1.0          # rank 1 = best
        ranks = np.apply_along_axis(stats.rankdata, 1, sign * mat)
        mean_rank = ranks.mean(axis=0)

        say("=" * 78)
        say(f"{metric}   ({n} splits x {k} methods; equivalence margin delta = {margin})")
        say("=" * 78)
        chi2, p = stats.friedmanchisquare(*mat.T)
        say(f"Friedman chi2({k-1}) = {chi2:.4f}   p = {p:.3e}   Kendall's W = {kendalls_w(ranks):.4f}")

        say("\nmean rank (1 = best)   mean value   SD")
        for i in np.argsort(mean_rank):
            say(f"  {methods[i]:16s} {mean_rank[i]:5.2f}   {mat[:, i].mean():.6f}   {mat[:, i].std(ddof=1):.6f}")
        spread = mat.mean(axis=0).max() - mat.mean(axis=0).min()
        say(f"\nspread between best and worst mean: {spread:.6f}   (margin {margin})")

        cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
        say(f"\nNemenyi critical difference (alpha 0.05): {cd:.3f} mean ranks")
        say(f"  observed mean-rank span: {mean_rank.max() - mean_rank.min():.3f}")

        floor = 2 / 2 ** n
        say(f"\nWilcoxon resolution floor: smallest exact two-sided p = {floor:.5f}; "
            f"Holm over {k*(k-1)//2} pairs = {floor*(k*(k-1)//2):.4f}")

        rows = []
        for i, j in itertools.combinations(range(k), 2):
            d = mat[:, i] - mat[:, j]
            rows.append({"metric": metric, "method_a": methods[i], "method_b": methods[j],
                         "mean_diff": d.mean(),
                         "rank_diff": abs(mean_rank[i] - mean_rank[j]),
                         "p_wilcoxon": 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue,
                         "p_tost": tost_p(d, margin)})
        res = pd.DataFrame(rows)
        res["p_holm"] = holm(res.p_wilcoxon.to_numpy())
        res["nemenyi_sig"] = res.rank_diff > cd
        res["equivalent"] = res.p_tost < 0.05
        res = res.sort_values("rank_diff", ascending=False).reset_index(drop=True)

        say(f"\npairs separated by Nemenyi          : {int(res.nemenyi_sig.sum())} / {len(res)}")
        say(f"pairs significant at Holm-adjusted 0.05: {int((res.p_holm < 0.05).sum())} / {len(res)}  "
            f"(unattainable here -- see resolution floor)")
        say(f"pairs equivalent within +/-{margin} (TOST): {int(res.equivalent.sum())} / {len(res)}")
        say(f"largest absolute mean difference        : {res.mean_diff.abs().max():.6f}")
        if res.nemenyi_sig.any():
            say("\n  pairs separated by Nemenyi:")
            for _, r in res[res.nemenyi_sig].iterrows():
                say(f"    {r.method_a:16s} vs {r.method_b:16s}  rank diff {r.rank_diff:.2f}  "
                    f"mean diff {r.mean_diff:+.6f}  {'equivalent' if r.equivalent else 'NOT equivalent'}")
        say()
        res.to_csv(a.out_dir / f"posthoc_{metric}.csv", index=False)

    (a.out_dir / "fs_statistical_tests_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {a.out_dir/'fs_statistical_tests_report.txt'}")


if __name__ == "__main__":
    main()
