# Changes to the code package for the revision

Every edit is marked in the source with a `# REV:` comment.

## Modified scripts

| file | line | change | reviewer item |
|---|---|---|---|
| `9.thyroid_shap_perclass_ranking.py` | 256 | `scoring="accuracy"` → `scoring="f1_macro"` | R2 c.4 |
| `fig4_5_shap_permutation_UPDATED.py` | 335 | `scoring='accuracy'` → `scoring='f1_macro'` | R2 c.4 |
| `fig4_5_shap_permutation_UPDATED.py` | 140 | `AdaBoostClassifier(...)` → adds `algorithm="SAMME"` | see below |
| `fig7_roc_curves.py` | 129 | `AdaBoostClassifier(...)` → adds `algorithm="SAMME"` | see below |
| `verify_env.py` | 29 | hard-coded Windows path → `os.path.dirname(os.path.abspath(__file__))` | R1 (deposit) |

**The AdaBoost change is not cosmetic.** Supplementary S7.3 states that all released scripts set
`algorithm='SAMME'` explicitly. Two did not. On scikit-learn 1.3.2 they inherited SAMME.R; on 1.6
or later they would have silently switched to SAMME, which by the supplementary's own measurement
moves AdaBoost's ROC-AUC from 0.9965 to 0.9905. Figure 5 and Figure 7 were therefore
version-dependent and the supplementary's claim was false as written. Both are now pinned.

## New scripts

| file | purpose | reviewer item |
|---|---|---|
| `make_dataset.py` | regenerates `thyroid_ann_7200.csv` from the raw UCI files with SHA-256 provenance | deposit integrity |
| `rebuild_table1.py` | rebuilds Table 1 from the raw files; asserts every value the reviewer recomputed | R1 major 1 |
| `verify_class_semantics.py` | record-level match against `allhypo` and `thyroid0387`; reads the thresholds out of `thyroid.theory` | R1 major 2, R2 c.5 |
| `fs_statistical_tests.py` | Friedman, Kendall's W, Nemenyi, Holm-corrected Wilcoxon, TOST equivalence | R1 major 3, R2 c.3 |
| `perclass_and_dispersion.py` | class-wise sensitivity/specificity/PR-AUC, Table 8 dispersion, balancing-strategy omnibus tests | R2 c.6, R1 major 3–4 |
| `fig5_permutation_panel.py` | assembles Figure 5 from `permutation_importance.csv`; the composite was previously produced by hand outside the pipeline | deposit integrity |
| `hfs_discrete_estimation.py` | runs the selection stage under both information-gain estimators on the same folds and seeds; produces Table 9b | R1 third round |

## Fixed after first test run

`rebuild_table1.py` used `DataFrame.to_markdown()`, which needs the optional `tabulate`
package and raised `ImportError` on a clean install of the pinned environment. Replaced with a
small local writer so the repository has no dependency outside the pinned list.

`verify_class_semantics.py` merged the ann and raw tables on join keys whose dtypes differed
(float on one side, int-like on the other), raising a `UserWarning`. Both sides are now cast
explicitly. The matched counts are unchanged (3,597 one-to-one, no discordance), verified on
two independent machines.

## Not changed, and why

Scripts 1–8, 10 and 11 are untouched, so their outputs are unchanged and do not need to be
regenerated for the revision. Rerun them only if you want the deposit to be reproduced end to end
from a clean checkout.

The per-class column names in `ablation_all_runs.csv` (`sens_hyperthyroid`, `prauc_subnormal`,
`spec_normal`) still carry the original class names. The code is correct; only the labels are
misleading. They were left alone rather than renamed, because renaming them silently would break
any external reader of the published CSVs. `perclass_and_dispersion.py` maps them to the
laboratory-phenotype designations on read, and the README documents the mapping.


---

# Third revision (v1.0.2)

No released script was modified. The release adds one script, the per-split outputs the paper
refers to, and one encoding repair.

## Added

| file | why |
|---|---|
| `code/hfs_discrete_estimation.py` | R1 asked whether the findings about the selection scheme depend on the information-gain estimator. They do. The script runs the stage both ways on identical folds and seeds. |
| `results/discrete_estimation_results/` | its four per-split outputs; Table 9b is read off them |
| `results/thyroid_sensitivity_results/` | Table 9 and Supplementary S5 |
| `results/thyroid_clinical_rule_results/` | Table 10, including the per-split `elapsed_s` behind the run-time ratio |
| `results/thyroid_calibration_results/` | Table 11 |
| `results/thyroid_shap_stability_results/` | Table 12, upper panel |
| `results/thyroid_smote_realism_results/` | Supplementary S3 |
| `results/thyroid_dca_results/` | Supplementary S4 |

R1 reported that the Data Availability Statement and Supplementary S7.4 promised "the per-split
output files from which every reported value is computed", and that six result directories were
absent from v1.0.1. All six are now present.

## Repaired

`results/thyroid_calibration_results/calibration_report.txt` had been written as UTF-8, then
read back as cp1252 and encoded as UTF-8 a second time, so its Turkish characters were mojibake
(`KALİBRASYON` appeared as `KALÄ°BRASYON`). The text is restored. No number was affected.

## A number corrected in the paper, not in the code

The run-time ratio in Table 10 and Sections 3.8 and 4.4 is 197, not the 193 previously printed.
193 came from dividing the rounded table cells (40.47 / 0.21). Computed from the per-split times
now deposited, the means are 40.472 s for random forest and 0.205 s for the tuned rule, giving
197.2 — and 197.5 as the mean of the per-split ratios. Table 10 now prints 0.205 s and 0.011 s so
that the ratio is reproducible from the table itself. The logistic-regression comparison
(2.960 / 0.205 = 14.4, "one-fourteenth") is unchanged.

## Not changed, and why

Scripts 1–11 and the figure scripts are untouched in this round. This is deliberate: the
discrete-estimation comparison is only like for like if the released scripts still reproduce the
deposited per-split values exactly, so the estimator change lives in a separate script rather
than as an edit to `1.thyroid_ablation_ann7200_SAMME.py`.
