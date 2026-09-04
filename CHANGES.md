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
