# Supplementary S7.3 — corrected text

Four defects were found in the current S7.3 during the revision. All four are in the same
subsection and should be fixed together. Each is stated below with what is wrong and why it
matters, followed by replacement text you can paste.

---

## Defect 1 — the SAMME claim is false for two scripts

**Current text:** "All scripts released with this paper set `algorithm='SAMME'` explicitly, so
that they reproduce identically on either side of the 1.6 boundary."

`AdaBoostClassifier` was instantiated without the `algorithm` argument in
`fig4_5_shap_permutation_UPDATED.py` and `fig7_roc_curves.py`. Both therefore inherited SAMME.R
under scikit-learn 1.3.2 and would have switched silently to SAMME on 1.6 or later — by the
supplementary's own measurement, a change from 0.9965 to 0.9905 in AdaBoost's ROC-AUC. Figure 5
and Figure 7 were consequently version-dependent. Both are now pinned and both figures have been
regenerated. `11.make_figures_1_6_7_8.py` had always set it correctly.

This is checkable by anyone who opens the deposited code, so state it rather than quietly fixing
it.

## Defect 2 — the recorded threadpoolctl version does not run

`threadpoolctl 2.2.0` is recorded. That version enumerates every DLL loaded into the process and
reads a version string from each; on some Windows installations one of them returns nothing and
the library raises `AttributeError`. Any script whose path reaches scikit-learn's neighbour search
— that is, anything involving KNN — then fails. We reproduced this on a clean installation built
from the recorded list, using both the revised and the original unmodified scripts, so it is a
property of the environment and not of the code.

A reviewer following S7.3 exactly would hit the same crash. The pin is now `>= 3.1.0`.

## Defect 3 — the convergence behaviour of the linear SVM is undocumented

`SVC(kernel="linear")` is instantiated with `max_iter=200000` in every script that uses it, and
emits `ConvergenceWarning: Solver terminated early` on the SMOTE-resampled training folds. The
limit was set deliberately — the default of −1 does not terminate in reasonable time on this data
— but the consequence is that the linear SVM's fit depends on the iteration budget, and this is
nowhere stated.

It is better to own this than to have it noticed. It also sits comfortably with the paper's
argument: a classifier whose solution depends on its optimisation budget is a further reason not
to read small differences between classifiers as findings about the methods.

## Defect 4 — corrupted sentence

The environment paragraph currently ends: "...may reflect SAMME.R behaviour.versions, together
with the operating system and CPU model." A fragment of an earlier sentence is spliced in.

---

## Replacement text for S7.3

> **Computational environment.** All analyses were run under Python 3.11.7 with scikit-learn
> 1.3.2, xgboost 2.0.3, imbalanced-learn 0.11.0, shap 0.49.1, numpy 1.26.4, pandas 2.1.4, scipy
> 1.11.4 and matplotlib 3.8.0.
>
> The scikit-learn pin is load-bearing. Version 1.6 removed the `SAMME.R` boosting algorithm and
> made `SAMME` the default for `AdaBoostClassifier`; on this dataset the two differ by
> approximately 0.006 ROC-AUC, so an unpinned script would produce different AdaBoost results
> either side of that release. Every script released with this work now sets `algorithm="SAMME"`
> explicitly. Two of them did not do so at first submission — the scripts generating Figure 5 and
> Figure 7 — and both figures have been regenerated with the algorithm pinned.
>
> The original analyses were run with threadpoolctl 2.2.0. We subsequently found that this
> version fails on some Windows installations when scikit-learn's neighbour search requests a
> thread limit, raising an `AttributeError` while enumerating loaded modules, which makes every
> script involving the k-nearest-neighbour classifier unrunnable. The library manages thread
> counts and takes no part in any computation, so results are unaffected; the deposited
> environment specification therefore requires threadpoolctl 3.1.0 or later.
>
> One further behaviour is expected rather than anomalous. `SVC(kernel="linear")` is instantiated
> with `max_iter=200000`, because the scikit-learn default of unbounded iteration does not
> terminate in reasonable time on the SMOTE-resampled training folds, and it emits a
> `ConvergenceWarning` indicating early termination. The reported results for the linear support
> vector machine were obtained under this budget.

If the surrounding text needs a sentence on hardware, append: "Runs were performed on an Intel
Core i5-3210M with 8 GB of memory under Windows 10."
