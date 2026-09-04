# Item 8 — depositing the code and data

Reviewer 1 asked for the code to be available **during review**, not after acceptance. That
request is the whole point: the reviewer has already found one table that did not come from the
pipeline, and wants to be able to check the rest. So the deposit has to be live before the
revision is resubmitted, and the manuscript has to carry a working link.

## Before you push

1. **Apply the four Supplementary S7.3 corrections** (`SUPPLEMENTARY_S7.3_revised.md`).
2. **Fill in the placeholders in `CITATION.cff`** — ORCID, GitHub username, release date.
3. **Copy your regenerated outputs into `results/`.** Include the CSVs and the figures; the whole
   set is a few megabytes and its presence is what lets a reviewer check a number without running
   anything. Keep the directory names the scripts produce, so the README's run order maps onto
   what is there.
4. **Decide the `target_label` question** (item 7). The column in `thyroid_ann_7200.csv` still
   reads `hyperthyroid` / `subnormal` / `normal`. Keeping it is defensible and arguably better,
   because it makes the admission in Section 4.1 verifiable — but only if the README says so,
   which it now does. What you cannot have is a paper saying the naming was abandoned and a
   deposited file that carries it without explanation.
5. **Do not commit `thyroid_ann_7200.csv` itself.** `make_dataset.py` regenerates it from the raw
   files in about a second and records the checksums. A repository whose data chain starts from a
   regenerable file is stronger than one that ships the derived artefact.
6. **Check nothing personal is in the tree** — absolute paths, machine names, credentials.
   `verify_env.py` had a hard-coded `C:\Users\bilco\...` path; that one is fixed, but grep the
   whole tree once more before pushing:
   ```
   grep -rn "C:\\\\Users\|bilco\|Desktop" --include=*.py --include=*.md .
   ```

## GitHub

```bash
cd pkg
git init
git add .
git commit -m "Code and data for the ann-thyroid benchmark audit"
git branch -M main
git remote add origin https://github.com/USERNAME/ann-thyroid-audit.git
git push -u origin main
```

Make the repository **public** — a private repository with an invitation link does not satisfy a
reviewer's request for availability during review.

Suggested name: `ann-thyroid-audit`. Description: "Controlled audit of the UCI ann-thyroid
multiclass benchmark: label semantics, method comparison, and reproducibility."

## Zenodo

The order matters here, because you need the DOI **inside the manuscript** you are about to
resubmit.

1. Sign in to Zenodo with your GitHub account, open **Settings → GitHub**, and switch the
   repository toggle **on**. This must happen before the release; Zenodo only archives releases
   created after the toggle is enabled.
2. In Zenodo, choose **New upload → Reserve DOI** if you want the identifier before publishing.
   Alternatively, create the GitHub release first and take the DOI Zenodo assigns.
3. On GitHub: **Releases → Create a new release**, tag `v1.0.0`, title
   "Initial release for peer review", publish.
4. Zenodo archives the release automatically and mints a DOI within a few minutes. Take the
   **concept DOI** (the one that always resolves to the latest version), not the version-specific
   one, for the manuscript.
5. In the Zenodo record, check that the author name, ORCID, affiliation and licence came through
   from `CITATION.cff`, and add the article title in **Related identifiers** once you have a
   preprint or article DOI.

If you revise the code after review, publish a new GitHub release; Zenodo will version the
record automatically and the concept DOI keeps resolving.

## Data availability statement for the manuscript

> The ann-thyroid benchmark and the sibling files used to establish its class semantics
> (`allhypo.data`, `allhypo.test`, `thyroid0387.data`, `thyroid.theory`) are publicly available
> from the UCI Machine Learning Repository, dataset 102, at
> https://archive.ics.uci.edu/dataset/102/thyroid+disease. All analysis code, the raw data files
> as used, and the complete generated results are archived at Zenodo
> (https://doi.org/10.5281/zenodo.XXXXXXX) and developed at
> https://github.com/USERNAME/ann-thyroid-audit. The working dataset is not distributed directly;
> `make_dataset.py` regenerates it from the raw files and records their SHA-256 digests, so that
> the analysis chain begins at the archive rather than at a derived artefact.

## Code availability statement

> All code is available under the MIT licence at https://github.com/USERNAME/ann-thyroid-audit
> and archived at https://doi.org/10.5281/zenodo.XXXXXXX. The repository includes the pinned
> environment specification, a script that verifies the environment against the versions used
> here, and scripts that regenerate every table and figure reported in the article. Table 1 and
> the class-semantics analysis are produced by scripts that assert their own results against
> independently recomputed values and exit non-zero on any mismatch.

## Draft response to Reviewer 1 on availability

> We agree that deferring release to acceptance was the wrong decision and have reversed it. The
> complete code, the raw UCI files, and all generated results are now public at
> [URL] and archived with a DOI at [DOI]. The repository includes the pinned environment, a
> `verify_env.py` that checks an installation against the versions used here, and a README giving
> the run order for every script. Two points may be of particular interest to the reviewer. The
> data chain now begins at the raw archive files rather than at a derived CSV:
> `make_dataset.py` regenerates the working dataset and records the SHA-256 digest of each input.
> And the scripts producing Table 1 and the class-semantics analysis verify their own output
> against independently recomputed values, so a mismatch is a failure rather than something a
> reader must notice. We also regenerated the complete figure set from a clean installation on a
> second machine; the recomputed values file matched the submitted one exactly.

## One thing worth mentioning in the letter

While preparing the deposit we ran the full pipeline from a clean checkout on a second machine
and compared the output against the submitted results. `figure_values.csv` matched in all 71
entries. That is a stronger statement than "the code is available", and it is exactly the
assurance a reviewer who has just found an unreproducible table will want. It also cost nothing
to obtain, since the check was run anyway.
