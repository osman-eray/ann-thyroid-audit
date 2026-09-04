#!/usr/bin/env python3
"""
make_dataset.py — provenance for thyroid_ann_7200.csv

Regenerates the working dataset from the raw UCI files so that the analysis chain
starts from a documented step rather than from an undocumented artefact:

    ann-train.data + ann-test.data  ->  thyroid_ann_7200.csv  ->  all analysis scripts

Records the SHA-256 of the inputs, applies the attribute names recovered from
allhypo.names (ann-thyroid.names supplies none), and verifies that the result is
identical in content to the file used for every experiment in the manuscript.

The class label column is written twice. `target` (1/2/3) is what every analysis
script reads. `target_label` carries the ORIGINAL distribution's class names, which
this paper shows to be incorrect: Class 1 and Class 2 are hypothyroid strata, not
hyperfunction. It is retained deliberately, so that the statement in Section 4.1
about the earlier designation can be checked against the file, and it is used by
no script. See README, "A note on the class names".

Usage:
    python make_dataset.py --raw-dir ../data --out thyroid_ann_7200.csv
    python make_dataset.py --raw-dir ../data --out thyroid_ann_7200.csv --compare existing.csv
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BINARY = [
    "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
]
ASSAY = ["TSH", "T3", "TT4", "T4U", "FTI"]
COLUMNS = ["age"] + BINARY + ASSAY + ["target"]

# Class names as distributed with the original benchmark. Shown to be incorrect in
# Section 4.1; retained for verifiability only. Nothing reads this mapping.
ORIGINAL_LABELS = {1: "hyperthyroid", 2: "subnormal", 3: "normal"}

EXPECTED_SHA256 = {
    "ann-train.data": "3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6",
    "ann-test.data":  "c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b",
}
EXPECTED_ROWS = {"ann-train.data": 3772, "ann-test.data": 3428}
EXPECTED_CLASS_SIZES = {1: 166, 2: 368, 3: 6666}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="directory holding ann-train.data and ann-test.data")
    ap.add_argument("--out", type=Path, default=Path("thyroid_ann_7200.csv"))
    ap.add_argument("--compare", type=Path, default=None,
                    help="optional: an existing CSV to check the result against")
    a = ap.parse_args()

    frames = []
    for name in ("ann-train.data", "ann-test.data"):
        path = a.raw_dir / name
        if not path.exists():
            sys.exit(f"[FATAL] missing {path}\n"
                     "        Download from "
                     "archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/")
        digest = sha256(path)
        status = "matches" if digest == EXPECTED_SHA256[name] else "DIFFERS FROM EXPECTED"
        print(f"[input] {name}  sha256={digest[:16]}...  ({status})")
        if status != "matches":
            print(f"        expected {EXPECTED_SHA256[name][:16]}... — the archive copy may have "
                  "changed; do not proceed silently.")
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
        if len(df) != EXPECTED_ROWS[name]:
            sys.exit(f"[FATAL] {name}: expected {EXPECTED_ROWS[name]} rows, found {len(df)}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    sizes = out["target"].value_counts().sort_index().to_dict()
    print(f"[build] pooled n={len(out)}  class sizes {sizes}")
    if sizes != EXPECTED_CLASS_SIZES:
        sys.exit(f"[FATAL] class sizes {sizes} != expected {EXPECTED_CLASS_SIZES}")

    out["target_label"] = out["target"].map(ORIGINAL_LABELS)

    if a.compare is not None:
        if not a.compare.exists():
            sys.exit(f"[FATAL] --compare file not found: {a.compare}")
        ref = pd.read_csv(a.compare)
        num = ["age"] + BINARY + ASSAY
        same = (len(ref) == len(out)
                and np.allclose(out[num].to_numpy(), ref[num].to_numpy())
                and (out["target"].to_numpy() == ref["target"].to_numpy()).all())
        print(f"[check] identical in content to {a.compare.name}: {same}")
        if not same:
            sys.exit("[FATAL] regenerated file differs from the reference — investigate before use")

    out.to_csv(a.out, index=False)
    print(f"[write] {a.out}  sha256={sha256(a.out)[:16]}...")
    print("\nThe continuous attributes remain on the [0,1] scale of the distributed files.\n"
          "Analysis scripts consume them as-is; the de-normalisation factors (age x100,\n"
          "TSH x1000, T3 x100, TT4 x1000, T4U x10, FTI x1000) are applied only where the\n"
          "manuscript reports values in clinical units. See rebuild_table1.py.")


if __name__ == "__main__":
    main()
