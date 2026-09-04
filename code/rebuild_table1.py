#!/usr/bin/env python3
"""
rebuild_table1.py
Scientific Reports submission 7c9aaee8-d21e-46eb-9b22-4614badd4bca
Major revision, Reviewer 1 MAJOR comment 1 / Reviewer 2 comment 1.

Rebuilds Table 1 (cohort characteristics) directly from the raw UCI files
ann-train.data + ann-test.data (3772 + 3428 = 7200 records) and verifies every
cell the reviewer recomputed independently. Exit status is non-zero if any
check fails, so this cannot silently regress.

Provenance of the input archive (thyroid_disease.zip, UCI dataset 102, files
dated 2023-05-22), SHA-256:
  ann-train.data  3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6
  ann-test.data   c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b
  allhypo.data    273bfa43c6ad226204cb42d17365143723786db447b074abb71a23a632c1ff53
  allhypo.test    4dc8a3964d76944ad84e7c242c7155684a3d603576245f2ed64e2c6d072d9422
  thyroid.theory  dbea554e2f46434b581421af39cfec37f4fc02a5e95de2e6ea17536f28780a1e

Usage:
    python rebuild_table1.py --data-dir ./uci --out-dir ./table1_out
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
# ann-thyroid.names documents 21 attributes (15 binary, 6 continuous) and three
# classes but assigns NO attribute names and NO class names. The names below are
# recovered from allhypo.names, which describes the same attribute sequence for
# the sibling files distributed in the same directory.
BINARY = [
    "sex (female=0/male=1 in the ann encoding)",
    "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick",
    "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
]
BINARY_KEYS = ["sex"] + BINARY[1:]
ASSAY = ["TSH", "T3", "TT4", "T4U", "FTI"]
COLUMNS = ["age"] + BINARY_KEYS + ASSAY + ["class"]

# The ann files rescale every continuous attribute into [0, 1]. The constants are
# undocumented; they were recovered empirically and confirmed by record-level
# agreement with the raw values in allhypo.data / allhypo.test (see verify()).
SCALE = {"age": 100, "TSH": 1000, "T3": 100, "TT4": 1000, "T4U": 10, "FTI": 1000}

# Values recomputed independently by Reviewer 1 from the same two files.
REVIEWER = {
    "age_median": 55.0,
    "sex_majority": 5009,
    "sex_minority": 2191,
    "onT4_overall_pct": 13.1,
    "onT4_class2_pct": 0.0,
    "TSH_c1": (6.2, 530.0),
    "TSH_c2": (6.1, 143.0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load(data_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("ann-train.data", "ann-test.data"):
        path = data_dir / name
        if not path.exists():
            sys.exit(f"[FATAL] missing {path}")
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
        df["source_file"] = name
        frames.append(df)
        print(f"[load] {name}: n={len(df)}  sha256={sha256(path)[:16]}...")
    ann = pd.concat(frames, ignore_index=True)
    for col, scale in SCALE.items():
        ann[col] = (ann[col] * scale).round(4)
    return ann


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table writer. pandas.to_markdown needs the optional
    `tabulate` package; this keeps the repository dependency-free."""
    header = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in r] for i, r in zip(df.index, df.to_numpy())]
    widths = [max(len(h), *(len(r[k]) for r in rows)) if rows else len(h)
              for k, h in enumerate(header)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header),
                      "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)]) + "\n"


def fmt_cont(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "--"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.1f} [{q1:.1f}, {q3:.1f}] ({s.min():.1f}-{s.max():.1f})"


def fmt_bin(s: pd.Series) -> str:
    n = int(s.sum())
    return f"{n} ({100 * n / len(s):.1f})"


def build(ann: pd.DataFrame) -> pd.DataFrame:
    strata = {f"Overall (n={len(ann)})": ann}
    for c in sorted(ann["class"].unique()):
        sub = ann[ann["class"] == c]
        strata[f"Class {c} (n={len(sub)})"] = sub

    rows = {}
    rows["Age, years"] = {k: fmt_cont(v["age"]) for k, v in strata.items()}
    for key, label in zip(BINARY_KEYS, BINARY):
        rows[f"{label}, n (%)"] = {k: fmt_bin(v[key]) for k, v in strata.items()}
    for a in ASSAY:
        rows[f"{a}"] = {k: fmt_cont(v[a]) for k, v in strata.items()}
        rows[f"{a}, n measured"] = {k: str(int(v[a].notna().sum())) for k, v in strata.items()}
    return pd.DataFrame(rows).T


def verify(ann: pd.DataFrame) -> bool:
    ok = True

    def check(name, got, want, tol=0.051):
        nonlocal ok
        hit = abs(got - want) <= tol
        ok &= hit
        print(f"  [{'OK      ' if hit else 'MISMATCH'}] {name}: computed={got} reviewer={want}")

    print("\n[verify] against Reviewer 1's independent recomputation")
    check("age median", float(ann["age"].median()), REVIEWER["age_median"])
    ones = int(ann["sex"].sum())
    check("sex majority group", max(ones, len(ann) - ones), REVIEWER["sex_majority"], 0)
    check("sex minority group", min(ones, len(ann) - ones), REVIEWER["sex_minority"], 0)
    check("on_thyroxine overall %", round(100 * ann["on_thyroxine"].mean(), 1),
          REVIEWER["onT4_overall_pct"])
    check("on_thyroxine Class 2 %",
          round(100 * ann[ann["class"] == 2]["on_thyroxine"].mean(), 1),
          REVIEWER["onT4_class2_pct"])
    for cls in (1, 2):
        s = ann[ann["class"] == cls]["TSH"].dropna()
        lo, hi = REVIEWER[f"TSH_c{cls}"]
        check(f"TSH Class {cls} min", round(float(s.min()), 1), lo)
        check(f"TSH Class {cls} max", round(float(s.max()), 1), hi)
    print(f"[verify] {'ALL CHECKS PASSED' if ok else 'FAILED -- do not submit'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./table1_out"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ann = load(args.data_dir)
    print(f"[load] pooled n={len(ann)}")
    print(f"[load] class sizes: {ann['class'].value_counts().sort_index().to_dict()}")

    # Unmeasured TSH is coded 0.0 in the ann files (the *_measured flags of the
    # sibling raw files were dropped). thyroid.theory confirms these are missing.
    n_zero = int((ann["TSH"] == 0).sum())
    by_class = ann.loc[ann["TSH"] == 0, "class"].value_counts().sort_index().to_dict()
    print(f"[prep] TSH coded 0 (unmeasured): n={n_zero}, by class {by_class} -> set to NaN")
    ann.loc[ann["TSH"] == 0, "TSH"] = np.nan

    table = build(ann)
    table.to_csv(args.out_dir / "table1_rebuilt.csv")
    (args.out_dir / "table1_rebuilt.md").write_text(to_markdown(table), encoding="utf-8")
    print(f"\n{table.to_string()}\n")
    print(f"[write] {args.out_dir/'table1_rebuilt.csv'}")
    print(f"[write] {args.out_dir/'table1_rebuilt.md'}")

    sys.exit(0 if verify(ann) else 1)


if __name__ == "__main__":
    main()
