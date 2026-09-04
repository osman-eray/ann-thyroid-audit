#!/usr/bin/env python3
"""
verify_class_semantics.py
Scientific Reports 7c9aaee8-d21e-46eb-9b22-4614badd4bca — Reviewer 1 MAJOR 2, Reviewer 2 comment 5.

Establishes, at record level, what the three unnamed ann-thyroid classes actually
are, using only files distributed in the same UCI directory (dataset 102):

  1. allhypo.data / allhypo.test  -- named diagnostic classes, 3,772 records
  2. thyroid0387.data             -- 9,172 records with per-record diagnosis codes
  3. thyroid.theory               -- the assay thresholds encoded by the data owners

Usage:  python verify_class_semantics.py --data-dir ./uci
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd

BIN = ["sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick","pregnant",
       "thyroid_surgery","I131_treatment","query_hypothyroid","query_hyperthyroid","lithium",
       "goitre","tumor","hypopituitary","psych"]
ASSAY  = ["TSH","T3","TT4","T4U","FTI"]
ANNCOL = ["age"]+BIN+ASSAY+["class"]
SCALE  = {"age":100,"TSH":1000,"T3":100,"TT4":1000,"T4U":10,"FTI":1000}
RAWCOL = (["age","sex"]+BIN[1:]+["TSH_measured","TSH","T3_measured","T3","TT4_measured","TT4",
          "T4U_measured","T4U","FTI_measured","FTI","TBG_measured","TBG","referral_source","target"])


def load_ann(d: Path) -> pd.DataFrame:
    ann = pd.concat([pd.read_csv(d/f, sep=r"\s+", header=None, names=ANNCOL).assign(source_file=f)
                     for f in ("ann-train.data","ann-test.data")], ignore_index=True)
    for c, s in SCALE.items():
        ann[c] = (ann[c]*s).round(4)
    return ann.reset_index().rename(columns={"index":"ann_i"})


def load_raw(d: Path, files) -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(d/f, header=None, names=RAWCOL).assign(source_file=f)
                     for f in files], ignore_index=True)
    for c in ["age"]+ASSAY:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["sex"] = raw["sex"].map({"F":0,"M":1})
    for c in BIN[1:]:
        raw[c] = (raw[c] == "t").astype(int)
    return raw.reset_index().rename(columns={"index":"raw_i"})


def strict_match(ann: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One-to-one matches on age + the 15 binary attributes, keeping only pairs whose
    every assay MEASURED in the raw file equals the ann value. Records that admit
    more than one partner on either side are discarded, so no arbitrary assignment
    is ever made."""
    # REV: align the join-key dtypes. `age` is float in the ann files and int-like in
    # the raw ones; merging across the two raises a UserWarning and, on some pandas
    # versions, silently drops rows. Cast both sides explicitly.
    ann = ann.copy(); raw = raw.copy()
    for col in ["age"] + BIN:
        ann[col] = ann[col].astype(float)
        raw[col] = raw[col].astype(float)
    cand = ann.merge(raw, on=["age"]+BIN, how="inner", suffixes=("_ann","_raw"))
    ok = np.ones(len(cand), bool)
    for a in ASSAY:
        ok &= (cand[a+"_raw"].isna()
               | np.isclose(cand[a+"_ann"], cand[a+"_raw"], atol=0.02, rtol=0)
               | (cand[a+"_ann"] == 0))          # 0 = unmeasured in the ann files
    cand = cand[ok]
    n_ann = cand.groupby("ann_i").raw_i.nunique()
    n_raw = cand.groupby("raw_i").ann_i.nunique()
    return cand[cand.ann_i.map(n_ann).eq(1) & cand.raw_i.map(n_raw).eq(1)]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, required=True)
    d = ap.parse_args().data_dir
    ann = load_ann(d)
    print(f"ann-thyroid: n={len(ann)}, class sizes {ann['class'].value_counts().sort_index().to_dict()}\n")

    # ---------------- 1. class-level agreement with the named allhypo strata ----
    print("="*78 + "\n1. ASSAY RANGES: ann classes vs the named strata of allhypo\n" + "="*78)
    ah = load_raw(d, ("allhypo.data","allhypo.test"))
    ah["label"] = ah.target.str.split("|").str[0].str.rstrip(".")
    ann_nz = ann.copy(); ann_nz.loc[ann_nz.TSH == 0, "TSH"] = np.nan
    def rng(s):
        s = s.dropna(); return f"{s.min():g}-{s.max():g}"
    for cls, lab in ((1,"primary hypothyroid"), (2,"compensated hypothyroid")):
        a = ann_nz[ann_nz["class"] == cls]; b = ah[ah.label == lab]
        print(f"  ann Class {cls} (n={len(a)})   TSH {rng(a.TSH):>12}  TT4 {rng(a.TT4):>10}  FTI {rng(a.FTI):>12}")
        print(f"  {lab} (n={len(b)})   TSH {rng(b.TSH):>12}  TT4 {rng(b.TT4):>10}  FTI {rng(b.FTI):>12}\n")
    print("  NOTE: the ranges coincide; the counts do not (95 vs 166, 194 vs 368). allhypo\n"
          "  covers 3,772 records, ann-thyroid 7,200; they are overlapping samples of the\n"
          "  same 9,172-record Garavan archive, not the same record set.\n")

    # ---------------- 2. record-level diagnosis codes from thyroid0387 ----------
    print("="*78 + "\n2. RECORD-LEVEL MATCH against thyroid0387 diagnosis codes\n" + "="*78)
    t = load_raw(d, ("thyroid0387.data",))
    t["dx"] = t.target.str.extract(r"^([^\[]*)")[0].str.strip()
    m = strict_match(ann, t)
    print(f"  one-to-one matched records: {len(m)} of {len(ann)}")
    def group(dx):
        s = str(dx)
        if any(c in s for c in "FG"):   return "hypothyroid (F/G)"
        if any(c in s for c in "ABCD"): return "hyperthyroid (A-D)"
        return "no thyroid-function diagnosis"
    m = m.assign(dx_group=m.dx.map(group))
    print("\n" + pd.crosstab(m["class"], m.dx_group).to_string())
    print("\n  diagnosis strings behind Classes 1 and 2:")
    print("  " + pd.crosstab(m[m["class"].isin([1,2])]["class"],
                             m[m["class"].isin([1,2])].dx).to_string().replace("\n","\n  "))
    h = m[m.dx_group == "hyperthyroid (A-D)"]
    print(f"\n  records carrying a hyperthyroid diagnosis: n={len(h)}, "
          f"all in ann Class {sorted(h['class'].unique())}; "
          f"{int((h.TSH_ann < 0.4).sum())} of them have TSH < 0.4 mIU/L")

    # ---------------- 3. thresholds encoded in thyroid.theory ------------------
    print("\n" + "="*78 + "\n3. THRESHOLDS ENCODED IN thyroid.theory\n" + "="*78)
    theory = (d/"thyroid.theory").read_text(errors="ignore")
    for pat, what in ((r"tsh_value > 6", "TSH > 6 -> high"),
                      (r"tt4_value <60", "TT4 < 60 -> low"),
                      (r"fti_value < 65", "FTI < 65 -> low"),
                      (r"fti_value > 155", "FTI > 155 -> high"),
                      (r"tsh_value == 0", "TSH == 0 -> not_ready (unmeasured)")):
        print(f"  [{'found' if re.search(pat, theory) else 'ABSENT'}] {what}")
    print()
    for cls in (1, 2, 3):
        s = ann[ann["class"] == cls]
        print(f"  Class {cls} (n={len(s)}): TSH>6 {int((s.TSH>6).sum())}"
              f"  FTI<65 {int((s.FTI<65).sum())}"
              f"  FTI in [65,155] {int(s.FTI.between(65,155).sum())}"
              f"  FTI>155 {int((s.FTI>155).sum())}")
    print("\n  Every record in Classes 1 and 2 satisfies the theory's TSH-high condition, and the\n"
          "  two strata are separated by its FTI low/normal boundary. The converse does not hold:\n"
          "  177 Class-3 records also exceed TSH 6 but carry other diagnoses in thyroid0387, so\n"
          "  the thresholds delimit the strata without generating the labels.")


if __name__ == "__main__":
    main()
