"""
07_check_reproduction.py
Compares freshly regenerated results/ with the reference tables shipped in reference/
(the files the paper's numbers come from).

  python 07_check_reproduction.py --results results --reference reference

Valid only when 03 was run on the released data/preds.npz (retraining the forecaster on a GPU
can change preds.npz slightly).

  summary.csv, vif_het_alpha0_1.csv : numeric match within --tol (default 1e-3 relative / 5e-3 absolute
                                      for the 2-decimal vif_het file)
  table_main_ci*.csv                : the point estimates must match to the printed precision; bootstrap
                                      endpoints depend on the resampling seed/implementation, so the largest
                                      endpoint difference is only REPORTED (not a failure).
Exit code 1 if any point/summary check fails.
"""
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd

NUM = re.compile(r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]")


def parse(cell):
    m = NUM.match(str(cell))
    return tuple(float(x) for x in m.groups())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--reference", default="reference")
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    ok = True

    def load(d, f):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print(f"[MISSING] {p}")
            return None
        return pd.read_csv(p)

    # summary.csv
    new, ref = load(a.results, "summary.csv"), load(a.reference, "summary.csv")
    if new is not None and ref is not None:
        key = ["method", "alpha", "lead_h"]
        m = ref.merge(new, on=key, suffixes=("_ref", "_new"))
        bad = 0
        for c in [c for c in ref.columns if c not in key]:
            diff = (m[c + "_ref"] - m[c + "_new"]).abs() / (m[c + "_ref"].abs() + 1e-9)
            bad += int((diff > a.tol).sum())
        print(f"summary.csv: {len(m)}/{len(ref)} rows matched, {bad} cells beyond tol {a.tol}")
        ok &= bad == 0 and len(m) == len(ref)

    # vif_het (reference has 2 decimals)
    new, ref = load(a.results, "vif_het_alpha0_1.csv"), load(a.reference, "vif_het_alpha0_1.csv")
    if new is not None and ref is not None:
        m = ref.merge(new, on=["method", "lead_h"], suffixes=("_ref", "_new"))
        mx = (m.vif_het_ref - m.vif_het_new).abs().max()
        print(f"vif_het_alpha0_1.csv: max |diff| = {mx:.4f} (reference is rounded to 0.01)")
        ok &= mx <= 0.0051 and len(m) == len(ref)

    # CI tables
    for f in ["table_main_ci.csv", "table_main_ci_alpha0_2.csv", "table_main_ci_alpha0_05.csv"]:
        new, ref = load(a.results, f), load(a.reference, f)
        if new is None or ref is None:
            continue
        m = ref.merge(new, on=["method", "lead_h"], suffixes=("_ref", "_new"))
        pt_ok, pt, ep = True, 0.0, 0.0
        for c, dec in [("coverage", 3), ("score", 2), ("vif", 2)]:
            for _, r in m.iterrows():
                r0, r1 = parse(r[c + "_ref"]), parse(r[c + "_new"])
                pt = max(pt, abs(r0[0] - r1[0]))
                pt_ok &= abs(r0[0] - r1[0]) < 1.01 * 10 ** -dec      # one unit of the last printed digit
                ep = max(ep, abs(r0[1] - r1[1]), abs(r0[2] - r1[2]))
        print(f"{f}: {len(m)}/{len(ref)} rows; max point diff {pt:.4f}, max endpoint diff {ep:.4f} (info)")
        ok &= pt_ok and len(m) == len(ref)

    print("RESULT:", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
