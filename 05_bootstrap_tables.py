"""
05_bootstrap_tables.py
Block-bootstrap tables of the paper, computed from the per-day arrays stored by 03_run_benchmark.py
(results/alpha_*/<method>.npz: spatial_cov, score_day) -- no forecasts or ERA5 needed.

  python 05_bootstrap_tables.py --results results --alphas 0.1 0.2 0.05

Protocol (paper, Sec. "Uncertainty quantification"): 95% moving-block bootstrap percentile
intervals over test initialisations; blocks of 30 days, 1,000 replicates; the calibration set and
the forecaster are held fixed. Paired intervals use the SAME resampled days for both methods.
Resamples are overlapping (not circular) blocks; the stack is truncated to the number of test days.
The VIF bootstrapped here is the HOMOGENEOUS version (as in the paper's tables): for each replicate,
var(c, ddof=1) / (p(1-p) * sum_h w_h^2 / W) with p = 1 - mean(c) of that replicate.

Outputs (results/):
  table_main_ci.csv, table_main_ci_alpha0_2.csv, table_main_ci_alpha0_05.csv   (leads 24, 120, 240 h)
  table_paired_alpha<a>.csv        differences vs split_local (score, VIF_hom, coverage)
  block_sensitivity.csv            paired differences for blocks of 10/30/60 days (alpha 0.1)
"""
import argparse
import os
import numpy as np
import pandas as pd
from conformal_lib import METHOD_NAMES

LEADS_TABLE = [24, 120, 240]
REFERENCE = "split_local"


def tag(alpha):
    return str(alpha).replace(".", "_")


def table_name(alpha):
    return "table_main_ci.csv" if np.isclose(alpha, 0.1) else f"table_main_ci_alpha{tag(alpha)}.csv"


def block_indices(n, block, B, seed):
    """(B, n) indices of a moving-block bootstrap (overlapping blocks, truncated to n)."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, k))
    return (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]


def metrics(c, s, idx, S):
    """c, s: (N, L) per-day spatial coverage and interval score. idx: (B, n). Returns dict of (B, L)."""
    cc = c[idx].astype(np.float64)                      # (B, n, L)
    cov = cc.mean(1)
    p = 1 - cov
    return {"coverage": cov,
            "score": s[idx].astype(np.float64).mean(1),
            "vif": cc.var(axis=1, ddof=1) / (p * (1 - p) * S)}


def point(c, s, S):
    return {k: v[0] for k, v in metrics(c, s, np.arange(len(c))[None], S).items()}


def load(results, alpha, method):
    r = np.load(os.path.join(results, f"alpha_{alpha}", f"{method}.npz"))
    return r["spatial_cov"].astype(np.float64), r["score_day"].astype(np.float64)


def fmt(x, lo, hi, d):
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.05])
    ap.add_argument("--methods", nargs="+", default=METHOD_NAMES)
    ap.add_argument("--B", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--block", type=int, default=30, help="block length in days")
    ap.add_argument("--block_lens", type=int, nargs="*", default=[10, 30, 60],
                    help="block lengths for the paired sensitivity check (alpha 0.1); empty = skip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta = np.load(os.path.join(a.results, "meta.npz"))
    leads_h, lat, W = [int(x) for x in meta["leads_h"]], meta["lat"], len(meta["lon"])
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()
    S = (w ** 2).sum() / W                                  # sum_i w_i^2 in the paper's notation
    summ = pd.read_csv(os.path.join(a.results, "summary.csv"))
    li = [leads_h.index(h) for h in LEADS_TABLE]

    def paired_rows(alpha, block, methods, D):
        n = D[REFERENCE][0].shape[0]
        idx = block_indices(n, block, a.B, a.seed)
        ref = metrics(*D[REFERENCE], idx, S)
        ref0 = point(*D[REFERENCE], S)
        rows = []
        for m in methods:
            if m == REFERENCE:
                continue
            bm, b0 = metrics(*D[m], idx, S), point(*D[m], S)
            for l in li:
                row = dict(alpha=alpha, block=block, method=m, lead_h=leads_h[l])
                for key, name in [("score", "d_score"), ("vif", "d_vif_hom"), ("coverage", "d_coverage")]:
                    d = bm[key][:, l] - ref[key][:, l]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    row.update({name: b0[key][l] - ref0[key][l], name + "_lo": lo, name + "_hi": hi,
                                name + "_excludes0": bool(lo > 0 or hi < 0)})
                rows.append(row)
        return rows

    for alpha in a.alphas:
        D = {m: load(a.results, alpha, m) for m in a.methods}
        n = next(iter(D.values()))[0].shape[0]
        idx = block_indices(n, a.block, a.B, a.seed)
        sm = summ[np.isclose(summ.alpha, alpha)]
        rows = []
        for m in a.methods:
            bm, p0 = metrics(*D[m], idx, S), point(*D[m], S)
            for l in li:
                s_ = sm[(sm.method == m) & (sm.lead_h == leads_h[l])].iloc[0]
                ci = {k: np.percentile(bm[k][:, l], [2.5, 97.5]) for k in bm}
                rows.append(dict(
                    method=m, lead_h=leads_h[l],
                    coverage=fmt(p0["coverage"][l], *ci["coverage"], 3),
                    score=fmt(p0["score"][l], *ci["score"], 2),
                    vif=fmt(p0["vif"][l], *ci["vif"], 2),
                    persist_excess=round(float(s_.miss_persist - s_.miss_persist_null), 3),
                    q05=round(float(s_.cov_day_q05), 3)))
        out = os.path.join(a.results, table_name(alpha))
        pd.DataFrame(rows).to_csv(out, index=False)
        print("saved", out)
        pr = pd.DataFrame(paired_rows(alpha, a.block, a.methods, D))
        out = os.path.join(a.results, f"table_paired_alpha{tag(alpha)}.csv")
        pr.to_csv(out, index=False)
        print("saved", out)

        if a.block_lens and np.isclose(alpha, 0.1):
            sens = pd.concat([pd.DataFrame(paired_rows(alpha, b, a.methods, D)) for b in a.block_lens])
            out = os.path.join(a.results, "block_sensitivity.csv")
            sens.to_csv(out, index=False)
            print("saved", out)


if __name__ == "__main__":
    main()
