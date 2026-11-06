"""
06_extra_diagnostics.py
Numbers in the paper that are not produced by 03_run_benchmark.py:

  results/vif_het_alpha<a>.csv    heterogeneity-aware overdispersion index (paper eq. for VIF)
  results/threshold_sweep.csv     share of forecasts with daily coverage < 1 - alpha - delta
  results/bias_by_year.csv        mean forecast error y - yhat by init year (needs --preds)
  results/synthetic_check.csv     ranges for the synthetic validation (needs --iid_summary)

  python 06_extra_diagnostics.py --results results --preds data/preds.npz \
         [--iid_summary results_iid/summary.csv]

vif_het = Var_n(c_{n,l}) / sum_i w_i^2 m_{l,i}(1 - m_{l,i}), with m_{l,i} the test miss rate of cell i
(1 - cov_map from the benchmark runner) and w_i = cos(lat_h) / (W * sum_h cos(lat_h)); Var uses ddof=1,
like the homogeneous VIF in conformal_lib.evaluate.
"""
import argparse
import os
import numpy as np
import pandas as pd
from conformal_lib import METHOD_NAMES


def tag(alpha):
    return str(alpha).replace(".", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.05])
    ap.add_argument("--methods", nargs="+", default=METHOD_NAMES)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--preds", default=None, help="data/preds.npz, for the bias-by-year table")
    ap.add_argument("--iid_summary", default=None, help="summary.csv of the synthetic i.i.d. run")
    a = ap.parse_args()

    meta = np.load(os.path.join(a.results, "meta.npz"))
    leads_h, lat, W = [int(x) for x in meta["leads_h"]], meta["lat"], len(meta["lon"])
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()               # row weights, sum to 1

    het_rows, sweep_rows = [], []
    for alpha in a.alphas:
        het = []
        for m in a.methods:
            r = np.load(os.path.join(a.results, f"alpha_{alpha}", f"{m}.npz"))
            c = r["spatial_cov"].astype(np.float64)               # (N, L)
            miss = 1.0 - r["cov_map"].astype(np.float64)          # (L, H, W) per-cell miss rate
            ref = ((w ** 2)[None, :] * (miss * (1 - miss)).sum(-1)).sum(-1) / W ** 2   # sum_i w_i^2 m(1-m)
            vif = c.var(axis=0, ddof=1) / ref
            for l, h in enumerate(leads_h):
                het.append(dict(method=m, lead_h=h, vif_het=round(float(vif[l]), 4)))
                for d in a.deltas:
                    sweep_rows.append(dict(method=m, alpha=alpha, lead_h=h, delta=d,
                                           share_below=float((c[:, l] < 1 - alpha - d).mean())))
        out = os.path.join(a.results, f"vif_het_alpha{tag(alpha)}.csv")
        pd.DataFrame(het).to_csv(out, index=False)
        print("saved", out)
    out = os.path.join(a.results, "threshold_sweep.csv")
    pd.DataFrame(sweep_rows).to_csv(out, index=False)
    print("saved", out)

    if a.preds:
        d = np.load(a.preds)
        split, init = d["split"], d["init_time"].astype("datetime64[D]")
        err = (d["true"] - d["pred"]).astype(np.float64).mean(axis=(2, 3))     # (N, L), unweighted over cells
        year = init.astype("datetime64[Y]").astype(int) + 1970
        rows = []
        for y in np.unique(year):
            e = err[year == y].mean(0)
            rows.append({"year": int(y), "role": ("calib" if (split[year == y] == "calib").all() else
                                      "test" if (split[year == y] == "test").all() else "mixed"),
                         **{f"lead_{h}h": round(float(v), 3) for h, v in zip(d["leads_h"], e)}})
        out = os.path.join(a.results, "bias_by_year.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("saved", out)

    if a.iid_summary:
        s = pd.read_csv(a.iid_summary)
        rows = []
        for m, g in s.groupby("method"):
            rows.append(dict(method=m, alpha=g.alpha.iloc[0],
                             mean_coverage_over_leads=g.coverage.mean(),
                             vif_min=g.vif.min(), vif_max=g.vif.max(),
                             max_abs_persist_minus_null=(g.miss_persist - g.miss_persist_null).abs().max()))
        out = os.path.join(a.results, "synthetic_check.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("saved", out)


if __name__ == "__main__":
    main()
