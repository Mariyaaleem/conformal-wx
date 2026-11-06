"""
03_run_benchmark.py
Runs every conformal method on the saved forecasts and writes metrics.

  python 03_run_benchmark.py --preds data/preds.npz --out results
Options: --alphas 0.1 0.2 0.05   --methods split_local aci_0.005 ...   --skip_rolling
NOTE: summary.csv is overwritten on every run, so use ONE run with all alphas and all methods
(the paper uses --alphas 0.1 0.2 0.05).
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from conformal_lib import (METHOD_NAMES, build_intervals, evaluate, lead_delays,
                           months_to_season)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/preds.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    ap.add_argument("--methods", nargs="+", default=METHOD_NAMES)
    ap.add_argument("--skip_rolling", action="store_true")
    a = ap.parse_args()

    d = np.load(a.preds, allow_pickle=False)
    pred, true, split = d["pred"], d["true"], d["split"]
    init = d["init_time"].astype("datetime64[D]")
    leads_h, lat, lon = d["leads_h"], d["lat"], d["lon"]
    res = np.abs(true - pred).astype(np.float32)
    cal, test = res[split == "calib"], res[split == "test"]
    ic, it = init[split == "calib"], init[split == "test"]
    assert ic.max() < it.min(), "calibration period must precede test period"
    assert np.all(np.diff(it).astype(int) == 1), "test inits must be consecutive days"
    season_c = months_to_season(ic.astype("datetime64[M]").astype(int) % 12 + 1)
    season_t = months_to_season(it.astype("datetime64[M]").astype(int) % 12 + 1)
    year_t = it.astype("datetime64[Y]").astype(int) + 1970
    delays = lead_delays(leads_h)
    print(f"cal {cal.shape}, test {test.shape}, leads(h) {list(leads_h)}, delays {delays}")

    methods = [m for m in a.methods if not (a.skip_rolling and m.startswith("rolling_"))]
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "meta.npz"), leads_h=leads_h, lat=lat, lon=lon)
    rows = []
    for alpha in a.alphas:
        adir = os.path.join(a.out, f"alpha_{alpha}")
        os.makedirs(adir, exist_ok=True)
        for m in methods:
            t0 = time.time()
            q = build_intervals(m, cal, test, alpha, season_c, season_t, delays)
            r = evaluate(q, test, alpha, lat, season_t, year_t)
            del q
            np.savez(os.path.join(adir, f"{m}.npz"), **r)
            for l, h in enumerate(leads_h):
                rows.append(dict(
                    method=m, alpha=alpha, lead_h=int(h),
                    coverage=r["coverage"][l], width_K=r["width"][l],
                    interval_score=r["interval_score"][l],
                    vif=r["vif"][l], n_eff=r["n_eff"][l], bad_day_frac=r["bad_day_frac"][l],
                    miss_persist=r["miss_persist"][l],
                    miss_persist_null=r["miss_persist_null"][l],
                    cov_day_q05=np.quantile(r["spatial_cov"][:, l], 0.05),
                    cov_day_min=r["spatial_cov"][:, l].min(),
                    worst_season_gap=np.max(np.abs(r["cov_season"][:, l] - (1 - alpha))),
                    worst_year_gap=np.max(np.abs(r["cov_year"][:, l] - (1 - alpha)))))
            print(f"[alpha={alpha}] {m:16s} cov(avg over leads)={np.mean(r['coverage']):.3f} "
                  f"({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("saved", os.path.join(a.out, "summary.csv"))


if __name__ == "__main__":
    main()
