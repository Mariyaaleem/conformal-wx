#!/usr/bin/env python3
"""
paired_hres.py -- paired block-bootstrap comparison of the day-scale model with the benchmark methods.

Run from the repository root (needs conformal_lib.py, decompose.py and feasible.py in the same folder).
Written for the HRES file, but works for any forecast file as long as feasible.py --save-daily was run on
the same --npz and 03_run_benchmark.py/06_extra_diagnostics.py were run on it (for the self-check).

Step 1 (once): save the day-scale model's daily series
  python feasible.py --npz data/preds_hres.npz --boot 200 --save-daily results_feasible_hres/daily.npz --out results_feasible_hres
Step 2:
  python paired_hres.py --npz data/preds_hres.npz --daily results_feasible_hres/daily.npz --results results_hres --boot 1000

What it does
  * Rebuilds each benchmark method's intervals with the repository's own build_intervals(), using the same
    calibration/test split, season labels and feedback delays as 03_run_benchmark.py.
  * Self-check: recomputes VIF (heterogeneity-aware) and the interval score for every method and lead and compares them
    with results/vif_het_alpha*.csv and results/summary.csv. If they do not match it stops (use --skip-check to override).
  * Paired moving-block bootstrap over test days (same resampled days for both methods):
        vif_diff = VIF(method) - VIF(day-scale model)       > 0 means the day-scale model has LOWER VIF
        is_diff  = IS(method)  - IS(day-scale model)        > 0 means the day-scale model has LOWER (better) interval score
    with 95% percentile intervals. Marginal coverage differs between methods, so read VIF differences with the coverage
    columns in the CSV.

Variance convention: this script computes VIF with the population variance (ddof=0, as decompose.py and feasible.py);
the repository's vif_het column uses the sample variance (ddof=1), larger by n/(n-1) (about 0.1% for 1,086 days, i.e.
at most about 0.01 in VIF). The self-check tolerance (0.02 in VIF) is chosen to cover this and nothing more.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import decompose as D

DEFAULT_METHODS = ["split_local", "mondrian_season", "aci_0.005", "aci_0.02", "rolling_90", "rolling_365"]


def counts(idx, n):
    B = idx.shape[0]
    U = np.zeros((B, n), dtype=np.float32)
    np.add.at(U, (np.repeat(np.arange(B), idx.shape[1]), idx.ravel()), 1.0)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="forecast file (keys: pred, true, init_time, split, leads_h, lat, lon)")
    ap.add_argument("--cal-label", default="calib")
    ap.add_argument("--test-label", default="test")
    ap.add_argument("--daily", required=True, help="npz written by feasible.py --save-daily")
    ap.add_argument("--results", required=True, help="folder written by 03_run_benchmark.py (for the self-check)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="csv path (default: <results>/paired_dayscale_alpha*.csv)")
    ap.add_argument("--skip-check", action="store_true", help="skip the comparison with the saved benchmark results")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if not os.path.isdir(a.results):
        sys.exit(f"--results {a.results!r} is not a folder; run 03_run_benchmark.py/06_extra_diagnostics.py first (it is also where the output csv is written)")

    import conformal_lib as CL
    d = np.load(a.npz, allow_pickle=False)
    pred, true, split = d["pred"], d["true"], d["split"]
    init = d["init_time"].astype("datetime64[D]")
    leads_h, lat = d["leads_h"], d["lat"]
    res = np.abs(true - pred).astype(np.float32)
    del pred, true
    cal, test = res[split == a.cal_label], res[split == a.test_label]
    ic, it = init[split == a.cal_label], init[split == a.test_label]
    assert len(ic) > 0 and len(it) > 0, f"no rows with split == {a.cal_label!r} / {a.test_label!r}; labels in file: {list(np.unique(split))}"
    assert ic.max() < it.min(), "all calibration inits must be earlier than all test inits"
    assert np.all(np.diff(it).astype(int) == 1), "test inits must be one per day without gaps"
    season_c = CL.months_to_season(ic.astype("datetime64[M]").astype(int) % 12 + 1)
    season_t = CL.months_to_season(it.astype("datetime64[M]").astype(int) % 12 + 1)
    delays = CL.lead_delays(leads_h)
    nt, L, H, W = test.shape
    P = H * W
    w = D.area_weights(lat, W)
    rf = test.reshape(nt, L, P)

    day = np.load(a.daily)
    Mdl, ISdl = day["M"], day["IS"]
    assert Mdl.shape == (nt, L, P) and ISdl.shape == (nt, L), (
        f"daily file shape {Mdl.shape} does not match test shape {(nt, L, P)}; rerun feasible.py --save-daily on the same npz")
    assert np.array_equal(day["leads_h"], leads_h)

    alpha_tag = str(a.alpha).replace(".", "_")
    vif_csv = os.path.join(a.results, f"vif_het_alpha{alpha_tag}.csv")
    sum_csv = os.path.join(a.results, "summary.csv")
    ref_vif = pd.read_csv(vif_csv) if os.path.exists(vif_csv) else None
    ref_sum = pd.read_csv(sum_csv) if os.path.exists(sum_csv) else None
    if ref_sum is not None:
        ref_sum = ref_sum[np.isclose(ref_sum.alpha, a.alpha)]

    idx = D.block_idx(nt, a.block, a.boot, rng)
    U = counts(idx, nt)
    # day-scale model: VIF replicates per lead (computed once)
    v_dl_pt = [D.vif_cov(Mdl[:, l], w)[0] for l in range(L)]
    v_dl_bs = [D.boot_vif(Mdl[:, l], w, idx) for l in range(L)]

    rows, check = [], []
    for m in a.methods:
        q = CL.build_intervals(m, cal, test, a.alpha, season_c, season_t, delays)
        assert q.shape == test.shape, f"{m}: interval array shape {q.shape} != test shape {test.shape}"
        if not np.isfinite(q).all():
            print(f"WARNING: {m} has {int((~np.isfinite(q)).sum())} non-finite half-widths; interval scores may be inf/nan")
        qf = q.reshape(nt, L, P)
        for l in range(L):
            Mm = rf[:, l] > qf[:, l]
            ISm = (w * (2 * qf[:, l] + (2 / a.alpha) * np.maximum(rf[:, l] - qf[:, l], 0))).sum(-1)
            v_pt, cov = D.vif_cov(Mm, w)
            is_pt = float(ISm.mean())
            if ref_vif is not None:
                r = ref_vif[(ref_vif.method == m) & (ref_vif.lead_h == int(leads_h[l]))]
                check.append((m, int(leads_h[l]), abs(v_pt - float(r.vif_het.iloc[0])) if len(r) else np.nan,
                              abs(is_pt - float(ref_sum[(ref_sum.method == m) & (ref_sum.lead_h == int(leads_h[l]))].interval_score.iloc[0]))
                              if ref_sum is not None else np.nan))
            v_bs = D.boot_vif(Mm, w, idx)
            dv = v_bs - v_dl_bs[l]
            dIS = ISm - ISdl[:, l]
            dis_bs = (U @ dIS) / nt
            lo_v, hi_v = np.percentile(dv, [2.5, 97.5])
            lo_i, hi_i = np.percentile(dis_bs, [2.5, 97.5])
            rows.append(dict(method=m, lead_h=int(leads_h[l]), coverage_method=cov,
                             vif_method=v_pt, vif_dayscale=v_dl_pt[l], vif_diff=v_pt - v_dl_pt[l], vif_lo=lo_v, vif_hi=hi_v,
                             is_method=is_pt, is_dayscale=float(ISdl[:, l].mean()), is_diff=is_pt - float(ISdl[:, l].mean()),
                             is_lo=lo_i, is_hi=hi_i))
        del q, qf
        print(f"done {m}")

    chk = pd.DataFrame(check, columns=["method", "lead_h", "abs_vif_diff", "abs_is_diff"])
    if len(chk):
        print("\n=== self-check against the repository's saved results (max abs difference per method) ===")
        print(chk.groupby("method")[["abs_vif_diff", "abs_is_diff"]].max().round(4).to_string())
        bad = (chk.abs_vif_diff > 0.02).any() or (chk.abs_is_diff > 0.003).any()
        if bad and not a.skip_check:
            sys.exit("SELF-CHECK FAILED: the VIF or interval score recomputed here does not match the saved results "
                     "(VIF tolerance 0.02, interval score 0.003 K). Check that --npz, --alpha and --results belong to the "
                     "same forecast file and run of 03_run_benchmark.py/06_extra_diagnostics.py. "
                     "Do not use the comparison until this is resolved (--skip-check overrides).")
    df = pd.DataFrame(rows)
    out = a.out or os.path.join(a.results, f"paired_dayscale_alpha{alpha_tag}.csv")
    df.to_csv(out, index=False)

    def fmt(r, lo, hi, d):
        star = "*" if (r[lo] > 0 or r[hi] < 0) else " "
        return f"{r[d]:+.2f} [{r[lo]:+.2f},{r[hi]:+.2f}]{star}"

    pd.set_option("display.width", 250)
    for name, d_, lo, hi in (("VIF: method minus day-scale (positive = day-scale model lower)", "vif_diff", "vif_lo", "vif_hi"),
                             ("Interval score (K): method minus day-scale (positive = day-scale model better)", "is_diff", "is_lo", "is_hi")):
        t = df.copy()
        t["cell"] = t.apply(lambda r: fmt(r, lo, hi, d_) if d_ == "vif_diff" else
                            f"{r[d_]:+.3f} [{r[lo]:+.3f},{r[hi]:+.3f}]" + ("*" if (r[lo] > 0 or r[hi] < 0) else " "), axis=1)
        print(f"\n=== {name}; * = 95% paired interval excludes 0 ===")
        print(t.pivot(index="lead_h", columns="method", values="cell")[a.methods].to_string())
    print("\nmethod coverage (mean over days) for reference:")
    print(df.pivot(index="lead_h", columns="method", values="coverage_method")[a.methods].round(3).to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
