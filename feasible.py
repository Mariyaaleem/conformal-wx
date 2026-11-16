#!/usr/bin/env python3
"""
feasible.py -- can the day-level error scale be predicted at issuance time, and does that help?

Needs decompose.py in the same folder. Input: an .npz forecast file with keys pred, true
(N, L, H, W), init_time, split ('calib' / 'test'), leads_h, lat, lon (see data/README.md).

PRIMARY METHOD (fixed before any real-data run; everything else is an ablation, all reported):
  A single global day-scale g[n,l] = area-weighted mean over cells of u = |err| / q_cell
  (q_cell = per-cell split-conformal quantile from the calibration rows). The scale is
  PREDICTED from information that is verified by the issuance time of forecast n:
     season : sin/cos (2 harmonics) of the valid day of year
     fresh  : log g of the 24 h forecast issued the day before (valid at n)
     stale  : log g of the same-lead forecast whose verification arrives at n
              (init n - d, d = max(1, lead_days))
  log g[n,l] ~ OLS on these. Scores are divided by the predicted scale, then calibrated
  per cell by split conformal. Calibration-day predictions are out-of-fold (fit on one
  calibration year, predict the other); test-day predictions use a model fit on all
  calibration days. Nothing from a test day's own residuals is used for its own scale.

Reported per lead: out-of-sample R^2 of log g, VIF, coverage, 5th pct of daily coverage,
width ratio vs per-cell split, and the share of the oracle gain captured.
Assumes verification (ERA5) is available when a forecast is valid (as in the paper's protocol).

Run:  python feasible.py --npz data/preds.npz --boot 200 --out results_feasible
      python feasible.py --npz data/preds_hres.npz --boot 200 --save-daily results_feasible_hres/daily.npz --out results_feasible_hres
Self-test:  python feasible.py --synthetic coupled --boot 20 --out /tmp/f1
            python feasible.py --synthetic null --boot 20 --out /tmp/f2
"""
import argparse, math, os, sys
import numpy as np
import pandas as pd
import decompose as D

PRIMARY = "PRIMARY season+fresh+stale"
VARIANTS = {
    "season": ["season"],
    "fresh": ["fresh"],
    "stale": ["stale"],
    "fresh+stale": ["fresh", "stale"],
    PRIMARY: ["season", "fresh", "stale"],
}


def make_coupled(kind, seed):
    """Synthetic: a latent valid-time scale shared across leads (coupled) or none (null)."""
    rng = np.random.default_rng(seed)
    n_cal, n_test, L, H, W = 730, 1086, 8, 32, 64
    N = n_cal + n_test
    leads = np.array([6, 12, 24, 48, 72, 120, 168, 240])
    lat = np.linspace(-87.1875, 87.1875, H); lon = np.arange(W) * 5.625
    cell_scale = (1.0 + 0.8 * np.abs(np.sin(np.deg2rad(lat))))[:, None]
    z = rng.standard_normal((N * L, H, W))
    if kind == "coupled":
        ky, kx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
        z = np.fft.ifft2(np.fft.fft2(z) * np.exp(-2 * np.pi ** 2 * 9.0 * (ky ** 2 + kx ** 2))).real
        z /= z.std()
    field = z.reshape(N, L, H, W)
    T = N + 12
    lv = np.zeros(T); e = rng.standard_normal(T)
    for t in range(1, T):
        lv[t] = 0.9 * lv[t - 1] + math.sqrt(1 - 0.81) * 0.4 * e[t]
    d = np.maximum(1, np.round(leads / 24)).astype(int)
    if kind == "coupled":
        logg = np.stack([lv[np.arange(N) + d[l]] for l in range(L)], axis=1)
    else:
        logg = np.zeros((N, L))
    true = (np.exp(logg)[:, :, None, None] * cell_scale[None, None] * field).astype(np.float32)
    init = pd.date_range("2017-01-01", periods=N, freq="D").strftime("%Y-%m-%dT%H:%M:%S").astype(str).to_numpy().astype("<U28")
    split = np.array(["calib"] * n_cal + ["test"] * n_test)
    return np.zeros_like(true), true, init, split, leads, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="forecast file (keys: pred, true, init_time, split, leads_h, lat, lon)"); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cal-label", default="calib"); ap.add_argument("--test-label", default="test")
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_feasible")
    ap.add_argument("--synthetic", choices=["none", "coupled", "null"], default="none")
    ap.add_argument("--save-daily", default=None, help="npz path: save primary-method misses and daily interval scores for paired comparisons")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    if a.synthetic != "none":
        pred, true, init, split, leads, lat, lon = make_coupled(a.synthetic, a.seed)
        print(f"[synthetic: {a.synthetic}] self-test, not weather")
    else:
        if not a.npz: sys.exit("give --npz PATH or --synthetic {coupled,null}")
        d = np.load(a.npz, allow_pickle=False)
        pred, true, init, split = d["pred"], d["true"], d["init_time"], d["split"]
        leads, lat, lon = d["leads_h"], d["lat"], d["lon"]
    Nall, L, H, W = pred.shape; P = H * W
    w = D.area_weights(lat, W)

    keep = np.where(np.isin(split, [a.cal_label, a.test_label]))[0]
    t_all = pd.to_datetime(init[keep])
    order = keep[np.argsort(t_all.values, kind="stable")]
    tt = pd.to_datetime(init[order])
    gaps = np.diff(tt.values).astype("timedelta64[D]").astype(int)
    if (gaps != 1).any():
        sys.exit(f"calibration+test inits are not one init per day without gaps ({(gaps != 1).sum()} gaps); the lagged day-scale features would be wrong")
    N = len(order)
    is_cal = (split[order] == a.cal_label)
    n_cal = int(is_cal.sum())
    assert is_cal[:n_cal].all() and not is_cal[n_cal:].any(), "all calibration inits must be earlier than all test inits"
    print(f"series {tt[0].date()}..{tt[-1].date()}  cal={n_cal} test={N - n_cal}")

    R = np.abs(true[order] - pred[order]).reshape(N, L, P).astype(np.float32)
    del pred, true
    q = D.conf_quantile(R[:n_cal], a.alpha)
    U = R / q; del R
    g = D.day_scale(U, np.zeros(P, dtype=int), w)[..., 0]          # (N,L)
    logg = np.log(g)

    l24 = int(np.where(leads == 24)[0][0])
    # feedback delay in days; equals max(1, ceil(lead/24h)) for the eight leads used here (1,1,1,2,3,5,7,10)
    dl = np.maximum(1, np.round(leads / 24)).astype(int)
    # Drop the first BURN calibration days: the lagged features (fresh, stale) need up to 10 earlier days
    # (stale delay is 10 days at the 240 h lead), so the earliest rows would have NaN features.
    BURN = 10
    n_idx = np.arange(N)
    fresh = np.full(N, np.nan); fresh[1:] = logg[:-1, l24]           # 24h fcst from n-1, valid at n
    stale = np.full((N, L), np.nan)
    for l in range(L):
        stale[dl[l]:, l] = logg[:-dl[l], l]
    valid_doy = (tt.dayofyear.values[:, None] + np.round(leads / 24)[None, :])
    ang = 2 * np.pi * valid_doy / 365.25
    sea = np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], axis=-1)  # (N,L,4)

    # Note: at the 24 h lead the 'fresh' and 'stale' columns are identical (both are log g of the 24 h forecast
    # from the previous day), so 'fresh+stale' and PRIMARY are rank-deficient there; np.linalg.lstsq returns the
    # minimum-norm solution and the fitted values are unaffected.
    def X(keys, l, idx):
        cols = [np.ones(len(idx))]
        if "season" in keys: cols += [sea[idx, l, k] for k in range(4)]
        if "fresh" in keys: cols.append(fresh[idx])
        if "stale" in keys: cols.append(stale[idx, l])
        return np.column_stack(cols)

    calv = np.arange(BURN, n_cal); test = np.arange(n_cal, N)
    half = (BURN + n_cal) // 2
    F1, F2 = np.arange(BURN, half), np.arange(half, n_cal)
    Ut = U[test]
    base_M = [Ut[:, l, :] > 1.0 for l in range(L)]

    def pipeline(l, gc, gt):
        Vc = (U[calv, l, :] / gc[:, None]).astype(np.float32)
        qt = D.conf_quantile(Vc, a.alpha)
        Hn = gt[:, None] * qt[None, :]                       # half-width in units of q_cell
        M = Ut[:, l, :] > Hn
        qw = (w * q[l])[None, :]
        wr = float((qw * Hn).sum(1).mean() / (w * q[l]).sum())
        # interval score (alpha): width + (2/alpha) * excess of |error| beyond the half-width
        dis = (qw * (2 * Hn + (2 / a.alpha) * np.maximum(Ut[:, l, :] - Hn, 0))).sum(1)   # (n_test,)
        return M, wr, dis

    rows, store, store_is, is_base_daily = [], {}, {}, {}
    for l in range(L):
        yc, yt = logg[calv, l], logg[test, l]
        vif_b, cov_b = D.vif_cov(base_M[l], w)
        c_b = 1.0 - base_M[l].astype(float) @ w
        qw = (w * q[l])[None, :]
        is_base_daily[l] = (qw * (2.0 + (2 / a.alpha) * np.maximum(Ut[:, l, :] - 1.0, 0))).sum(1)
        row = dict(lead_h=int(leads[l]), vif_base=vif_b, cov_base=cov_b, p5_base=float(np.percentile(c_b, 5)),
                   is_base=float(is_base_daily[l].mean()))
        Mo, wro, _ = pipeline(l, g[calv, l], g[test, l])
        row["vif_oracle"] = D.vif_cov(Mo, w)[0]
        for name, keys in VARIANTS.items():
            gc = np.empty(len(calv))
            for tr, te in ((F2, F1), (F1, F2)):
                beta = np.linalg.lstsq(X(keys, l, tr), logg[tr, l], rcond=None)[0]
                gc[np.isin(calv, te).nonzero()[0]] = np.exp(X(keys, l, te) @ beta)
            beta = np.linalg.lstsq(X(keys, l, calv), yc, rcond=None)[0]
            yhat = X(keys, l, test) @ beta
            r2 = 1 - ((yt - yhat) ** 2).sum() / ((yt - yc.mean()) ** 2).sum()
            M, wr, dis = pipeline(l, gc, np.exp(yhat))
            v, cv = D.vif_cov(M, w)
            c = 1.0 - M.astype(float) @ w
            row[f"r2|{name}"] = r2; row[f"vif|{name}"] = v; row[f"is|{name}"] = float(dis.mean())
            if name == PRIMARY:
                row.update(cov_primary=cv, p5_primary=float(np.percentile(c, 5)), width_ratio_primary=wr)
                store[l] = M; store_is[l] = dis
        rows.append(row)
    df = pd.DataFrame(rows)
    df["is_ratio_primary"] = df[f"is|{PRIMARY}"] / df.is_base
    df["share_of_oracle_gain_primary"] = (df.vif_base - df[f"vif|{PRIMARY}"]) / (df.vif_base - df.vif_oracle)

    if a.boot > 0:
        idx = D.block_idx(len(test), a.block, a.boot, rng)
        ci = []
        for l in range(L):
            Xb, Xp = base_M[l].astype(np.float32), store[l].astype(np.float32)
            vb, vp = D.boot_vif(Xb, w, idx), D.boot_vif(Xp, w, idx)
            f = lambda x: np.percentile(x, [2.5, 97.5])
            isd = (is_base_daily[l][idx] - store_is[l][idx]).mean(1)      # >0: primary has lower (better) score
            ci.append(dict(lead_h=int(leads[l]), drop_lo=f(vb - vp)[0], drop_hi=f(vb - vp)[1],
                           is_drop_lo=f(isd)[0], is_drop_hi=f(isd)[1]))
        df = df.merge(pd.DataFrame(ci), on="lead_h")
    if a.save_daily:
        np.savez_compressed(a.save_daily, M=np.stack([store[l] for l in range(L)], 1).astype(np.uint8),
                            IS=np.stack([store_is[l] for l in range(L)], 1),
                            base_M=np.stack(base_M, 1).astype(np.uint8),
                            base_IS=np.stack([is_base_daily[l] for l in range(L)], 1), leads_h=leads)
        print("saved daily series to", a.save_daily)
    df.to_csv(os.path.join(a.out, "feasible.csv"), index=False)

    pd.set_option("display.width", 220)
    r2c = ["lead_h"] + [f"r2|{n}" for n in VARIANTS]
    print("\n=== out-of-sample R^2 of log day-scale (test days; 0 = no better than calibration mean) ===")
    print(df[r2c].round(3).to_string(index=False))
    vc = ["lead_h", "vif_base"] + [f"vif|{n}" for n in VARIANTS] + ["vif_oracle"]
    print("\n=== VIF (lower is better; oracle = own-day scale, not usable) ===")
    print(df[vc].round(2).to_string(index=False))
    pc = ["lead_h", "cov_base", "cov_primary", "p5_base", "p5_primary", "width_ratio_primary", "share_of_oracle_gain_primary"]
    if a.boot > 0: pc += ["drop_lo", "drop_hi"]
    print("\n=== PRIMARY method vs per-cell split (drop = base VIF - primary VIF, 95% block bootstrap) ===")
    print(df[pc].round(3).to_string(index=False))
    ic = ["lead_h", "is_base", "is|season", f"is|{PRIMARY}", "is_ratio_primary"] + (["is_drop_lo", "is_drop_hi"] if a.boot > 0 else [])
    print("\n=== mean interval score (alpha), area-weighted, in K; lower is better; is_drop = base - primary ===")
    print(df[ic].round(3).to_string(index=False))
    print(f"\nsaved {a.out}/feasible.csv")


if __name__ == "__main__":
    main()
