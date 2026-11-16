#!/usr/bin/env python3
"""
decompose.py -- what drives the spatial overdispersion index (VIF)?

Diagnostic only (NOT a method). Input: an .npz forecast file (data/preds.npz for the CNN,
data/preds_hres.npz for ECMWF HRES) with keys
  pred, true  (N, 8, 32, 64) float32   init_time, split   leads_h   lat   lon
where split is 'calib' (calibration rows) or 'test'.

Per-cell split conformal intervals are rebuilt from the 'calib' rows exactly as in the
paper (ceil((m+1)(1-alpha))-th smallest calibration residual per lead and cell), and
evaluated on the 'test' rows.

Variance convention: the VIF here uses the population variance of the daily coverage
(ddof=0). The vif_het column written by 06_extra_diagnostics.py uses the sample variance
(ddof=1), i.e. it is larger by the factor n/(n-1), about 0.1% for 1,086 test days.

Three analyses
  A. VIF of per-cell split conformal (heterogeneity-aware denominator). For the CNN file
     at alpha = 0.1 it is checked against the paper's Table 3 row (see --skip-table3-check).
  B. Day-scale decomposition (diagnostic upper/lower brackets). Normalised residuals
     are divided by a day-level scale, scores are recalibrated per cell, VIF recomputed.
       * oracle (self-normalised): scale = that day's area-weighted mean over ALL cells
         (global, or per region). Uses the verifying truth, so it is not a forecast
         procedure; it is also biased LOW because it forces the daily mean to equal 1.
       * split-half: scale from a random half of the cells, misses evaluated on the other
         half only (both directions averaged). No self-normalisation artifact, but noisy
         scale estimates add variance, so it is biased HIGH.
       * null floor: the identical pipeline on residuals permuted independently across
         days for every cell (no dependence of any kind). It shows what each pipeline
         returns when there is nothing to find. share_* = (base - pipeline)/(base - floor).
  C. VIF by distance: VIF - 1 is split exactly into contributions from pairs of cells
     in distance bins (diagonal terms give the "1"), plus miss correlation by distance.
  D. Moving-block bootstrap (blocks of --block test days) for A and B.

Usage (from the repository root):
  python decompose.py --npz data/preds.npz      --out results_decomp
  python decompose.py --npz data/preds_hres.npz --out results_decomp_hres --skip-table3-check
Self-test on synthetic data (no file needed):
  python decompose.py --synthetic structured --boot 20 --out /tmp/syn
  python decompose.py --synthetic null --boot 20 --out /tmp/null
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

PAPER_TABLE3_SPLIT_PERCELL = np.array([3.70, 4.40, 3.85, 4.58, 5.23, 6.03, 6.24, 6.47])  # alpha=0.1
DIST_EDGES_KM = np.array([0, 1000, 2000, 3000, 5000, 8000, 12000, 20100], dtype=float)


# --------------------------------------------------------------------------- basics
def area_weights(lat_deg, W):
    c = np.cos(np.deg2rad(lat_deg))
    w = np.repeat(c[:, None], W, axis=1).ravel()
    return w / w.sum()


def conf_quantile(res, alpha):
    """ceil((m+1)(1-alpha))-th smallest along axis 0 (inf if rank > m)."""
    m = res.shape[0]
    k = math.ceil((m + 1) * (1 - alpha))
    if k > m:
        return np.full(res.shape[1:], np.inf, dtype=res.dtype)
    return np.partition(res, k - 1, axis=0)[k - 1]


def vif_cov(M, w):
    """M: (n, P) 0/1 misses for one lead. Returns (VIF, mean coverage).
    VIF = Var_n(c) / sum_i w_i^2 m_i (1-m_i), c_n = area-weighted fraction covered,
    m_i = miss rate of cell i. Population variance (ddof=0); the repository's vif_het uses ddof=1."""
    M = np.asarray(M, dtype=np.float64)
    c = 1.0 - M @ w
    m = M.mean(axis=0)
    den = float(np.sum(w ** 2 * m * (1 - m)))
    return float(np.var(c) / den), float(c.mean())


def boot_vif(M, w, idx):
    """VIF for every bootstrap resample at once. M (n,P) 0/1 misses, idx (B,n) day indices.
    Identical to calling vif_cov on M[idx[b]] for each b, but uses day multiplicities (fast)."""
    n = M.shape[0]
    X = np.asarray(M, dtype=np.float32)
    c = 1.0 - X.astype(np.float64) @ w
    B = idx.shape[0]
    U = np.zeros((B, n), dtype=np.float32)
    np.add.at(U, (np.repeat(np.arange(B), idx.shape[1]), idx.ravel()), 1.0)
    m = (U @ X).astype(np.float64) / idx.shape[1]
    den = (m * (1 - m)) @ (w ** 2)
    mc = (U.astype(np.float64) @ c) / idx.shape[1]
    vc = (U.astype(np.float64) @ (c * c)) / idx.shape[1] - mc ** 2
    return vc / den


def region_groups(H, W, nlat, nlon):
    hh = (np.arange(H) * nlat) // H
    ww = (np.arange(W) * nlon) // W
    return (hh[:, None] * nlon + ww[None, :]).ravel()


def day_scale(U, groups, w):
    """U (n,L,P) normalised residuals -> (n,L,G) area-weighted mean per region (one matmul)."""
    n, L, P = U.shape
    G = int(groups.max()) + 1
    Wm = np.zeros((P, G), dtype=np.float32)
    for g in range(G):
        sel = groups == g
        Wm[sel, g] = (w[sel] / w[sel].sum()).astype(np.float32)
    s = U.reshape(-1, P).astype(np.float32) @ Wm
    return s.reshape(n, L, G).astype(np.float64)


def oracle_variant(Uc, Ut, groups, w, q, alpha, perm_c=None, perm_t=None):
    """Divide normalised residuals by the day scale, recalibrate per cell, return test misses
    (n_t, L, P) bool and the mean width ratio per lead (oracle / base)."""
    sc = day_scale(Uc, groups, w)
    st = day_scale(Ut, groups, w)
    if perm_c is not None:
        sc = sc[perm_c]
    if perm_t is not None:
        st = st[perm_t]
    gc = sc[..., groups]  # (n,L,P)
    gt = st[..., groups]
    Vc = (Uc / gc).astype(np.float32)
    Vt = (Ut / gt).astype(np.float32)
    qt = conf_quantile(Vc, alpha)  # (L,P)
    M = Vt > qt
    num = (w * q * (gt * qt).mean(axis=0)).sum(axis=-1)  # (L,)
    den = (w * q).sum(axis=-1)
    return M, num / den


def split_half(Uc, Ut, selA, selB, w, alpha):
    """Day scale from cells selA, misses on cells selB. Returns misses (n_t,L,|B|) and weights."""
    wA = (w[selA] / w[selA].sum()).astype(np.float32)
    sc = (Uc[:, :, selA].astype(np.float32) @ wA)[..., None].astype(np.float64)
    st = (Ut[:, :, selA].astype(np.float32) @ wA)[..., None].astype(np.float64)
    Vc = (Uc[:, :, selB] / sc).astype(np.float32)
    Vt = (Ut[:, :, selB] / st).astype(np.float32)
    qB = conf_quantile(Vc, alpha)
    wB = w[selB] / w[selB].sum()
    return Vt > qB, wB


# --------------------------------------------------------------------------- distance decomposition
def great_circle_km(lat_deg, lon_deg):
    H, W = len(lat_deg), len(lon_deg)
    la = np.deg2rad(np.repeat(lat_deg, W))
    lo = np.deg2rad(np.tile(lon_deg, H))
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def vif_by_distance(M, w, binidx, nb):
    """M (n,P). Returns (vif, contributions[nb], corr_mean[nb]); vif = 1 + sum(contributions)."""
    X = np.asarray(M, dtype=np.float64)
    n = X.shape[0]
    Xc = X - X.mean(0)
    cov = Xc.T @ Xc / n
    d = np.diag(cov).copy()
    den = float(np.sum(w ** 2 * d))
    ww = np.outer(w, w)
    contrib = np.bincount(binidx.ravel(), weights=(ww * cov).ravel(), minlength=nb + 1)[:nb] / den
    sd = np.sqrt(d)
    ok = sd > 0
    corr = cov / np.where(ok[:, None] & ok[None, :], sd[:, None] * sd[None, :], np.nan)
    cm = np.array([np.nanmean(corr[binidx == b]) if np.any(binidx == b) else np.nan for b in range(nb)])
    var_c = float(w @ cov @ w)
    return var_c / den, contrib, cm


# --------------------------------------------------------------------------- synthetic data
def make_synthetic(kind, seed):
    rng = np.random.default_rng(seed)
    n_cal, n_test, L, H, W = 730, 1086, 8, 32, 64
    n = n_cal + n_test
    lat = np.linspace(-87.1875, 87.1875, H)
    lon = np.arange(W) * 5.625
    cell_scale = (1.0 + 0.8 * np.abs(np.sin(np.deg2rad(lat))))[:, None]
    z = rng.standard_normal((n * L, H, W))
    if kind == "structured":
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        filt = np.exp(-2 * np.pi ** 2 * 3.0 ** 2 * (ky ** 2 + kx ** 2))
        z = np.fft.ifft2(np.fft.fft2(z) * filt).real
        z /= z.std()
    field = z.reshape(n, L, H, W)
    if kind == "structured":
        logg = np.zeros((n, L))
        eps = rng.standard_normal((n, L))
        for t in range(1, n):
            logg[t] = 0.8 * logg[t - 1] + math.sqrt(1 - 0.64) * 0.35 * eps[t]
        g = np.exp(logg)
    else:
        g = np.ones((n, L))
    true = (g[:, :, None, None] * cell_scale[None, None] * field).astype(np.float32)
    pred = np.zeros_like(true)
    init = pd.date_range("2017-01-01", periods=n, freq="D").strftime("%Y-%m-%dT%H:%M:%S").astype(str).to_numpy().astype("<U28")
    split = np.array(["calib"] * n_cal + ["test"] * n_test)
    leads = np.array([6, 12, 24, 48, 72, 120, 168, 240])
    return pred, true, init, split, leads, lat, lon


# --------------------------------------------------------------------------- bootstrap
def block_idx(n, block, B, rng):
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="forecast file with keys pred, true, init_time, split, leads_h, lat, lon")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default="results_decomp")
    ap.add_argument("--cal-label", default="calib", help="value of the split key for calibration rows")
    ap.add_argument("--test-label", default="test", help="value of the split key for test rows")
    ap.add_argument("--skip-table3-check", action="store_true",
                    help="do not compare with the paper's Table 3 (that table is for the CNN preds.npz only; use for HRES)")
    ap.add_argument("--nlat", type=int, default=4, help="regional oracle: latitude bands")
    ap.add_argument("--nlon", type=int, default=4, help="regional oracle: longitude blocks")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates (0 = skip)")
    ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", choices=["none", "structured", "null"], default="none")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- load
    if a.synthetic != "none":
        pred, true, init, split, leads, lat, lon = make_synthetic(a.synthetic, a.seed)
        print(f"[synthetic: {a.synthetic}] self-test, results are NOT about weather")
    else:
        if not a.npz:
            sys.exit("give --npz PATH or --synthetic")
        d = np.load(a.npz, allow_pickle=False)
        pred, true, init, split = d["pred"], d["true"], d["init_time"], d["split"]
        leads, lat, lon = d["leads_h"], d["lat"], d["lon"]
    N, L, H, W = pred.shape
    P = H * W
    print("splits:", dict(zip(*np.unique(split, return_counts=True))), "| leads_h:", list(leads))
    if np.abs(lat).max() <= 3.2:  # radians?
        sys.exit("lat looks like radians; convert to degrees first")
    w = area_weights(lat, W)

    def pick(label):
        idx = np.where(split == label)[0]
        if len(idx) == 0:
            sys.exit(f"no rows with split == {label!r}; labels in the file: {list(np.unique(split))} "
                     f"(set --cal-label / --test-label)")
        t = pd.to_datetime(init[idx])
        idx = idx[np.argsort(t.values, kind="stable")]
        return idx, pd.to_datetime(init[idx])

    ci, tc = pick(a.cal_label)
    ti, tt = pick(a.test_label)
    gaps = np.diff(tt.values).astype("timedelta64[D]").astype(int)
    print(f"cal n={len(ci)} ({tc[0].date()}..{tc[-1].date()}), test n={len(ti)} ({tt[0].date()}..{tt[-1].date()}); "
          f"test day gaps != 1: {(gaps != 1).sum()}")

    Rc = np.abs(true[ci] - pred[ci]).reshape(len(ci), L, P).astype(np.float32)
    Rt = np.abs(true[ti] - pred[ti]).reshape(len(ti), L, P).astype(np.float32)
    del pred, true

    # ---- A. base per-cell split conformal
    q = conf_quantile(Rc, a.alpha)  # (L,P)
    Mb = Rt > q
    Uc, Ut = Rc / q, Rt / q
    del Rc, Rt
    groups_g = np.zeros(P, dtype=int)
    groups_r = region_groups(H, W, a.nlat, a.nlon)

    Mg, wr_g = oracle_variant(Uc, Ut, groups_g, w, q, a.alpha)
    Mr, wr_r = oracle_variant(Uc, Ut, groups_r, w, q, a.alpha)

    # split-half (random fixed partition of cells; both directions)
    perm_cells = rng.permutation(P)
    halfA, halfB = np.sort(perm_cells[: P // 2]), np.sort(perm_cells[P // 2:])
    M_ab, w_b = split_half(Uc, Ut, halfA, halfB, w, a.alpha)   # scale from A, evaluate B
    M_ba, w_a = split_half(Uc, Ut, halfB, halfA, w, a.alpha)   # scale from B, evaluate A

    # null floor: permute days independently for every cell/lead, rerun identical pipelines
    Uc_n = rng.permuted(Uc, axis=0)
    Ut_n = rng.permuted(Ut, axis=0)
    Mb_n = Ut_n > 1.0                                          # U = r/q so cal quantile is 1
    Mg_n, _ = oracle_variant(Uc_n, Ut_n, groups_g, w, q, a.alpha)
    Mr_n, _ = oracle_variant(Uc_n, Ut_n, groups_r, w, q, a.alpha)
    M_ab_n, _ = split_half(Uc_n, Ut_n, halfA, halfB, w, a.alpha)
    M_ba_n, _ = split_half(Uc_n, Ut_n, halfB, halfA, w, a.alpha)
    del Uc_n, Ut_n

    rows = []
    for l in range(L):
        vb, cb = vif_cov(Mb[:, l], w)
        vg, cg = vif_cov(Mg[:, l], w)
        vr, cr = vif_cov(Mr[:, l], w)
        # split-half and its matching base (same cells)
        vh = 0.5 * (vif_cov(M_ab[:, l], w_b)[0] + vif_cov(M_ba[:, l], w_a)[0])
        vbh = 0.5 * (vif_cov(Mb[:, l][:, halfB], w_b)[0] + vif_cov(Mb[:, l][:, halfA], w_a)[0])
        vh_n = 0.5 * (vif_cov(M_ab_n[:, l], w_b)[0] + vif_cov(M_ba_n[:, l], w_a)[0])
        rows.append(dict(
            lead_h=int(leads[l]), cov_base=cb, vif_base=vb,
            vif_oracle_global=vg, vif_oracle_region=vr,
            vif_split_half=vh, vif_base_halfcells=vbh,
            floor_base=vif_cov(Mb_n[:, l], w)[0],
            floor_oracle_global=vif_cov(Mg_n[:, l], w)[0],
            floor_oracle_region=vif_cov(Mr_n[:, l], w)[0],
            floor_split_half=vh_n,
            cov_oracle_global=cg, cov_oracle_region=cr,
            width_ratio_global=wr_g[l], width_ratio_region=wr_r[l]))
    df = pd.DataFrame(rows)
    df["share_oracle_global"] = (df.vif_base - df.vif_oracle_global) / (df.vif_base - df.floor_oracle_global)
    df["share_oracle_region"] = (df.vif_base - df.vif_oracle_region) / (df.vif_base - df.floor_oracle_region)
    df["share_split_half"] = (df.vif_base_halfcells - df.vif_split_half) / (df.vif_base_halfcells - df.floor_split_half)

    # ---- table check (CNN preds.npz only: PAPER_TABLE3_SPLIT_PERCELL are the CNN values)
    if a.synthetic == "none" and abs(a.alpha - 0.1) < 1e-9 and L == 8 and not a.skip_table3_check:
        df["paper_table3"] = PAPER_TABLE3_SPLIT_PERCELL
        df["diff_vs_paper"] = df.vif_base - df.paper_table3
        worst = df.diff_vs_paper.abs().max()
        print(f"\nTable 3 check (CNN preds.npz only; per-cell split, alpha=0.1): max |diff| = {worst:.3f}")
        if worst > 0.05:
            print("  WARNING: does not reproduce the paper's Table 3. This is expected for any file other than the\n"
                  "  released CNN preds.npz (e.g. HRES: rerun with --skip-table3-check). For the CNN file, do not\n"
                  "  interpret the decomposition until resolved (check split labels, residual definition, quantile\n"
                  "  rule, weights).")

    # ---- C. VIF by distance (base and oracle-global)
    dist = great_circle_km(lat, lon)
    nb = len(DIST_EDGES_KM) - 1
    binidx = np.clip(np.digitize(dist, DIST_EDGES_KM) - 1, 0, nb - 1)
    binidx[np.arange(P), np.arange(P)] = nb  # diagonal -> ignored bin
    drows = []
    for name, M in (("base", Mb), ("oracle_global", Mg)):
        for l in range(L):
            v, con, cm = vif_by_distance(M[:, l], w, binidx, nb)
            assert abs(v - (1 + con.sum())) < 1e-6, (name, l, v, 1 + con.sum())
            for b in range(nb):
                drows.append(dict(variant=name, lead_h=int(leads[l]), vif=v,
                                  bin_km=f"{int(DIST_EDGES_KM[b])}-{int(DIST_EDGES_KM[b + 1])}",
                                  vif_contribution=con[b], mean_miss_corr=cm[b]))
    dd = pd.DataFrame(drows)

    # ---- D. block bootstrap (paired)
    if a.boot > 0:
        idx = block_idx(len(ti), a.block, a.boot, rng)
        lo_hi = []
        f = lambda x: tuple(np.percentile(x, [2.5, 97.5]))
        for l in range(L):
            Xb = Mb[:, l].astype(np.float32)
            Xg = Mg[:, l].astype(np.float32)
            Xab, Xba = M_ab[:, l].astype(np.float32), M_ba[:, l].astype(np.float32)
            Xbb, Xba_b = Mb[:, l][:, halfB].astype(np.float32), Mb[:, l][:, halfA].astype(np.float32)
            vb_ = boot_vif(Xb, w, idx)
            vg_ = boot_vif(Xg, w, idx)
            vh_ = 0.5 * (boot_vif(Xab, w_b, idx) + boot_vif(Xba, w_a, idx))
            vbh_ = 0.5 * (boot_vif(Xbb, w_b, idx) + boot_vif(Xba_b, w_a, idx))
            lo_hi.append(dict(lead_h=int(leads[l]),
                              base_lo=f(vb_)[0], base_hi=f(vb_)[1],
                              oracle_lo=f(vg_)[0], oracle_hi=f(vg_)[1],
                              split_lo=f(vh_)[0], split_hi=f(vh_)[1],
                              split_drop_lo=f(vbh_ - vh_)[0], split_drop_hi=f(vbh_ - vh_)[1]))
        df = df.merge(pd.DataFrame(lo_hi), on="lead_h")

    df.to_csv(os.path.join(a.out, "decomp_vif.csv"), index=False)
    dd.to_csv(os.path.join(a.out, "vif_by_distance.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== VIF decomposition (alpha=%.2f) ===" % a.alpha)
    cols = ["lead_h", "cov_base", "vif_base", "vif_oracle_global", "floor_oracle_global", "share_oracle_global",
            "vif_oracle_region", "floor_oracle_region", "share_oracle_region"]
    print(df[cols].round(3).to_string(index=False))
    print("\n=== split-half (scale from one half of cells, misses on the other) ===")
    cols2 = ["lead_h", "vif_base_halfcells", "vif_split_half", "floor_split_half", "share_split_half", "width_ratio_global"]
    print(df[cols2].round(3).to_string(index=False))
    print("\nfloor_base (should be ~1):", df.floor_base.round(2).tolist())
    print("\n=== VIF - 1 by distance (base) ===")
    print(dd[dd.variant == "base"].pivot(index="lead_h", columns="bin_km", values="vif_contribution")
          [[f"{int(DIST_EDGES_KM[b])}-{int(DIST_EDGES_KM[b + 1])}" for b in range(nb)]].round(2).to_string())

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(L)
        lab = [f"{int(h)}h" if h < 24 else f"{int(h // 24)}d" for h in leads]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(x, df.vif_base, "o-", label="per-cell split (base)")
        ax[0].plot(x, df.vif_oracle_global, "s-", label="oracle global day-scale")
        ax[0].plot(x, df.vif_oracle_region, "^-", label=f"oracle regional ({a.nlat}x{a.nlon})")
        ax[0].plot(x, df.vif_split_half, "d-", label="split-half scale (conservative)")
        ax[0].plot(x, df.floor_oracle_global, "k:", label="null floor of the oracle pipeline")
        ax[0].axhline(1, color="k", lw=0.6)
        ax[0].set_xticks(x, lab); ax[0].set_ylabel("VIF"); ax[0].legend(fontsize=8)
        ax[0].set_title("Spatial overdispersion before/after oracle day-scale removal")
        piv = dd[dd.variant == "base"].pivot(index="lead_h", columns="bin_km", values="vif_contribution")
        piv = piv[[f"{int(DIST_EDGES_KM[b])}-{int(DIST_EDGES_KM[b + 1])}" for b in range(nb)]]
        bottom = np.zeros(L)
        for c in piv.columns:
            ax[1].bar(x, piv[c].values, bottom=bottom, label=c + " km"); bottom += piv[c].values
        ax[1].set_xticks(x, lab); ax[1].set_ylabel("VIF - 1 contribution"); ax[1].legend(fontsize=7)
        ax[1].set_title("Where the excess comes from (pair distance), base")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "decomposition.png"), dpi=150)
        print(f"\nsaved {a.out}/decomp_vif.csv, vif_by_distance.csv, decomposition.png")
    except Exception as e:  # plotting is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
