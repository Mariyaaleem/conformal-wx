"""
conformal_lib.py
Conformal prediction methods + dependence-aware evaluation metrics for gridded,
multi-lead weather forecasts. Pure numpy/scipy.

Array convention: residual arrays are |y - yhat| with shape (N, L, H, W)
  N = forecast initialisations (daily), L = lead times, H = latitudes, W = longitudes.
"""
import numpy as np
from scipy.stats import norm

_SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
SEASON_NAMES = ["DJF", "MAM", "JJA", "SON"]


def months_to_season(months):
    return np.array([_SEASON[int(m)] for m in months])


def lead_delays(leads_h, init_spacing_h=24):
    """Feedback delay (in init steps): a forecast issued at init t with lead tau is
    only verifiable once t + tau has happened, so online methods (ACI, rolling) may
    only use residuals from inits t' <= t - ceil(tau / spacing)."""
    return [max(1, int(np.ceil(h / init_spacing_h))) for h in leads_h]


def conformal_quantile(scores, alpha, axis=0):
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = scores.shape[axis]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        shape = list(scores.shape)
        del shape[axis]
        return np.full(shape, np.inf, dtype=np.float32)
    return np.partition(scores, k - 1, axis=axis).take(k - 1, axis=axis)


# ----------------------------------------------------------------------------- methods
def gaussian_baseline(cal, nt, alpha):
    """Parametric baseline: per-lead, per-cell RMSE * z. Assumes Gaussian errors."""
    sigma = np.sqrt(np.mean(cal.astype(np.float64) ** 2, axis=0))
    q = (norm.ppf(1 - alpha / 2) * sigma).astype(np.float32)
    return np.broadcast_to(q[None], (nt,) + q.shape)


def split_pooled(cal, nt, alpha):
    """One quantile per lead, pooled over all cells and calibration times."""
    L = cal.shape[1]
    q = np.array([conformal_quantile(cal[:, l].ravel(), alpha) for l in range(L)], dtype=np.float32)
    return np.broadcast_to(q[None, :, None, None], (nt,) + cal.shape[1:])


def split_local(cal, nt, alpha):
    """One quantile per lead AND per grid cell (marginal conformal at each cell)."""
    q = conformal_quantile(cal, alpha, axis=0).astype(np.float32)
    return np.broadcast_to(q[None], (nt,) + q.shape)


def mondrian_season(cal, season_cal, season_test, alpha):
    """Local conformal calibrated separately within each season of the init date."""
    L, H, W = cal.shape[1:]
    qs = np.zeros((4, L, H, W), np.float32)
    for s in range(4):
        m = season_cal == s
        qs[s] = conformal_quantile(cal[m], alpha, axis=0)
    return qs[season_test]


def aci_local(cal, test, alpha, delays, gamma):
    """Adaptive Conformal Inference (Gibbs & Candes 2021), run independently at every
    cell and lead, with realistic delayed feedback. The level is clipped to the
    calibration range (interval never wider than the max calibration residual)."""
    Nc, L, H, W = cal.shape
    Nt = test.shape[0]
    srt = [np.ascontiguousarray(np.sort(cal[:, l], axis=0)) for l in range(L)]
    a = np.full((L, H, W), alpha, dtype=np.float64)
    q_out = np.empty((Nt, L, H, W), dtype=np.float32)
    for t in range(Nt):
        for l in range(L):
            d = delays[l]
            if t - d >= 0:
                miss = (test[t - d, l] > q_out[t - d, l]).astype(np.float64)
                a[l] += gamma * (alpha - miss)
            k = np.clip(np.ceil((Nc + 1) * (1.0 - a[l])).astype(np.int64), 1, Nc)
            q_out[t, l] = np.take_along_axis(srt[l], (k - 1)[None], axis=0)[0]
    return q_out


def rolling_local(cal, test, alpha, delays, window):
    """Rolling-window local conformal: calibrate each cell/lead on the most recent
    `window` VERIFIED residuals (calibration period + already-verified test period)."""
    pool = np.concatenate([cal, test], axis=0)
    Nc, Nt = cal.shape[0], test.shape[0]
    L, H, W = cal.shape[1:]
    out = np.empty((Nt, L, H, W), np.float32)
    for t in range(Nt):
        g = Nc + t
        for l in range(L):
            end = g - delays[l] + 1
            start = max(0, end - window)
            out[t, l] = conformal_quantile(pool[start:end, l], alpha, axis=0)
    return out


METHOD_NAMES = ["gaussian", "split_pooled", "split_local", "mondrian_season",
                "aci_0.005", "aci_0.02", "rolling_90", "rolling_365"]


def build_intervals(name, cal, test, alpha, season_cal, season_test, delays):
    nt = test.shape[0]
    if name == "gaussian":
        return gaussian_baseline(cal, nt, alpha)
    if name == "split_pooled":
        return split_pooled(cal, nt, alpha)
    if name == "split_local":
        return split_local(cal, nt, alpha)
    if name == "mondrian_season":
        return mondrian_season(cal, season_cal, season_test, alpha)
    if name.startswith("aci_"):
        return aci_local(cal, test, alpha, delays, float(name.split("_")[1]))
    if name.startswith("rolling_"):
        return rolling_local(cal, test, alpha, delays, int(name.split("_")[1]))
    raise ValueError(name)


# --------------------------------------------------------------------------- metrics
def evaluate(q, test, alpha, lat, season_test, year_test, delta=0.10):
    """Interval = yhat +/- q. Returns a dict of arrays (all per lead unless noted)."""
    Nt, L, H, W = test.shape
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()

    def wm(x):  # area-weighted mean over the last two axes (H, W)
        return (x.mean(-1) * w).sum(-1)

    cover = test <= q
    qpos = np.maximum(q, 0)
    out = {}
    out["coverage"] = wm(cover.mean(0))                                   # (L,)
    out["cov_map"] = cover.mean(0).astype(np.float32)                     # (L,H,W)
    out["cov_season"] = np.stack([wm(cover[season_test == s].mean(0)) for s in range(4)])
    years = np.unique(year_test)
    out["years"] = years
    out["cov_year"] = np.stack([wm(cover[year_test == y].mean(0)) for y in years])
    out["width"] = wm(2 * qpos.mean(0))
    out["score_day"] = wm(2 * qpos + (2 / alpha) * np.maximum(test - qpos, 0)).astype(np.float32)
    out["interval_score"] = out["score_day"].mean(0)
    # --- dependence diagnostics (area-weighted) ---
    c = (cover.mean(axis=3) * w).sum(axis=2)                       # (Nt, L)
    p = 1 - out["coverage"]
    ref = p * (1 - p) * (w ** 2).sum() / W
    out["spatial_cov"] = c.astype(np.float32)
    out["vif"] = c.var(axis=0, ddof=1) / ref
    out["n_eff"] = (W / (w ** 2).sum()) / out["vif"]
    out["bad_day_frac"] = (c < (1 - alpha - delta)).mean(0)
    miss = ~cover

    def persist(m):
        num = (m[1:] & m[:-1]).sum(axis=(0, 2, 3))
        den = np.maximum(m[:-1].sum(axis=(0, 2, 3)), 1)
        return num / den

    out["miss_persist"] = persist(miss)
    rng = np.random.default_rng(0)
    out["miss_persist_null"] = np.mean([persist(miss[rng.permutation(Nt)]) for _ in range(10)], axis=0)
    return out
