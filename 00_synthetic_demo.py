"""
00_synthetic_demo.py
Creates a SYNTHETIC preds.npz so you can test the whole pipeline (steps 3 and 4) in
minutes, without any data download or training.

  --iid     exchangeable, spatially independent errors  -> sanity check: every method
            should give ~(1-alpha) coverage, VIF ~ 1, miss persistence ~ alpha.
  (default) spatially + temporally correlated, seasonal, drifting errors -> mimics the
            non-exchangeable situation the paper studies.
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter


def make(n_cal, n_test, iid, seed=0, H=32, W=64):
    rng = np.random.default_rng(seed)
    leads_h = np.array([6, 12, 24, 48, 72, 120, 168, 240])
    L = len(leads_h)
    N = n_cal + n_test
    sc = 0.5 + 0.35 * np.sqrt(leads_h / 24.0)
    lat = np.linspace(-87.1875, 87.1875, H)
    lon = np.arange(W) * 360.0 / W
    dates = np.datetime64("2017-01-01") + np.arange(N)
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    phi = 0.0 if iid else 0.7
    sig = 0 if iid else 3.0

    def smooth(z):
        return z if sig == 0 else gaussian_filter(z, sigma=sig, mode=["nearest", "wrap"])

    norm_const = np.std(np.stack([smooth(rng.standard_normal((H, W))) for _ in range(40)]))
    err = np.zeros((N, L, H, W), np.float32)
    z = np.zeros((L, H, W))
    for t in range(N):
        for l in range(L):
            e = smooth(rng.standard_normal((H, W))) / norm_const
            z[l] = phi * z[l] + np.sqrt(1 - phi ** 2) * e
        scale = np.ones((H, W))
        if not iid:
            seas = np.cos(2 * np.pi * (doy[t] - 15) / 365.0)
            scale = scale * (1 + 0.5 * seas * np.sign(lat)[:, None] * (np.abs(lat)[:, None] / 90.0))
            if t >= n_cal:
                scale = scale * (1 + 0.25 * (t - n_cal) / n_test)  # drift after calibration
        err[t] = (z * sc[:, None, None] * scale[None]).astype(np.float32)
    pred = (280 + 10 * np.cos(np.deg2rad(lat))[None, None, :, None] * np.ones((N, L, H, W))).astype(np.float32)
    true = pred + err
    split = np.array(["calib"] * n_cal + ["test"] * n_test)
    return dict(pred=pred, true=true, init_time=dates.astype(str), split=split,
                leads_h=leads_h, lat=lat, lon=lon)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--n_cal", type=int, default=365)
    ap.add_argument("--n_test", type=int, default=730)
    ap.add_argument("--out", default="data/synthetic_preds.npz")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **make(a.n_cal, a.n_test, a.iid))
    print("saved", a.out)
