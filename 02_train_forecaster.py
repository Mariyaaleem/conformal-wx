"""
02_train_forecaster.py
Trains a small periodic-padded residual CNN that predicts 2m temperature at 8 lead times
(6h ... 10 days) from the current + previous 6-hourly fields, then writes calibration/test
forecasts for the conformal benchmark.

  python 02_train_forecaster.py --data data/era5_64x32.npz --out data/preds.npz --epochs 8

Splits (by initialisation date, 00 UTC for calib/test):
  train 1979-2015 | val 2016 | calibration 2017-2018 | test 2019-01-01 .. 2021-12-21
(the ERA5 slice ends 2021-12-31, and inits whose 10-day target lies beyond it are dropped,
so the test set is 1,086 consecutive daily inits; calibration is 730 inits)
The conformal methods are model-agnostic, so the forecaster only needs to be reasonable.
A GPU (Colab/Kaggle T4) takes roughly 10-30 min for 8 epochs; CPU works but is slow
(use --train_stride 4 --epochs 4).
Seeds are set but GPU kernels are not forced to be deterministic: retraining can give slightly
different forecasts. The released data/preds.npz is the canonical input for the paper's numbers.
"""
import argparse
import copy
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS = [1, 2, 4, 8, 12, 20, 28, 40]  # lead times in 6-hour steps -> 6h,12h,1d,2d,3d,5d,7d,10d


class CConv(nn.Module):
    """3x3 conv, periodic in longitude, replicate-padded in latitude."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3)

    def forward(self, x):
        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.c1, self.c2 = CConv(c, c), CConv(c, c)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(F.gelu(self.n(x)))))


class Net(nn.Module):
    def __init__(self, cin, cout, c=64, nblocks=6):
        super().__init__()
        self.stem = CConv(cin, c)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(nblocks)])
        self.n = nn.GroupNorm(8, c)
        self.head = CConv(c, cout)

    def forward(self, x):
        return self.head(F.gelu(self.n(self.blocks(self.stem(x)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/era5_64x32.npz")
    ap.add_argument("--out", default="data/preds.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train_stride", type=int, default=2, help="use every k-th 6-hourly init for training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)

    d = np.load(a.data)
    t2m, z500, lat, lon = d["t2m"], d["z500"], d["lat"], d["lon"]
    tm = d["time"].astype("datetime64[h]")
    T, H, W = t2m.shape
    day = tm.astype("datetime64[D]")
    hour = (tm - day.astype("datetime64[h]")).astype(int)
    year = tm.astype("datetime64[Y]").astype(int) + 1970
    doy = (day - day.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)

    ar = np.arange(T)
    valid = (ar >= 1) & (ar + max(STEPS) < T)
    train_idx = np.where(valid & (year <= 2015))[0][:: a.train_stride]
    val_idx = np.where(valid & (year == 2016) & (hour == 0))[0]
    cal_idx = np.where(valid & np.isin(year, [2017, 2018]) & (hour == 0))[0]
    test_idx = np.where(valid & (year >= 2019) & (hour == 0))[0]
    print("train/val/cal/test inits:", len(train_idx), len(val_idx), len(cal_idx), len(test_idx))

    ntr = int(np.searchsorted(year, 2016))  # all times before 2016 define the normalisation / climatology
    tmean, tstd = t2m[:ntr].mean(0), t2m[:ntr].std(0) + 1e-6
    zmean, zstd = z500[:ntr].mean(0), z500[:ntr].std(0) + 1e-6
    Fn = np.stack([(t2m - tmean) / tstd, (z500 - zmean) / zstd], 1).astype(np.float32)  # (T,2,H,W)
    dstd = (Fn[1:ntr:8] - Fn[0:ntr - 1:8]).std(axis=(0, 2, 3)).astype(np.float32)        # tendency scales
    sc = np.array([np.std(t2m[train_idx[::8] + s] - t2m[train_idx[::8]]) for s in STEPS], np.float32)
    print("per-lead target scale (K):", np.round(sc, 2))

    Fn_t = torch.from_numpy(Fn).to(dev)
    Tk = torch.from_numpy(t2m).to(dev)
    dstd_t = torch.from_numpy(dstd).to(dev)[None, :, None, None]
    sc_t = torch.from_numpy(sc).to(dev)
    sin_t = torch.from_numpy(np.sin(2 * np.pi * doy / 365.25).astype(np.float32)).to(dev)
    cos_t = torch.from_numpy(np.cos(2 * np.pi * doy / 365.25).astype(np.float32)).to(dev)
    coslat = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32)).to(dev)[None, None, :, None]

    def make_x(idx):
        B = idx.shape[0]
        f0 = Fn_t[idx]
        dd = (f0 - Fn_t[idx - 1]) / dstd_t
        s = sin_t[idx][:, None, None, None].expand(B, 1, H, W)
        c = cos_t[idx][:, None, None, None].expand(B, 1, H, W)
        return torch.cat([f0, dd, s, c, coslat.expand(B, 1, H, W)], 1)  # 7 channels

    def make_y(idx):
        t0 = Tk[idx]
        return torch.stack([(Tk[idx + s] - t0) / sc_t[l] for l, s in enumerate(STEPS)], 1)

    model = Net(7, len(STEPS)).to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    tr = torch.from_numpy(train_idx).to(dev)
    va = torch.from_numpy(val_idx).to(dev)
    spe = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe, pct_start=0.1)
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        tot = 0.0
        for i in range(spe):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            loss = F.mse_loss(model(make_x(idx)), make_y(idx))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = np.mean([F.mse_loss(model(make_x(va[j:j + 128])), make_y(va[j:j + 128])).item()
                          for j in range(0, len(va), 128)])
        print(f"epoch {ep + 1}/{a.epochs}  train {tot / spe:.4f}  val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    def predict(idx_np):
        P, Y = [], []
        idx_all = torch.from_numpy(idx_np).to(dev)
        with torch.no_grad():
            for j in range(0, len(idx_all), 128):
                idx = idx_all[j:j + 128]
                out = model(make_x(idx))
                P.append((Tk[idx][:, None] + out * sc_t[None, :, None, None]).cpu().numpy())
                Y.append(torch.stack([Tk[idx + s] for s in STEPS], 1).cpu().numpy())
        return np.concatenate(P).astype(np.float32), np.concatenate(Y).astype(np.float32)

    Pc, Yc = predict(cal_idx)
    Pt, Yt = predict(test_idx)

    # ---- context for the paper: is the forecaster competent? (lat-weighted RMSE, test period)
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()
    rm = lambda err: np.sqrt((np.mean(err ** 2, axis=-1) * w).sum(-1).mean(0))  # (L,)
    key_tr = doy[:ntr] * 4 + hour[:ntr] // 6
    order = np.argsort(key_tr, kind="stable")
    ks, starts = np.unique(key_tr[order], return_index=True)
    sums = np.add.reduceat(t2m[:ntr][order].astype(np.float64), starts, axis=0)
    cnts = np.diff(np.append(starts, len(order)))
    clim = np.full((366 * 4, H, W), np.nan)
    clim[ks] = sums / cnts[:, None, None]
    clim_pred = np.stack([clim[doy[test_idx + s] * 4 + hour[test_idx + s] // 6] for s in STEPS], 1)
    pers = np.repeat(t2m[test_idx][:, None], len(STEPS), 1)
    skill = {"leads_h": [6 * s for s in STEPS],
             "rmse_model": rm(Pt - Yt).tolist(),
             "rmse_persistence": rm(pers - Yt).tolist(),
             "rmse_climatology": rm(clim_pred - Yt).tolist()}
    print(json.dumps(skill, indent=1))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(os.path.join(os.path.dirname(a.out), "forecaster_skill.json"), "w") as f:
        json.dump(skill, f, indent=1)
    np.savez(a.out,
             pred=np.concatenate([Pc, Pt]), true=np.concatenate([Yc, Yt]),
             init_time=np.concatenate([day[cal_idx], day[test_idx]]).astype(str),
             split=np.array(["calib"] * len(cal_idx) + ["test"] * len(test_idx)),
             leads_h=np.array([6 * s for s in STEPS]), lat=lat, lon=lon)
    print("saved", a.out)


if __name__ == "__main__":
    main()
