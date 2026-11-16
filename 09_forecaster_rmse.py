#!/usr/bin/env python3
"""
09_forecaster_rmse.py -- area-weighted test RMSE per lead time of the forecasts in one or more preds files.

This is the RMSE of the forecaster rows in the paper's skill table (Table 1) for the CNN (data/preds.npz) and for ECMWF HRES
(data/preds_hres.npz):  RMSE(l) = sqrt( mean over test days of  sum_cells w_i (pred - true)^2 ),  w_i proportional to cos(latitude),
sum w_i = 1, over the rows whose split key equals --test-label (2019-01-01 to 2021-12-21, 1,086 inits).
Persistence and climatology RMSE of the CNN come from 02_train_forecaster.py (data/forecaster_skill.json); they are not recomputed here.

Usage (from the repository root):
  python 09_forecaster_rmse.py --preds data/preds.npz data/preds_hres.npz --out results_hres/forecaster_rmse.csv
Output: a table printed to the screen (2 decimals, K) and a csv with one column per file (4 decimals).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def wrmse(path, test_label):
    d = np.load(path, allow_pickle=False)
    pred, true, split = d["pred"], d["true"], d["split"]
    c = np.cos(np.deg2rad(d["lat"]))
    w = np.repeat(c[:, None], pred.shape[-1], axis=1).ravel()
    w = w / w.sum()
    t = split == test_label
    if not t.any():
        sys.exit(f"{path}: no rows with split == {test_label!r}; labels in the file: {list(np.unique(split))}")
    e2 = ((pred[t] - true[t]) ** 2).reshape(t.sum(), pred.shape[1], -1)
    return d["leads_h"], np.sqrt((e2 * w).sum(-1).mean(0)), int(t.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True, help="one or more forecast files (keys pred, true, split, leads_h, lat)")
    ap.add_argument("--test-label", default="test")
    ap.add_argument("--out", default=None, help="csv path (optional)")
    a = ap.parse_args()
    cols, leads = {}, None
    for p in a.preds:
        lh, r, n = wrmse(p, a.test_label)
        if leads is not None and not np.array_equal(leads, lh):
            sys.exit("the files have different lead times")
        leads = lh
        cols[os.path.splitext(os.path.basename(p))[0]] = r
        print(f"{p}: {n} test inits")
    df = pd.DataFrame({"lead_h": leads.astype(int), **cols})
    print("\narea-weighted test RMSE (K)")
    print(df.round(2).to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.round(4).to_csv(a.out, index=False)
        print("saved", a.out)


if __name__ == "__main__":
    main()
