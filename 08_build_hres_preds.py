#!/usr/bin/env python3
"""
08_build_hres_preds.py -- build data/preds_hres.npz (ECMWF IFS HRES forecasts) from data/preds.npz.

The output has the same keys, shapes and split labels as preds.npz:
    pred      HRES 2 m temperature forecast (K), (N, L, H, W) float32   <-- replaced
    true      ERA5 2 m temperature target, copied unchanged from preds.npz
    init_time, split, leads_h, lat, lon                                  copied unchanged
so every script that reads preds.npz (03, 06, decompose.py, feasible.py, paired_hres.py) also runs on it.
(Do NOT run 07_check_reproduction.py on it: the reference tables are for the CNN only.)

Source: WeatherBench 2 (Rasp et al., 2024), public Google Cloud bucket, read anonymously:
    gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr
Its 2m_temperature variable has dimensions (time = initialisation time, prediction_timedelta, longitude,
latitude) in that archive; they are looked up by NAME and transposed to (time, lead, lat, lon).
The initialisation times and leads are those stored in preds.npz.

Needs: numpy, pandas, xarray, zarr, gcsfs (requirements-hres.txt) and internet access to Google Cloud Storage.
Reading ~15k small chunks takes a few minutes. The result depends on the archive: if WeatherBench 2 changes or
re-uploads it, the file can change; use --compare to check a rebuilt file against an existing one.

Usage:
  python 08_build_hres_preds.py                                   # data/preds.npz -> data/preds_hres.npz
  python 08_build_hres_preds.py --compare data/preds_hres.npz     # rebuild and compare with an existing file;
                                                                  # writes nothing
  python 08_build_hres_preds.py --zarr /local/copy.zarr           # a local copy of the archive instead of gs://
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ZARR_DEFAULT = "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr"
VARIABLE = "2m_temperature"
OUT_DIMS = ("time", "prediction_timedelta", "latitude", "longitude")   # (time, lead, lat, lon)
KEYS = ("pred", "true", "init_time", "split", "leads_h", "lat", "lon")


def locate(zarr_times, zarr_leads, init_time, leads_h):
    """Positions of the requested inits and leads inside the archive's time and prediction_timedelta axes."""
    lh = np.asarray(leads_h)
    if not np.all(lh == np.round(lh)):
        sys.exit(f"leads_h must be whole hours, got {lh}")
    want_t = pd.DatetimeIndex(pd.to_datetime(np.asarray(init_time).astype(str)))
    arch_t = pd.DatetimeIndex(zarr_times)
    if not arch_t.is_unique:
        sys.exit("the archive's time axis has duplicate entries")
    ti = arch_t.get_indexer(want_t)
    if (ti < 0).any():
        bad = want_t[ti < 0]
        sys.exit(f"{len(bad)} requested init times are not in the archive (first: {bad[0]}); "
                 f"archive covers {arch_t.min()} .. {arch_t.max()}")
    want_l = pd.TimedeltaIndex(pd.to_timedelta(lh.astype(np.int64), unit="h"))
    arch_l = pd.TimedeltaIndex(zarr_leads)
    li = arch_l.get_indexer(want_l)
    if (li < 0).any():
        sys.exit(f"leads {lh[li < 0].tolist()} h are not in the archive's prediction_timedelta axis")
    return ti, li


def read_hres(ds, ti, li, batch=100):
    """Read (len(ti), len(li), H, W) from the archive, `batch` initialisations at a time."""
    da = ds[VARIABLE]
    if set(da.dims) != set(OUT_DIMS):
        sys.exit(f"unexpected dimensions {da.dims} for {VARIABLE}; expected {OUT_DIMS} in some order")
    parts = []
    for s in range(0, len(ti), batch):
        sel = da.isel(time=ti[s:s + batch], prediction_timedelta=li)   # outer (orthogonal) indexing
        parts.append(np.asarray(sel.transpose(*OUT_DIMS).values, dtype=np.float32))
        print(f"  read inits {s}..{min(s + batch, len(ti)) - 1} of {len(ti)}", flush=True)
    return np.concatenate(parts, axis=0)


def build(ds, src):
    """Return the dict of arrays for preds_hres.npz. `ds` is the opened archive, `src` the loaded preds.npz."""
    lat, lon = src["lat"], src["lon"]
    for name, mine in (("latitude", lat), ("longitude", lon)):
        theirs = np.asarray(ds[name].values)
        if theirs.shape != mine.shape or not np.allclose(theirs, mine, atol=1e-3):
            sys.exit(f"{name} of the archive does not equal {name} in preds.npz "
                     f"(archive: {theirs[:3]}..., preds.npz: {mine[:3]}...). Not silently reordering; check the grid.")
    ti, li = locate(ds["time"].values, ds["prediction_timedelta"].values, src["init_time"], src["leads_h"])
    pred = read_hres(ds, ti, li)
    expected = src["true"].shape
    if pred.shape != expected:
        sys.exit(f"HRES array shape {pred.shape} != shape of preds.npz['true'] {expected}")
    if not np.isfinite(pred).all():
        sys.exit(f"{int((~np.isfinite(pred)).sum())} non-finite HRES values; refusing to write the file")
    out = {k: src[k] for k in KEYS}
    out["pred"] = pred
    return out


def sanity_report(out):
    """Plain, unweighted mean |HRES - ERA5| per lead, only to catch unit/offset mistakes (not the paper's RMSE)."""
    err = np.abs(out["pred"].astype(np.float64) - out["true"].astype(np.float64)).mean(axis=(0, 2, 3))
    print("sanity check, unweighted mean |HRES - ERA5| in K per lead (NOT the paper's area-weighted RMSE):")
    for h, e in zip(out["leads_h"], err):
        print(f"  {int(h):4d} h  {e:6.2f}")
    if err.max() > 10:
        print("WARNING: errors above 10 K suggest a unit, longitude or time-alignment problem; do not use the file.")


def compare(new, path):
    old = np.load(path, allow_pickle=False)
    ok = True
    for k in KEYS:
        if k not in old.files:
            print(f"  {k}: missing in {path}")
            ok = False
        elif old[k].shape != new[k].shape:
            print(f"  {k}: shape {old[k].shape} vs {new[k].shape}")
            ok = False
        elif k in ("pred", "true", "lat", "lon"):
            diff = float(np.max(np.abs(old[k].astype(np.float64) - new[k].astype(np.float64))))
            same = bool(np.array_equal(old[k], new[k]))
            print(f"  {k}: {'identical' if same else f'max abs difference {diff:.3g}'}")
            ok &= same
        else:
            same = bool(np.array_equal(old[k], new[k]))
            print(f"  {k}: {'identical' if same else 'DIFFERENT'}")
            ok &= same
    print("RESULT:", f"rebuilt file equals {path}" if ok else f"rebuilt file DIFFERS from {path}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="data/preds.npz", help="CNN forecast file that defines inits, leads, grid, targets")
    ap.add_argument("--out", default="data/preds_hres.npz")
    ap.add_argument("--zarr", default=ZARR_DEFAULT, help="gs:// URL (read anonymously) or local path of the HRES archive")
    ap.add_argument("--compare", metavar="EXISTING_NPZ", help="rebuild and compare with this file; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    a = ap.parse_args()

    if not os.path.exists(a.preds):
        sys.exit(f"{a.preds} not found (see data/README.md)")
    if not a.compare and os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists; use --force to overwrite or --compare {a.out} to check it")
    src = np.load(a.preds, allow_pickle=False)
    missing = [k for k in KEYS if k not in src.files]
    if missing:
        sys.exit(f"{a.preds} lacks keys {missing}")
    src = {k: src[k] for k in KEYS}

    import xarray as xr   # imported late so that --help works without the extra packages
    kw = {"storage_options": {"token": "anon"}} if a.zarr.startswith("gs://") else {}
    print("opening", a.zarr)
    ds = xr.open_zarr(a.zarr, chunks=None, **kw)

    out = build(ds, src)
    sanity_report(out)
    if a.compare:
        sys.exit(0 if compare(out, a.compare) else 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, {k: (v.shape, str(v.dtype)) for k, v in out.items()})


if __name__ == "__main__":
    main()
