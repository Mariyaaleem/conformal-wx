"""
01_download_era5.py
Downloads a small slice of ERA5 from the public WeatherBench 2 bucket (64x32 grid, 6-hourly):
2m temperature + 500 hPa geopotential, 1979 onward  ->  data/era5_64x32.npz  (~1 GB)
NOTE: the store is named 1959-2022 but its time axis ends on 2021-12-31 18 UTC, so the
default --end simply returns everything up to that date (62,824 steps from 1979-01-01).

No Google account needed (anonymous access). Run it on Colab/Kaggle if your connection is slow.
Source store (listed in the WeatherBench 2 docs):
  gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr
"""
import argparse
import os
import numpy as np
import xarray as xr

PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--start", default="1979-01-01")
    ap.add_argument("--end", default="2022-12-31T18:00")
    ap.add_argument("--out", default="data/era5_64x32.npz")
    a = ap.parse_args()

    ds = xr.open_zarr(a.path, storage_options={"token": "anon"})
    print(ds)  # check variable / dimension names here if anything below fails
    t2m = ds["2m_temperature"].sel(time=slice(a.start, a.end))
    z500 = ds["geopotential"].sel(level=500, time=slice(a.start, a.end))
    t2m = t2m.transpose("time", "latitude", "longitude")
    z500 = z500.transpose("time", "latitude", "longitude")
    years = np.unique(t2m.time.dt.year.values)
    T_parts, Z_parts, time_parts = [], [], []
    for y in years:  # load year by year so progress is visible and memory stays low
        sl = slice(f"{y}-01-01", f"{y}-12-31T18:00")
        T_parts.append(t2m.sel(time=sl).values.astype(np.float32))
        Z_parts.append(z500.sel(time=sl).values.astype(np.float32))
        time_parts.append(t2m.sel(time=sl).time.values)
        print("loaded", y, flush=True)
    T = np.concatenate(T_parts)
    Z = np.concatenate(Z_parts)
    time = np.concatenate(time_parts).astype("datetime64[h]")
    assert np.all(np.diff(time).astype(int) == 6), "expected a regular 6-hourly series"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, t2m=T, z500=Z, time=time.astype(str),
             lat=t2m.latitude.values, lon=t2m.longitude.values)
    print("saved", a.out, T.shape)


if __name__ == "__main__":
    main()
