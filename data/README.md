# data/

Large files are **not** in git. `preds.npz` is archived at https://doi.org/10.5281/zenodo.23998807 (v1.0.0). `preds_hres.npz` is
archived with the code release v1.1.0, https://doi.org/10.5281/zenodo.24262268.

| file | what | size | needed for |
|---|---|---|---|
| `preds.npz` | calibration + test forecasts of the CNN (seed 0, epoch-3 checkpoint): `pred`, `true` (N,8,32,64), `init_time`, `split`, `leads_h`, `lat`, `lon`; 730 calibration + 1,086 test daily 00 UTC inits | 238,269,012 bytes (about 238 MB) | steps 03-07, `decompose.py`, `feasible.py` (reproduces every CNN number in the paper) |
| `preds_hres.npz` | same keys, inits, leads, grid and `true` (ERA5) as `preds.npz`, but `pred` is the ECMWF IFS HRES 2 m temperature forecast from the WeatherBench 2 archive | 193,196,462 bytes (about 193 MB) | HRES analysis (`./run_all.sh hres`); **not** for `07_check_reproduction.py` |
| `era5_64x32.npz` | ERA5 2 m temperature + Z500, 5.625 deg, 6-hourly, 1979-01-01 to 2021-12-31 (62,824 steps) | 1,037,351,872 bytes (about 1.04 GB) | steps 02 only (or regenerate with `01_download_era5.py`) |
| `forecaster_skill.json` | test RMSE of model / persistence / climatology (in git) | small | paper Table (skill) |

`split` is `calib` (calibration rows) or `test`; `leads_h` = 6, 12, 24, 48, 72, 120, 168, 240; `pred`/`true` are 2 m temperature in K.

## Rebuilding `preds_hres.npz`

```bash
pip install -r requirements-hres.txt
python 08_build_hres_preds.py                                  # data/preds.npz -> data/preds_hres.npz
python 08_build_hres_preds.py --compare data/preds_hres.npz     # check a rebuilt file against an existing one
```
It reads `gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr` (anonymous access,
variable `2m_temperature`) at the inits and leads stored in `preds.npz`; `true`, `init_time`, `split`, `leads_h`, `lat`
and `lon` are copied unchanged. The result depends on that archive: it is a regridded copy of ECMWF's operational
forecasts, ECMWF changed its IFS model cycle several times during 2017-2021, and WeatherBench 2 may re-upload or
change it. The released `preds_hres.npz` is the canonical input for every HRES number in the paper; a rebuilt
file may differ if the archive has changed.

SHA-256:
```
a7000c2e4dc07da6b4c3517662ce13798619511f5f997e818135dd9b71c543cc  preds.npz
30d41a432a4ee90e1e02e7af05c5f6de4bbf8f671459ef0a6184314014417d17  preds_hres.npz
87dc7b86313239f43055676c17c67e55b6c38ebd8bbca463985a1e406722c7c6  era5_64x32.npz
```
Check with `sha256sum data/preds.npz data/preds_hres.npz`.

## Licence and attribution of the data

`preds.npz` and `preds_hres.npz` are derived from ERA5 and (for `preds_hres.npz`) from ECMWF IFS HRES forecasts, both obtained
from the WeatherBench 2 archive (Rasp et al., 2024). Please keep these attributions if you reuse or redistribute the files.

* **ECMWF IFS HRES forecasts:** Copyright (c) European Centre for Medium-Range Weather Forecasts (ECMWF), 2017-2021 (years of the
  forecasts used). Source: www.ecmwf.int. ECMWF open data is published under the Creative Commons Attribution 4.0 International
  licence (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. `preds_hres.npz` is a modified version: 2 m temperature only,
  regridded to 5.625 deg by WeatherBench 2, subset to the 1,816 initialisations and 8 lead times of this study.
* **ERA5:** Hersbach et al. (2020), generated using / contains modified Copernicus Climate Change Service information; accessed
  through WeatherBench 2. `true` in both files, and the CNN training data, come from ERA5.
* **WeatherBench 2:** Rasp et al. (2024), arXiv:2308.15560 (CC BY 4.0).
* **This repository's own products** (`preds.npz` CNN forecasts, code, results): the code is MIT-licensed (`LICENSE`); the files in
  the Zenodo records (which contain the code zip and the data files) are released under CC BY 4.0, as for v1.0; the source terms above
  (ECMWF, Copernicus/ERA5, WeatherBench 2) continue to apply to the derived data, so cite them when you reuse `preds.npz` or
  `preds_hres.npz`.

The wording above follows the attribution statements that ECMWF and WeatherBench 2 use for these datasets; check the current
terms on the WeatherBench 2 data guide (https://weatherbench2.readthedocs.io/en/latest/data-guide.html) and on the Copernicus
Climate Data Store before you publish the Zenodo record.
