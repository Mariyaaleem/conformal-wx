# data/

Large files are **not** in git. They are hosted at: `(https://doi.org/10.5281/zenodo.23998807)`.

| file | what | size | needed for |
|---|---|---|---|
| `preds.npz` | calibration + test forecasts of the CNN (seed 0, epoch-3 checkpoint): `pred`, `true` (N,8,32,64), `init_time`, `split`, `leads_h`, `lat`, `lon`; 730 calibration + 1,086 test daily 00 UTC inits | 238,269,012 bytes (227 MiB) | steps 03-07 (reproduces every number in the paper) |
| `era5_64x32.npz` | ERA5 2 m temperature + Z500, 5.625 deg, 6-hourly, 1979-01-01 to 2021-12-31 (62,824 steps) | ~1 GB | steps 02 only (or regenerate with `01_download_era5.py`) |
| `forecaster_skill.json` | test RMSE of model / persistence / climatology (in git) | small | paper Table (skill) |

SHA-256 of `preds.npz`: `a7000c2e4dc07da6b4c3517662ce13798619511f5f997e818135dd9b71c543cc`
Check with `sha256sum data/preds.npz` (Linux/Colab) or `certutil -hashfile data\preds.npz SHA256` (Windows).`.

ERA5 (Hersbach et al., 2020; Copernicus Climate Change Service) accessed through the WeatherBench 2 archive
(Rasp et al., 2024) under the Copernicus licence; cite both if you reuse the data. `preds.npz` is derived from ERA5.
