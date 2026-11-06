# conformal-wx

Code, result tables and figure scripts for

> **Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence**
> (benchmark paper, arXiv: `[TODO: arXiv id]`)

We benchmark eight interval-construction procedures (Gaussian/RMSE, split conformal pooled and per-cell,
per-cell + season, ACI with two step sizes, rolling windows of 90 and 365 days) for gridded 2 m temperature
forecasts at eight lead times (6 h to 10 d), and report dependence diagnostics (spatial overdispersion index,
quantiles of daily spatial coverage, miss persistence against a shuffled-days null, coverage by season and year)
with block-bootstrap intervals.

**Scope and caveats (please read).** ERA5 is a *reanalysis*, not observations. The forecaster is a small CNN
(452,744 parameters) with no skill over climatology beyond about 5 days (see `reference/forecaster_skill.json`).
One variable (2 m temperature), one coarse resolution (5.625 deg, 32 x 64), two calibration years (2017-2018)
and three test years (2019-2021). The bootstrap resamples test days only and holds the calibration set and the
forecaster fixed, so intervals for the split methods understate total uncertainty. Comparisons are exploratory
and not corrected for multiplicity. The heterogeneity-aware overdispersion reference was adopted after seeing the
pooled-split dip at 12 h (paper, Limitations). Calibration/test are separated in time with no purge gap; we observed
no clear coverage drift over the three test years.

## Repository layout

```
conformal_lib.py            conformal methods + metrics (numpy/scipy only)
00_synthetic_demo.py        synthetic forecasts (i.i.d. control, or correlated/seasonal/drifting)
01_download_era5.py         ERA5 slice from the WeatherBench 2 bucket -> data/era5_64x32.npz
02_train_forecaster.py      train the CNN, write data/preds.npz and data/forecaster_skill.json
03_run_benchmark.py         all methods x alphas -> results/summary.csv, results/alpha_<a>/<method>.npz
04_make_figures.py          figures 1-5 and results/table_main.{csv,tex}
05_bootstrap_tables.py      block-bootstrap tables (main, other alphas, paired, block-length sensitivity)
06_extra_diagnostics.py     heterogeneity-aware VIF, threshold sweep, bias by year, synthetic-check summary
07_check_reproduction.py    compare regenerated results with reference/
run_all.sh                  one command: reproduce | full | synthetic
reference/                  the CSV/JSON tables the paper's numbers come from (for step 07)
data/README.md              where to get preds.npz and the ERA5 slice (not stored in git)
results/alpha_<a>/*.npz     small per-method arrays (per-day coverage and score, maps) so that 05/06 can be re-run
                            without the 1 GB data
```

## Reproduce the paper

Needs `data/preds.npz` (see `data/README.md`). No GPU, no download of ERA5.

```bash
pip install -r requirements.txt
# put preds.npz in data/ (Zenodo/Drive link in data/README.md), then:
./run_all.sh reproduce
```

This runs, in order: `03` (all methods, alphas 0.1 0.2 0.05 in **one** run, because `summary.csv` is overwritten
on each run) -> `06` -> `05` -> `04` -> `07`. The order matters: `04` reads the heterogeneity-aware VIF written by `06`.
`./run_all.sh full` additionally downloads ERA5 and retrains the forecaster (needs a GPU and the packages in
`requirements-train.txt`). `./run_all.sh synthetic` runs the i.i.d. sanity check described in the paper's appendix.

On Colab:
```python
!git clone https://github.com/Mariyaaleem/conformal-wx.git && cd conformal-wx && pip -q install -r requirements.txt
# download data/preds.npz here, then
!cd conformal-wx && ./run_all.sh reproduce
```

Retraining (`full`) is **not** bit-reproducible: seeds are fixed (seed 0) but GPU kernels are not forced to be
deterministic, so a retrained forecaster can differ slightly. The released `preds.npz` is the canonical input for every
number in the paper.

## Which output produces which part of the paper

| paper item | file |
|---|---|
| forecaster skill table | `data/forecaster_skill.json` (written by `02`; copy in `reference/`) |
| main table (coverage, interval score, VIF^hom, 95% CI) | `results/table_main_ci.csv` (`05`) |
| paired differences vs per-cell split | `results/table_paired_alpha0_1.csv` (`05`) |
| block-length sensitivity (10/30/60 days) | `results/block_sensitivity.csv` (`05`) |
| VIF (heterogeneity-aware), VIF vs VIF^hom appendix | `results/vif_het_alpha0_1.csv` (`06`), `vif` column of `results/summary.csv` |
| miss persistence, 5th percentile, worst day, season/year gaps | `results/summary.csv` (`03`) |
| share of forecasts with low daily coverage (delta = 0.02/0.05/0.10) | `results/threshold_sweep.csv` (`06`) |
| forecast bias by year | `results/bias_by_year.csv` (`06`, needs `data/preds.npz`) |
| other significance levels | `results/table_main_ci_alpha0_2.csv`, `..._alpha0_05.csv`, `results/table_paired_alpha0_2.csv`, `..._0_05.csv` |
| synthetic validation | `./run_all.sh synthetic` writes `results_iid/synthetic_check.csv` (not committed) |
| figures 1-5 | `results/figures/` (`04`) |

Conventions: the bootstrap tables report the **homogeneous** VIF (as in the paper); the heterogeneity-aware version
has point estimates only. `vif_het_alpha<a>.csv` and `table_main_ci_alpha<a>.csv` use `_` instead of `.` in the alpha tag.

`results/table_main.csv/.tex` (from `04`) are **point-estimate summaries** (coverage, width, interval score, VIF^hom and,
when `06` has been run, the heterogeneity-aware VIF). They contain no bootstrap intervals. The paper's tables with 95%
block-bootstrap intervals are `results/table_main_ci*.csv`, produced by `05_bootstrap_tables.py`. If
`vif_het_alpha<a>.csv` is missing, `04` warns and Fig. 5 falls back to the homogeneous VIF (stated on the axis label).

## Method notes

* Splits by initialisation date: train 1979-2015, val 2016, calibration 2017-2018 (730 daily 00 UTC inits),
  test 2019-01-01 to 2021-12-21 (1,086 consecutive daily 00 UTC inits). The ERA5 store is named 1959-2022 but its time
  axis ends on 2021-12-31 18 UTC; inits whose 10-day target lies beyond that are excluded.
* Online methods (ACI, rolling windows) use delayed feedback: a lead-tau forecast issued at init n may only use residuals
  from inits n' <= n - max(1, ceil(tau / 24 h)), i.e. delays 1, 1, 1, 2, 3, 5, 7, 10 days.
* Leads of 6 h and 12 h verify at 06 and 12 UTC, all others at 00 UTC (diurnal confound; hollow markers in the figures).
* Bootstrap (`05`): 95% moving-block percentile intervals over test days, blocks of 30 days, 1,000 replicates, seed 0;
  paired intervals use the same resampled days for both methods; calibration set and forecaster held fixed.

## Status of checks

| step | status |
|---|---|
| `01`, `02`, `03`, `04` | run on Google Colab with real data |
| `00`, `03`-`07` | run end-to-end on synthetic data (small case) while preparing this release |
| `05`-`07` on real data vs `reference/` | `07_check_reproduction.py` reports OK: `summary.csv` 192/192 rows within 1e-3 of the reference; point estimates and bootstrap endpoints of the three CI tables identical to the reference tables; `vif_het` within the 0.01 rounding of the reference |

Runtimes: not measured. `05`/`06` only read the small per-method arrays in `results/alpha_*/` and finish in minutes on CPU.

## Versions

`requirements.txt` pins the analysis packages used to produce the paper's tables (Python 3.13.x; numpy 2.4.6,
scipy 1.16.3, pandas 2.2.3, matplotlib 3.10.0). `requirements-train.txt` (steps 01-02: torch, xarray, zarr, gcsfs, dask)
is not pinned.

## Citation and licence

Code: MIT (see `LICENSE`). Cite the paper and, if you use the released forecasts, the Zenodo archive
(`CITATION.cff`). Data: ERA5 (Hersbach et al., 2020; Copernicus Climate Change Service) via WeatherBench 2
(Rasp et al., 2024); `preds.npz` is derived from ERA5, see `data/README.md`.
