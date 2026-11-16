# conformal-wx

Code, result tables and figure scripts for

> **Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence**
> (benchmark paper, arXiv: `[TODO: arXiv id]`)

We benchmark eight interval-construction procedures (Gaussian/RMSE, split conformal pooled and per-cell,
per-cell + season, ACI with two step sizes, rolling windows of 90 and 365 days) for gridded 2 m temperature
forecasts at eight lead times (6 h to 10 d), and report dependence diagnostics (spatial overdispersion index,
quantiles of daily spatial coverage, miss persistence against a shuffled-days null, coverage by season and year)
with block-bootstrap intervals. Two forecasters are used: a small CNN trained for this study (`data/preds.npz`) and
ECMWF IFS HRES from the WeatherBench 2 archive (`data/preds_hres.npz`). A further analysis asks how much of the
spatial overdispersion is a shared day-level error scale, and whether that scale can be predicted at issuance time.

**Release v1.1.0** adds the HRES forecaster (`08_build_hres_preds.py`, `run_all.sh hres`), the day-scale decomposition
(`decompose.py`), the day-scale predictability test (`feasible.py`) and the paired comparison (`paired_hres.py`).
Nothing in the v1.0.0 scripts (`00`-`07`, `conformal_lib.py`) was changed.

**Scope and caveats (please read).** ERA5 is a *reanalysis*, not observations. The CNN has 452,744 parameters and no skill
over climatology beyond about 5 days (see `reference/forecaster_skill.json`).
One variable (2 m temperature), one coarse resolution (5.625 deg, 32 x 64), two calibration years (2017-2018)
and three test years (2019-2021). The bootstrap resamples test days only and holds the calibration set and the
forecaster fixed, so intervals for the split methods understate total uncertainty. Comparisons are exploratory
and not corrected for multiplicity. The heterogeneity-aware overdispersion reference was adopted after seeing the
pooled-split dip at 12 h (paper, Limitations). Calibration/test are separated in time with no purge gap.
For the CNN we observed no clear coverage drift over the three test years; **for HRES per-cell split coverage drifts across the
test years at short leads** (0.886, 0.879 and 0.864 in 2019, 2020 and 2021 at 6 h; coverage-by-year columns of `results_hres/summary.csv`).

**HRES caveat.** `preds_hres.npz` depends on the WeatherBench 2 HRES archive (regridded from ECMWF's operational
archive), not on anything in this repository, and HRES is a moving target: ECMWF changed IFS cycles during 2017-2021,
with changes inside both periods (calibration window: cycles 43r3, operational from July 2017, and 45r1, 5 June 2018; test window:
46r1, 11 June 2019, 47r1, 30 June 2020, 47r2, 11 May 2021, and 47r3, 12 October 2021; dates as in the paper, Section 4.10), so the
forecast error distribution is not stationary across calibration and test. HRES is initialised from ECMWF's operational
analysis, not from ERA5, so short-lead errors also contain analysis-reanalysis differences. If the archive is changed
or re-uploaded the rebuilt file can change; `08_build_hres_preds.py --compare` checks a rebuilt file against an existing one.

**Day-scale analyses are diagnostics.** In `decompose.py` the oracle day scale uses the verifying truth (not a forecast
procedure, biased low), the split-half version is biased high, and the null floor shows what each pipeline returns when
there is no dependence. In `feasible.py` the day scale is predicted from information available at issuance time; it lowers
the VIF, but the interval score improves by at most 0.6 percent relative to per-cell split conformal (CNN 0.1 to 0.6 percent,
HRES 0.1 to 0.4 percent, as in the paper's Sections 4.9 and 4.11), and ACI(0.02) and Mondrian-season obtain better interval scores. For HRES, from 12 h to 3 d no method is best on both VIF and interval score.
See `results_feasible*/feasible.csv` and `results_hres/paired_dayscale_alpha0_1.csv` for the numbers.

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
07_check_reproduction.py    compare regenerated results with reference/ (CNN only)
08_build_hres_preds.py      build data/preds_hres.npz (ECMWF HRES, WeatherBench 2) from data/preds.npz   [new in v1.1.0]
09_forecaster_rmse.py       area-weighted test RMSE of the CNN and HRES forecasts per lead               [new in v1.1.0]
decompose.py                day-scale decomposition: oracle, null floor, split-half, VIF by distance      [new in v1.1.0]
feasible.py                 can the day-level error scale be predicted at issuance time?                  [new in v1.1.0]
paired_hres.py              paired block bootstrap: day-scale model vs the benchmark methods              [new in v1.1.0]
run_all.sh                  one command: reproduce | full | synthetic | dayscale | hres
reference/                  the CSV/JSON tables the paper's numbers come from (for step 07, CNN only)
data/README.md              where to get preds.npz, preds_hres.npz and the ERA5 slice (not stored in git)
results/alpha_<a>/*.npz     small per-method arrays (per-day coverage and score, maps) so that 05/06 can be re-run
                            without the 1 GB data
```
`decompose.py`, `feasible.py` and `paired_hres.py` import each other and `conformal_lib.py`, so run them from the
repository root (or keep the files in one folder).

## Reproduce the paper (CNN)

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

## Day-scale analysis of the CNN (v1.1.0)

```bash
./run_all.sh dayscale          # decompose.py and feasible.py on data/preds.npz -> results_decomp/, results_feasible/
```
`decompose.py` compares its per-cell split VIF with the paper's Table 3 (CNN, alpha = 0.1) and prints a warning if the
two differ by more than 0.05.

## HRES analysis (v1.1.0)

Needs `data/preds.npz` and either `data/preds_hres.npz` (download, see `data/README.md`) or internet access to rebuild it.

```bash
pip install -r requirements.txt -r requirements-hres.txt     # the second file only to rebuild preds_hres.npz
./run_all.sh hres
```
This runs, in order, and prints the wall-clock time of each step:

| step | command | output |
|---|---|---|
| build forecasts (skipped if the file exists) | `python 08_build_hres_preds.py` | `data/preds_hres.npz` |
| benchmark, all methods, alphas 0.1 0.2 0.05 | `python 03_run_benchmark.py --preds data/preds_hres.npz --out results_hres --alphas 0.1 0.2 0.05` | `results_hres/` |
| heterogeneity-aware VIF etc. | `python 06_extra_diagnostics.py --results results_hres --preds data/preds_hres.npz` | `results_hres/` |
| bootstrap tables | `python 05_bootstrap_tables.py --results results_hres` | `results_hres/` |
| figures | `python 04_make_figures.py --results results_hres --alpha 0.1` | `results_hres/figures/` |
| day-scale decomposition | `python decompose.py --npz data/preds_hres.npz --boot 1000 --skip-table3-check --out results_decomp_hres` | `results_decomp_hres/` |
| day-scale predictability | `python feasible.py --npz data/preds_hres.npz --boot 1000 --save-daily results_feasible_hres/daily.npz --out results_feasible_hres` | `results_feasible_hres/` |
| paired comparison | `python paired_hres.py --npz data/preds_hres.npz --daily results_feasible_hres/daily.npz --results results_hres --boot 1000` | `results_hres/paired_dayscale_alpha0_1.csv` |
| forecaster RMSE | `python 09_forecaster_rmse.py --preds data/preds.npz data/preds_hres.npz --out results_hres/forecaster_rmse.csv` | `results_hres/forecaster_rmse.csv` |

`07_check_reproduction.py` is **not** run for HRES: its reference tables are for the CNN only. The `--skip-table3-check`
flag of `decompose.py` is needed for the same reason (otherwise it prints a "Table 3 check ... WARNING", which is expected
and meaningless for HRES). `paired_hres.py` recomputes the VIF and interval score of every benchmark method and compares
them with `results_hres/vif_het_alpha0_1.csv` and `results_hres/summary.csv` before comparing; it stops if they differ
(`--skip-check` overrides). `BOOT=200 ./run_all.sh hres` uses fewer bootstrap replicates in `decompose.py`/`feasible.py`.

Optional robustness runs with longer bootstrap blocks (the paper reports the 60- and 90-day blocks with 200 replicates):
```bash
for B in 60 90; do
  python decompose.py --npz data/preds_hres.npz --boot 200 --block $B --skip-table3-check --out results_decomp_hres_b$B
  python feasible.py  --npz data/preds_hres.npz --boot 200 --block $B --out results_feasible_hres_b$B
done
```
The same commands with `data/preds.npz` (and without `--skip-table3-check`) give the CNN block-length checks.

To check a rebuilt HRES file against an existing one (needs internet; writes nothing):
`python 08_build_hres_preds.py --compare data/preds_hres.npz`.

## Which results come from which script

| result | file | script |
|---|---|---|
| CNN forecaster skill table (all three rows) | `data/forecaster_skill.json` (copy in `reference/`) | `02` |
| CNN and HRES area-weighted test RMSE per lead (forecaster rows of the skill table) | `results_hres/forecaster_rmse.csv` | `09_forecaster_rmse.py` |
| CNN persistence and climatology RMSE | `data/forecaster_skill.json` | `02` |
| main table (coverage, interval score, VIF^hom, 95% CI) | `results/table_main_ci.csv` | `05` |
| paired differences vs per-cell split | `results/table_paired_alpha0_1.csv` | `05` |
| block-length sensitivity (10/30/60 days) | `results/block_sensitivity.csv` | `05` |
| VIF (heterogeneity-aware), VIF vs VIF^hom appendix | `results/vif_het_alpha0_1.csv`, `vif` column of `results/summary.csv` | `06`, `03` |
| miss persistence, 5th percentile, worst day, season/year gaps | `results/summary.csv` | `03` |
| share of forecasts with low daily coverage (delta = 0.02/0.05/0.10) | `results/threshold_sweep.csv` | `06` |
| forecast bias by year | `results/bias_by_year.csv` (needs `data/preds.npz`) | `06` |
| other significance levels | `results/table_main_ci_alpha0_2.csv`, `..._alpha0_05.csv`, `results/table_paired_alpha0_2.csv`, `..._0_05.csv` | `05` |
| synthetic validation | `results/synthetic_check.csv` | `06 --iid_summary` |
| figures 1-5 | `results/figures/` | `04` |
| **the same benchmark tables for HRES** | `results_hres/` (same file names as above) | `03`, `06`, `05`, `04` on `data/preds_hres.npz` |
| day-scale decomposition (oracle, null floor, split-half, bootstrap intervals) | `results_decomp/decomp_vif.csv` (CNN), `results_decomp_hres/decomp_vif.csv` (HRES) | `decompose.py` |
| VIF by pair distance, miss correlation by distance | `results_decomp*/vif_by_distance.csv` | `decompose.py` |
| decomposition figure | `results_decomp*/decomposition.png` | `decompose.py` |
| day-scale predictability (out-of-sample R^2, VIF, coverage, width ratio, interval score, bootstrap) | `results_feasible/feasible.csv` (CNN), `results_feasible_hres/feasible.csv` (HRES) | `feasible.py` |
| daily series of the day-scale model (input of the paired comparison) | `results_feasible_hres/daily.npz` | `feasible.py --save-daily` |
| paired VIF and interval-score differences, day-scale model vs benchmark methods | `results_hres/paired_dayscale_alpha0_1.csv` | `paired_hres.py` |

Conventions: the bootstrap tables report the **homogeneous** VIF (as in the paper); the heterogeneity-aware version
has point estimates only. `vif_het_alpha<a>.csv` and `table_main_ci_alpha<a>.csv` use `_` instead of `.` in the alpha tag.
The VIF in `decompose.py`, `feasible.py` and `paired_hres.py` uses the population variance of the daily coverage; the
`vif_het` column of `06` uses the sample variance, larger by the factor n/(n-1) (about 0.1% for 1,086 days).

`results/table_main.csv/.tex` (from `04`) are **point-estimate summaries** (coverage, width, interval score, VIF^hom and,
when `06` has been run, the heterogeneity-aware VIF). They contain no bootstrap intervals. The paper's tables with 95%
block-bootstrap intervals are `results/table_main_ci*.csv`, produced by `05_bootstrap_tables.py`. If
`vif_het_alpha<a>.csv` is missing, `04` warns and Fig. 5 falls back to the homogeneous VIF (stated on the axis label).

## Method notes

* Splits by initialisation date: train 1979-2015, val 2016, calibration 2017-2018 (730 daily 00 UTC inits),
  test 2019-01-01 to 2021-12-21 (1,086 consecutive daily 00 UTC inits). The ERA5 store is named 1959-2022 but its time
  axis ends on 2021-12-31 18 UTC; inits whose 10-day target lies beyond that are excluded. (HRES uses the same inits, leads
  and ERA5 targets; the CNN is trained on 1979-2015, HRES involves no training here.)
* Online methods (ACI, rolling windows) use delayed feedback: a lead-tau forecast issued at init n may only use residuals
  from inits n' <= n - max(1, ceil(tau / 24 h)), i.e. delays 1, 1, 1, 2, 3, 5, 7, 10 days.
* Leads of 6 h and 12 h verify at 06 and 12 UTC, all others at 00 UTC (diurnal confound; hollow markers in the figures).
* Bootstrap (`05`): 95% moving-block percentile intervals over test days, blocks of 30 days, 1,000 replicates, seed 0;
  paired intervals use the same resampled days for both methods; calibration set and forecaster held fixed.
  `decompose.py`/`feasible.py`/`paired_hres.py` use the same moving-block scheme (`--block`, `--boot`, `--seed`).
* Day-scale model (`feasible.py`): a single global scale per day and lead, predicted by OLS on log scale from the season
  (two harmonics of the valid day of year), the log scale of the previous day's 24 h forecast (`fresh`) and the log scale of
  the same-lead forecast whose verification arrives at issuance (`stale`); calibration-day predictions are out-of-fold
  (fit on one calibration year, predict the other). Variants with fewer features are reported as ablations.

## Status of checks

| step | status |
|---|---|
| `01`, `02`, `03`, `04` | run on Google Colab with real data |
| `00`, `03`-`07` | run end-to-end on synthetic data (small case) while preparing v1.0.0 |
| `05`-`07` on real data vs `reference/` | `./run_all.sh reproduce` on Colab (CPU), 2026-11-16: `07` result **OK** (`summary.csv` 192/192 rows within 0.001; `vif_het` max difference 0.0050, the reference is rounded to 0.01; `table_main_ci*.csv` point and endpoint differences 0.0000) |
| `decompose.py`, `feasible.py`, `paired_hres.py` | run on the real CNN and HRES files; self-tests on synthetic data (`--synthetic structured/null`, `--synthetic coupled/null`) |
| `08_build_hres_preds.py` | `--compare data/preds_hres.npz`, 2026-11-16: the rebuilt file is identical to the released `preds_hres.npz` in every key. Recent xarray versions print a harmless `FutureWarning` about decoding `prediction_timedelta` |
| `./run_all.sh hres` | ran to completion on Colab (CPU), 2026-11-16, including the `paired_hres.py` self-check against `results_hres/` |

Runtimes on a Colab CPU runtime (Python 3.13, 2026-11-16; `./run_all.sh hres`, `[run_all]` lines): `03_run_benchmark.py` on HRES (8 methods, alphas 0.1/0.2/0.05) 410 s (`rolling_365` alone about 93 s per alpha on one machine, 129 s on this runtime), `06` 2 s, `05` 20 s, `04` 4 s, `decompose.py` (1,000 replicates) 28 s, `feasible.py` (1,000 replicates) 11 s, `paired_hres.py` (1,000 replicates, six methods) 196 s; about 11 minutes in total. `01`, `02` (GPU) and `08` were not re-timed for this release (`08` reads about 15,000 small chunks from Google Cloud Storage; the time depends on the connection).
`./run_all.sh dayscale|hres` print `[run_all] <seconds> s : <command>` after every step.

Synthetic self-tests of the new scripts (no data needed):
```bash
python decompose.py --synthetic structured --boot 20 --out /tmp/syn      # and: --synthetic null
python feasible.py  --synthetic coupled    --boot 20 --out /tmp/f1       # and: --synthetic null
```

## Pinning versions

`requirements.txt` and `requirements-hres.txt` are pinned to the versions used for the release runs (Colab CPU, Python 3.13.18). `requirements-train.txt` is not pinned. Print the versions in your own environment with:

```python
import sys, numpy, scipy, pandas, matplotlib, torch, xarray, zarr, gcsfs, dask
print(sys.version.split()[0])
for m in (numpy, scipy, pandas, matplotlib, torch, xarray, zarr, gcsfs, dask):
    print(f"{m.__name__}=={m.__version__}")
```

## Citation and licence

Code: MIT (see `LICENSE`). The Zenodo records, which also contain the data files derived from ERA5 and ECMWF HRES, are CC BY 4.0 (see `data/README.md`). Cite the paper and, if you use the released forecasts, the Zenodo archives
(`CITATION.cff`; this release, v1.1.0, with `preds_hres.npz`: https://doi.org/10.5281/zenodo.24262268; `preds.npz`: https://doi.org/10.5281/zenodo.23998807). Data: ERA5 (Hersbach et al., 2020; Copernicus Climate Change Service) and ECMWF IFS HRES, both via
WeatherBench 2 (Rasp et al., 2024); `preds.npz` and `preds_hres.npz` are derived from them, see `data/README.md`.
