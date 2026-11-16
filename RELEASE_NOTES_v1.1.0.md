# conformal-wx v1.1.0

Adds a second forecaster (ECMWF IFS HRES) and the day-scale analyses that accompany the paper. The v1.0.0 pipeline
(`00`-`07`, `conformal_lib.py`) and its results are unchanged.

## Added
- `08_build_hres_preds.py`: builds `data/preds_hres.npz` from `data/preds.npz` and the WeatherBench 2 HRES archive
  (`--compare` checks a rebuilt file against an existing one).
- `run_all.sh hres`: build HRES forecasts if missing, benchmark into `results_hres/`, day-scale analysis, paired comparison.
  It does not run `07_check_reproduction.py` (its reference tables are CNN-only).
- `run_all.sh dayscale`: day-scale analysis of the CNN forecasts. Each step prints its wall-clock time.
- `decompose.py`: day-scale decomposition of the spatial overdispersion index (oracle, null floor, split-half, VIF by
  pair distance, block bootstrap). New `--skip-table3-check` (the Table 3 comparison applies to the CNN file only).
- `feasible.py`: day-scale predictability test at issuance time, interval score, block bootstrap, `--save-daily`.
- `paired_hres.py`: paired block-bootstrap comparison of the day-scale model with the benchmark methods, with a
  self-check against the saved benchmark results.
- `requirements-hres.txt` (xarray, zarr, gcsfs); README sections on the HRES analysis, commands, runtimes and a
  results-to-script table; HRES entries in `data/README.md`; `CITATION.cff` updated to v1.1.0.

## Changed (no numerical change; verified by identical CSV outputs on the synthetic self-tests)
- `decompose.py`: default calibration label is `calib` (was `cal` with a fallback); clearer messages and docstrings.
- `feasible.py`, `paired_hres.py`: clearer docstrings, comments and error messages; `paired_hres.py` gains
  `--cal-label/--test-label` and checks that `--results` exists.

## Notes and limitations
- `preds_hres.npz` depends on the WeatherBench 2 HRES archive, which is a regridded copy of ECMWF's operational forecasts;
  ECMWF changed IFS cycles during 2017-2021 (43r3 and 45r1 in the calibration window; 46r1, 47r1, 47r2, 47r3 in the test window), so HRES errors are not stationary across calibration and test; per-cell split coverage drifts across the test years at short leads.
- The day-scale model lowers the VIF but improves the interval score by at most 0.6 percent (CNN 0.1 to 0.6 percent, HRES 0.1 to 0.4 percent); ACI(0.02) and Mondrian-season have
  better interval scores. For HRES, from 12 h to 3 d no method is best on both VIF and interval score.
- VIF in the new scripts uses the population variance; `vif_het` uses the sample variance (factor n/(n-1), about 0.1%).
- Data files are not in git: `preds.npz` is at https://doi.org/10.5281/zenodo.23998807; `preds_hres.npz` is archived with this release. See `data/README.md` for links and SHA-256 checksums.
