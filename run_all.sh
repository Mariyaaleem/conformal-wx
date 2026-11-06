#!/usr/bin/env bash
# Usage:
#   ./run_all.sh reproduce    # regenerate every table/figure from data/preds.npz   (no GPU, no download)
#   ./run_all.sh full         # also download ERA5 and retrain the forecaster       (GPU recommended)
#   ./run_all.sh synthetic    # i.i.d. synthetic sanity check used in the paper's appendix (minutes)
set -euo pipefail
MODE="${1:-reproduce}"
case "$MODE" in
  synthetic)
    python 00_synthetic_demo.py --iid --out data/iid.npz
    python 03_run_benchmark.py --preds data/iid.npz --out results_iid \
        --methods gaussian split_pooled split_local aci_0.005
    python 06_extra_diagnostics.py --results results_iid --alphas 0.1 \
        --methods gaussian split_pooled split_local aci_0.005 --iid_summary results_iid/summary.csv
    ;;
  full|reproduce)
    if [ "$MODE" = "full" ]; then
      python 01_download_era5.py
      python 02_train_forecaster.py --epochs 8
    fi
    [ -f data/preds.npz ] || { echo "data/preds.npz missing (see data/README.md)"; exit 1; }
    python 03_run_benchmark.py --preds data/preds.npz --out results --alphas 0.1 0.2 0.05
    python 06_extra_diagnostics.py --results results --preds data/preds.npz
    python 05_bootstrap_tables.py --results results
    python 04_make_figures.py --results results --alpha 0.1
    # point estimates must match the shipped reference tables (valid for the released preds.npz only)
    python 07_check_reproduction.py --results results --reference reference
    ;;
  *) echo "usage: $0 {reproduce|full|synthetic}"; exit 2;;
esac
