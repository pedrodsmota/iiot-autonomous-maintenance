#!/usr/bin/env bash
# Run the whole offline pipeline, in order (see prognostics/README.md).
# Steps 06 and 09 train many models and take the longest.
set -euo pipefail
cd "$(dirname "$0")"
python 01_sync_data.py
python 02_extract_features.py
python 03_pca_features.py
python 04_construct_health_labels.py
python 05_build_dataset.py
python 06_tune_hyperparameters.py
python 07_train_final_model.py
python 08_evaluate_baselines.py
for s in adxl piezo mpu; do python 09_loso_eval.py "$s"; done
python 10_choose_thresholds.py
python 11_loso_decision_zone_error.py
python plots/plot_rul_trace.py
