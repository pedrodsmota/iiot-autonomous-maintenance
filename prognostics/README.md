# prognostics/ - offline pipeline

From the raw recordings to the trained models, their evaluation, and the thresholds
of the prescriptive rule (dissertation, Sections 5.4, 5.6, 6.3, and 7.2).
The scripts are numbered in order of execution; each one starts with a docstring
saying what it reads, what it writes, and which part of the dissertation it implements.

| Step | Script | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `01_sync_data.py` | `data/raw/`, `data/metadata/sessions.csv` | `data/synchronized/` | minutes |
| 2 | `02_extract_features.py` | `data/synchronized/` | `data/features/<sensor>_features.csv` | minutes |
| 3 | `03_pca_features.py` | features | `data/features/<sensor>_features_pca.csv` | seconds |
| 4 | `04_construct_health_labels.py` | PCA features | health stages, HI labels, `<sensor>_label_info.json`, `session_timeline.json` | seconds |
| 5 | `05_build_dataset.py` | features + labels | `data/processed/<sensor>_dataset.pkl`, `models/scaler_<sensor>.pkl` | seconds |
| 6 | `06_tune_hyperparameters.py` | dataset | `models/best_hyperparameters_<sensor>.pkl`, `results/optuna_trials_<sensor>.csv` | the longest step |
| 7 | `07_train_final_model.py` | dataset, scaler, hyperparameters | `models/model_<sensor>.pt`, `results/metrics.json` | ~10 min |
| 8 | `08_evaluate_baselines.py` | dataset, scaler, metrics | `results/baselines.csv` | minutes |
| 9 | `09_loso_eval.py <sensor>` | dataset, hyperparameters | `results/loso_<sensor>.json`, `results/loso_<sensor>_pred.csv` | ~20 min per sensor |
| 10 | `10_choose_thresholds.py` | dataset, scaler, model | `results/thresholds.json` | seconds |
| 11 | `11_loso_decision_zone_error.py` | LOSO predictions | (printed) | seconds |
| - | `plots/plot_rul_trace.py` | dataset, scaler, model | `results/figures/rul_trace_<sensor>.*` | seconds |

**Disk space and shortcuts.** Step 1 writes about 36 GB of CSV to `data/synchronized/`.
To skip steps 1 and 2, unzip `features.zip` from the dataset into `data/`, which gives
`data/features/`, and start at step 3. To skip step 6, keep the hyperparameters in
`models/`.

Times measured on a MacBook with Apple MPS, from a clean clone: step 7 took 10 min for the
three sensors and step 9 took 17 min for the piezoelectric sensor. With the versions in
`environment/requirements.txt`, steps 3-5, 7, 8, 10, and 11 and step 9 for that sensor gave
the same numbers as those in `results/`.

Run everything with `bash run_all.sh`. Shared code: `pipeline_common.py` (paths,
parameters, window keys, time line, HI <-> RUL, contiguous sequences, seeds) and
`model_common.py` (the CNN-BiLSTM and the sequence dataset).

## Key design choices

- **Same preprocessing as in operation.** Resample to 2800 Hz first, then a causal
  high-pass filter that starts from the steady state; the online scripts do exactly
  the same, and `online/check_online_features.py` checks it window by window.
- **Real time axis.** Windows are placed on the operating time line (seconds since
  the start of the collection), so missing or discarded windows do not compress it.
- **Labels.** GMM health stages (with a time feature and a cumulative maximum) give
  the first predicting time (FPT); after it the HI follows `HI = 2 - 2^t'`, which
  can be inverted into a RUL (Equation 5.4). The labels therefore encode elapsed
  time as much as condition - see Section 5.4.2 of the dissertation before reusing them.
- **No leakage through sequences.** Sequences only contain contiguous windows; each
  test block gives one sequence made of its own windows; Optuna validates on a
  subset of the training blocks, never on the test set.
- **Optimistic test error.** Test blocks come from the same life cycle as the
  training blocks; `09_loso_eval.py` holds out whole sessions and shows errors
  2-3x higher. Keep this in mind when comparing with other work.

## Notes

- The piezoelectric sensor was replaced after session 14; the per-session
  calibration factor comes from `sessions.csv` (see `dataset/README.md`).
- Seeds are fixed, but training on CPU, CUDA, and Apple MPS gives slightly
  different weights; expect small differences from the values in the dissertation.
