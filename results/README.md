# results/

Outputs of the offline pipeline, as reported in the dissertation.

| File | Content | Dissertation |
|---|---|---|
| `metrics.json` | HI RMSE/MAE on the test sequences and RUL error after the FPT, per sensor | Table 6.3 |
| `baselines.csv` | CNN-BiLSTM vs. random forest, ridge, and amplitude regression | Table 6.4 |
| `loso_<sensor>.json` | Errors with whole sessions held out (overall, before/after the FPT, per session) | Table 6.5 |
| `loso_<sensor>_pred.csv` | Out-of-sample predictions with whole sessions held out | Figure 6.6, §7.2 |
| `thresholds.json` | Calibration of the thresholds (q80, q90, q95 options; q95 was used) | Table 5.6 |
| `optuna_trials_<sensor>.csv` | All Optuna trials | Table 5.4 |
| `label_info_<sensor>.json` | Definition of the health labels: t_FPT, t_EoL, and L (seconds) | §5.4.2 |
| `figures/` | Prediction, LOSO, and RUL trace figures | Figures 6.3-6.6 |
