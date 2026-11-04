# results/

| File | Content |
|---|---|
| `article_table4.csv` | Table 4 of the article, as published |
| `figures/prediction_<sensor>_hi_only.pdf` | Predicted and reference HI on the test sequences (Figures 4-6) |

## Reproducing the values

The final evaluation used the deployed models, without retraining. With `prognostics/evaluate_article_models.py`, on the datasets built for the article:

| Sensor | RMSE / MAE in the article | RMSE / MAE re-evaluated |
|---|---|---|
| ADXL345 | 0.0334 / 0.0225 | 0.0334 / 0.0225 |
| Piezoelectric | 0.0426 / 0.0312 | 0.0426 / 0.0312 |
| MPU9250 | 0.0644 / 0.0376 | 0.0644 / 0.0376 |

**Not preserved:** the script that produced the baseline rows of Table 4, and the one that drew the figures. `prognostics/08_evaluate_baselines.py`, on the `main` branch, implements the same three baselines with the later evaluation.
