"""
Step 11 - RUL error in the decision zone (real RUL between 2.5 and 7 h) with
whole sessions held out (dissertation, Section 7.2).

Reproduces the values discussed in Section 7.2: MAE of 1.0 / 1.4 / 2.5 h and
95th percentile of the optimistic error of 1.4 / 1.2 / 4.4 h (ADXL345 /
piezoelectric / MPU9250), against the margins of 52 / 20 / 62 min derived from
the block split (step 10).

Reads   results/loso_<sensor>_pred.csv  (step 9)
        data/processed/<sensor>_dataset.pkl  (definition of the labels)
"""
import pickle

import numpy as np
import pandas as pd

from pipeline_common import PROCESSED_DIR, RESULTS_DIR, SENSORS, hi_to_rul

ZONE_H = (2.5, 7.0)   # same as step 10

for sensor in SENSORS:
    with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "rb") as f:
        info = pickle.load(f)["label_info"]
    d = pd.read_csv(RESULTS_DIR / f"loso_{sensor}_pred.csv")
    rul_true = (info["t_eol_s"] - d["t_s"].values) / 3600
    rul_pred = hi_to_rul(d["hi_pred"].values, info["L_s"]) / 3600
    zone = (rul_true >= ZONE_H[0]) & (rul_true <= ZONE_H[1]) & (d["hi_true"].values < 1.0)
    err = rul_pred[zone] - rul_true[zone]          # > 0: optimistic (late order)
    print(f"{sensor:<6} n={zone.sum():>5}  bias {err.mean():+.2f} h  MAE {np.abs(err).mean():.2f} h  "
          f"RMSE {np.sqrt(np.mean(err ** 2)):.2f} h  p90 {np.quantile(err, 0.9):+.2f} h  "
          f"p95 {np.quantile(err, 0.95):+.2f} h")
