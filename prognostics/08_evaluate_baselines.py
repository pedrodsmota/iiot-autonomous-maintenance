"""
Step 8 - Simpler regressors trained and evaluated on the same sequences as the
CNN-BiLSTM models (dissertation, Section 6.3, Table 6.4).

Reads   data/processed/<sensor>_dataset.pkl, models/scaler_<sensor>.pkl,
        results/metrics.json (to put the CNN-BiLSTM in the same table)
Writes  results/baselines.csv

Baselines
  * random forest (100 trees, at least 5 samples per leaf) and ridge regression,
    fed with the mean of the features over the sequence and the features of its
    last window;
  * linear regression on the mean RMS of each axis over the sequence, which is
    equivalent to a threshold on the vibration amplitude.
"""

import csv
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from pipeline_common import MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, SENSORS


def summaries(Xs, starts, T):
    cs = np.vstack([np.zeros((1, Xs.shape[1])), np.cumsum(Xs.astype(np.float64), axis=0)])
    means = (cs[starts + T] - cs[starts]) / T
    return np.hstack([means, Xs[starts + T - 1]])


def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return (float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
            float(np.mean(np.abs(y_pred - y_true))))


if __name__ == "__main__":
    metrics_path = RESULTS_DIR / "metrics.json"
    dl = json.load(open(metrics_path)) if metrics_path.exists() else {}
    rows = []
    for sensor in SENSORS:
        with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "rb") as f:
            ds = pickle.load(f)
        with open(MODEL_DIR / f"scaler_{sensor}.pkl", "rb") as f:
            scaler = pickle.load(f)
        T = ds["time_steps"]
        Xs = scaler.transform(ds["X_raw"]).astype(np.float32)
        tr, te = ds["starts"]["train"], ds["starts"]["test"]
        F_tr, F_te = summaries(Xs, tr, T), summaries(Xs, te, T)
        y_tr, y_te = ds["y"][tr + T - 1], ds["y"][te + T - 1]
        rms = [i for i, c in enumerate(ds["feature_cols"]) if c.endswith("_rms")]

        if sensor in dl:
            m = dl[sensor]
            rows.append([sensor, "CNN-BiLSTM", m["rmse"], m["mae"]])
        models = {
            "Random forest": (RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=SEED), F_tr, F_te),
            "Ridge regression": (Ridge(alpha=1.0), F_tr, F_te),
            "RMS-based linear regression": (LinearRegression(), F_tr[:, rms], F_te[:, rms]),
        }
        for name, (reg, a, b) in models.items():
            reg.fit(a, y_tr)
            rows.append([sensor, name, *evaluate(y_te, reg.predict(b))])
        for r in rows[-(len(models) + (sensor in dl)):]:
            print(f"{r[0]:<6} {r[1]:<30} RMSE {r[2]:.4f}  MAE {r[3]:.4f}")

    out = RESULTS_DIR / "baselines.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "model", "rmse", "mae"])
        w.writerows(rows)
    print(f"\nSaved to {out}")
