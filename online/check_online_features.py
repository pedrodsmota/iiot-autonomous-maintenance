"""
Compare the features written by an online script (--dump-features) with the
training features of the same session (dissertation, Section 5.5.1).

  python3 check_online_features.py dump.csv data/features/adxl345_features.csv trial_12.csv

The online scripts do not write the windows without signal (variance of the
first axis below ACTIVITY_THRESHOLD), and the training pipeline discards them
too, so they are removed here from the training side before aligning the two
row by row; otherwise each window without signal would misalign all the
following ones.

Median relative differences of the order of 1e-7 (float32 precision) indicate
that the online preprocessing is identical to training. Skewness and kurtosis
may show larger maxima when they are close to zero.
"""
import sys

import numpy as np
import pandas as pd

from online_common import ACTIVITY_THRESHOLD

online_csv, train_csv, trial = sys.argv[1:4]
tr = pd.read_csv(train_csv)
tr = tr[tr["trial"] == trial].drop(columns=["trial", "window_idx", "time_start_sec"])
var_col = [c for c in tr.columns if c.endswith("_var")][0]
dead = (tr[var_col] <= ACTIVITY_THRESHOLD).values
idx_train = np.flatnonzero(~dead)                 # index of the window in the training file
cols, tr = tr.columns, tr.values[~dead]
on = np.loadtxt(online_csv, delimiter=",", ndmin=2)
if on.shape[1] == tr.shape[1] + 1:
    on = on[:, 1:]
n = min(len(tr), len(on))
rel = np.abs(on[:n] - tr[:n]) / (np.abs(tr[:n]) + 1e-9)
print(f"windows: training {len(tr) + dead.sum()} ({dead.sum()} without signal, positions "
      f"{np.flatnonzero(dead).tolist()[:10]}), online {len(on)}, compared {n}")
print(f"median relative error {np.median(rel):.2e}")
for j in np.argsort(-rel.max(axis=0))[:3]:
    print(f"  {cols[j]:<32} max {rel[:, j].max():.2e}  median {np.median(rel[:, j]):.2e}")
bad = np.flatnonzero(np.median(rel, axis=1) > 1e-3)
if len(bad):
    print(f"  {len(bad)} windows with a median error > 1e-3, training windows "
          f"{idx_train[bad][:10].tolist()}{' ...' if len(bad) > 10 else ''}")
else:
    print("  no window with a median error > 1e-3")
