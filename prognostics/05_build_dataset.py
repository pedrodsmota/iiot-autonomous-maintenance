"""
Step 5 - Build the dataset of each sensor: block split, scaler, and the start
indices of the sequences (dissertation, Section 5.4.3, Table 5.3).

Reads   data/features/<sensor>_features.csv
        data/features/<sensor>_training_dataset_with_targets.csv
Writes  data/processed/<sensor>_dataset.pkl
        models/scaler_<sensor>.pkl

Split
  * The windows, ordered by operating time, are divided into consecutive blocks
    of 60 windows, randomly assigned (seed 42) to training (80 %) and test (20 %);
    20 % of the training blocks are set aside as the validation set of Optuna,
    so the test set is used neither in the tuning nor in the training.
  * A sequence only contains contiguous windows (same session, no window
    discarded in between):
      - training: every sequence of 60 contiguous windows within training
        blocks, including those that span adjacent training blocks;
      - test: one sequence per test block, formed by its own 60 windows, if
        they are contiguous (blocks interrupted by a gap are excluded).
  * Target: the HI of the last window of the sequence (the current instant).
  * The .pkl stores the raw features and the start indices, not the sequences.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline_common import (EXCLUDED_SESSIONS, FEAT_DIR, FEATURE_FILES, MODEL_DIR, PROCESSED_DIR,
                             SENSORS, TIME_STEPS, add_window_keys, block_mask,
                             block_split, contiguity, load_label_info,
                             sequence_starts, test_starts)

RAW_META = {"trial", "window_idx", "time_start_sec"}


def build(sensor):
    raw = pd.read_csv(FEAT_DIR / FEATURE_FILES[sensor])
    feature_cols = [c for c in raw.columns if c not in RAW_META]
    raw = add_window_keys(raw)
    raw = raw[~raw["session"].isin(EXCLUDED_SESSIONS.get(sensor, set()))]

    tg = pd.read_csv(FEAT_DIR / f"{sensor}_training_dataset_with_targets.csv",
                     usecols=["session", "win", "t", "Health_Index_HI", "RUL_s"])
    df = raw.merge(tg, on=["session", "win"], how="inner", validate="one_to_one")
    df = df.sort_values("t").reset_index(drop=True)

    X_raw = df[feature_cols].values.astype(np.float64)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0).astype(np.float32)
    y = df["Health_Index_HI"].values.astype(np.float32)
    session = df["session"].values
    win = df["win"].values
    cont = contiguity(session, win)
    n = len(df)

    split = block_split(n)
    m_train = block_mask(split["train"], n)
    m_inner = block_mask(split["inner_train"], n)
    m_val = block_mask(split["inner_val"], n)

    starts = {
        "train": sequence_starts(m_train, cont),
        "inner_train": sequence_starts(m_inner, cont),
        "inner_val": sequence_starts(m_val, cont),
        "test": test_starts(split["test"], cont, n),
    }

    scaler = StandardScaler().fit(X_raw[m_train])
    with open(MODEL_DIR / f"scaler_{sensor}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    dataset = {
        "sensor": sensor, "feature_cols": feature_cols, "time_steps": TIME_STEPS,
        "target": "HI of the last window of the sequence",
        "X_raw": X_raw, "y": y, "rul_s": df["RUL_s"].values.astype(np.float32),
        "t": df["t"].values, "session": session, "win": win,
        "split": split, "starts": starts, "label_info": load_label_info(sensor),
    }
    with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    excluded = len(split["test"]) - len(starts["test"])
    print(f"[{sensor.upper()}] {n} windows | {split['n_blocks']} blocks "
          f"(training {len(split['train'])}, test {len(split['test'])}) | "
          f"sequences: training {len(starts['train'])}, inner training "
          f"{len(starts['inner_train'])}, validation {len(starts['inner_val'])}, "
          f"test {len(starts['test'])} ({excluded} test blocks excluded because of gaps)")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for s in SENSORS:
        build(s)
