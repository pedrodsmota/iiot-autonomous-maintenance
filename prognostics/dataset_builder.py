import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import re
import os

DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

def get_base_trial_name(name):
    m = re.search(r'trial_(\d+)', str(name))
    return f"trial_{m.group(1)}.csv" if m else name

def trial_sort_key(t):
    m = re.search(r'trial_(\d+)', str(t))
    return int(m.group(1)) if m else 999

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

if __name__ == '__main__':
    feat_dir = DATA_DIR / "features"
    out_dir  = Path(__file__).resolve().parent
    out_dir.mkdir(exist_ok=True)

    TIME_STEPS  = 60
    TRAIN_RATIO = 0.80
    SEED        = 42

    sensors = {
        'adxl':  feat_dir / "adxl345_features.csv",
        'piezo': feat_dir / "piezo_features.csv",
        'mpu':   feat_dir / "mpu9250_features.csv",
    }
    META_COLS = {'trial', 'window_idx', 'time_start_sec',
                 'trial_base', 'trial_num', 'Health_Index_HI'}

    for sensor_name, csv_path in sensors.items():
        print(f"\n{'─'*60}")
        print(f"  Building dataset for: {sensor_name.upper()}")
        print(f"{'─'*60}")

        # LOAD INDIVIDUAL TARGETS FOR EACH SENSOR
        target_file = feat_dir / f"{sensor_name}_training_dataset_with_targets.csv"
        print(f"Loading HI targets from {target_file.name} ...")
        df_targets = pd.read_csv(target_file)
        df_targets['trial_base'] = df_targets['trial'].apply(get_base_trial_name)
        hi_lookup = df_targets[['trial_base', 'window_idx', 'Health_Index_HI']].copy()

        df = pd.read_csv(csv_path)
        df['trial_base'] = df['trial'].apply(get_base_trial_name)
        
        if sensor_name == 'adxl':
            df = df[df['trial_base'] != 'trial_8.csv'].reset_index(drop=True)

        df = pd.merge(df, hi_lookup, on=['trial_base', 'window_idx'], how='inner')
        df['trial_num'] = df['trial_base'].apply(trial_sort_key)
        df = df.sort_values(['trial_num', 'window_idx']).reset_index(drop=True)

        feature_cols = [c for c in df.columns if c not in META_COLS]

        X_raw = df[feature_cols].values.astype(np.float64)
        y     = df['Health_Index_HI'].values.astype(np.float32)

        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)

        n_blocks = len(X_raw) // TIME_STEPS
        block_idx = np.arange(n_blocks)

        rng = np.random.default_rng(seed=SEED)
        rng.shuffle(block_idx)

        split_point  = int(n_blocks * TRAIN_RATIO)
        train_blocks = np.sort(block_idx[:split_point])
        test_blocks  = np.sort(block_idx[split_point:])

        def blocks_to_arrays(blocks, X, y, block_size):
            X_parts = [X[b*block_size:(b+1)*block_size] for b in blocks]
            y_parts = [y[b*block_size:(b+1)*block_size] for b in blocks]
            return np.concatenate(X_parts).astype(np.float32), \
                   np.concatenate(y_parts).astype(np.float32)

        X_train_raw, y_train = blocks_to_arrays(train_blocks, X_raw, y, TIME_STEPS)
        X_test_raw,  y_test  = blocks_to_arrays(test_blocks,  X_raw, y, TIME_STEPS)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

        with open(out_dir / f"scaler_{sensor_name}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, TIME_STEPS)
        X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test,  TIME_STEPS)

        dataset = {
            'X_train':      X_train_seq,
            'y_train':      y_train_seq,
            'X_test':       X_test_seq,
            'y_test':       y_test_seq,
            'feature_cols': feature_cols,
            'input_dim':    X_train_seq.shape[2],
            'time_steps':   TIME_STEPS,
            'n_blocks':     n_blocks,
            'train_blocks': train_blocks,
            'test_blocks':  test_blocks,
        }

        out_file = out_dir / f"{sensor_name}_dataset.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(dataset, f)

        print(f"  Feature columns   : {len(feature_cols)}")
        print(f"  Total blocks      : {n_blocks}  ({len(X_raw)} windows)")
        print(f"  Train blocks      : {len(train_blocks)}  →  {len(X_train_seq)} sequences  "
              f"(HI range [{y_train.min():.3f} → {y_train.max():.3f}])")
        print(f"  Test  blocks      : {len(test_blocks)}   →  {len(X_test_seq)} sequences  "
              f"(HI range [{y_test.min():.3f} → {y_test.max():.3f}])")

    print("\nAll three sensor datasets built successfully.")
