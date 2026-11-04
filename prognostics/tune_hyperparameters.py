import numpy as np
import pandas as pd
from pathlib import Path
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import gc
import re
import os

DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

def get_base_trial_name(name):
    m = re.search(r'trial_(\d+)', str(name))
    return f"trial_{m.group(1)}.csv" if m else name

def trial_sort_key(t):
    m = re.search(r'trial_(\d+)', str(t))
    return int(m.group(1)) if m else 999

class SequenceDataset(Dataset):
    """
    Dynamically slices sequences to avoid massive RAM allocation and OS locking.
    Instead of building a giant overlapping NumPy array, we just keep the 2D
    array in memory and slice what we need per batch.
    """
    def __init__(self, X, y, time_steps):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.time_steps = time_steps

    def __len__(self):
        return len(self.X) - self.time_steps

    def __getitem__(self, i):
        return self.X[i:i + self.time_steps], self.y[i + self.time_steps]

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters, cnn_kernel, lstm_hidden, lstm_layers, dropout):
        super(CNN_BiLSTM, self).__init__()
        self.cnn  = nn.Conv1d(input_dim, cnn_filters, cnn_kernel, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        x = self.relu(self.cnn(x.transpose(1, 2)))
        x = self.pool(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze()

# ── Load data function per sensor ───────────────────────────────────────────
def load_sensor_data(sensor_name, feat_dir: Path, train_ratio=0.80, inner_val_ratio=0.80):
    print(f"\nLoading {sensor_name.upper()} features for tuning ...")
    
    sensor_files = {
        'adxl': 'adxl345_features.csv',
        'piezo': 'piezo_features.csv',
        'mpu': 'mpu9250_features.csv'
    }
    
    df = pd.read_csv(feat_dir / sensor_files[sensor_name])
    df_hi = pd.read_csv(feat_dir / f"{sensor_name}_training_dataset_with_targets.csv")

    df['trial_base']    = df['trial'].apply(get_base_trial_name)
    df_hi['trial_base'] = df_hi['trial'].apply(get_base_trial_name)
    hi_lookup = df_hi[['trial_base', 'window_idx', 'Health_Index_HI']]

    # Exclude trial_8 for ADXL, but include it for piezo and mpu
    if sensor_name == 'adxl':
        df = df[df['trial_base'] != 'trial_8.csv'].reset_index(drop=True)
        
    df = pd.merge(df, hi_lookup, on=['trial_base', 'window_idx'], how='inner')
    df['trial_num'] = df['trial_base'].apply(trial_sort_key)
    df = df.sort_values(['trial_num', 'window_idx']).reset_index(drop=True)

    META = {'trial', 'window_idx', 'time_start_sec', 'trial_base', 'trial_num', 'Health_Index_HI'}
    feature_cols = [c for c in df.columns if c not in META]

    X_raw = np.where(np.isfinite(df[feature_cols].values), df[feature_cols].values, 0.0)
    y     = df['Health_Index_HI'].values.astype(np.float32)

    # ── Block shuffle split matching dataset_builder ──
    # Note: For tuning, block size of 90 ensures safe splits for max look-back
    BLOCK_SIZE = 90
    n_blocks = len(X_raw) // BLOCK_SIZE
    block_idx = np.arange(n_blocks)

    rng = np.random.default_rng(seed=42)
    rng.shuffle(block_idx)

    # Outer split (80% of data for tuning)
    split_point = int(n_blocks * train_ratio)
    dev_blocks  = block_idx[:split_point]

    # Inner split (80% of dev for train, 20% of dev for val)
    inner_split  = int(len(dev_blocks) * inner_val_ratio)
    train_blocks = np.sort(dev_blocks[:inner_split])
    val_blocks   = np.sort(dev_blocks[inner_split:])

    def blocks_to_arrays(blocks, X, y, block_size):
        X_parts = [X[b*block_size:(b+1)*block_size] for b in blocks]
        y_parts = [y[b*block_size:(b+1)*block_size] for b in blocks]
        return np.concatenate(X_parts).astype(np.float32), np.concatenate(y_parts).astype(np.float32)

    X_tr_raw, y_tr = blocks_to_arrays(train_blocks, X_raw, y, BLOCK_SIZE)
    X_vl_raw, y_vl = blocks_to_arrays(val_blocks, X_raw, y, BLOCK_SIZE)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_vl = scaler.transform(X_vl_raw).astype(np.float32)

    print(f"  Input dim       : {X_tr.shape[1]}")
    print(f"  Inner train size: {len(X_tr)} windows")
    print(f"  Inner val size  : {len(X_vl)} windows")
    
    return {'X_tr': X_tr, 'y_tr': y_tr, 'X_vl': X_vl, 'y_vl': y_vl, 'input_dim': X_tr.shape[1]}

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial, sensor_data):
    time_steps   = trial.suggest_categorical("time_steps",   [30, 60, 90])
    cnn_filters  = trial.suggest_categorical("cnn_filters",  [16, 32, 64])
    cnn_kernel   = trial.suggest_categorical("cnn_kernel",   [3, 5])
    lstm_hidden  = trial.suggest_categorical("lstm_hidden",  [32, 64, 128])
    lstm_layers  = trial.suggest_int("lstm_layers", 1, 2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate= trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Replaced create_sequences (which explodes RAM) with dynamic Datasets
    train_dataset = SequenceDataset(sensor_data['X_tr'], sensor_data['y_tr'], time_steps)
    val_dataset   = SequenceDataset(sensor_data['X_vl'], sensor_data['y_vl'], time_steps)
    
    if len(train_dataset) <= 0 or len(val_dataset) <= 0:
        return float('inf')

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False)

    model = CNN_BiLSTM(sensor_data['input_dim'], cnn_filters, cnn_kernel,
                       lstm_hidden, lstm_layers, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

    for _ in range(15):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    model.eval()
    val_loss, n = 0.0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            val_loss += criterion(model(bx), by).item() * bx.size(0)
            n += bx.size(0)

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_loss / n

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    feat_dir = DATA_DIR / "features"
    out_dir  = Path(__file__).resolve().parent

    sensors = ['piezo', 'mpu']

    for sensor in sensors:
        sensor_data = load_sensor_data(sensor, feat_dir)

        print(f"\nStarting Bayesian Optimisation for {sensor.upper()} — 30 trials ...")
        study = optuna.create_study(direction="minimize", study_name=f"CNN_BiLSTM_PHM_{sensor}")
        
        study.optimize(lambda trial: objective(trial, sensor_data), n_trials=30)

        print(f"\nBest hyperparameters found for {sensor.upper()}:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

        out_path = out_dir / f"best_hyperparameters_{sensor}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(study.best_trial.params, f)
        print(f"\nSaved {sensor.upper()} params to {out_path}\n")
        print("─" * 80)
