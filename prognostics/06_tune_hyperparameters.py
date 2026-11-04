"""
Step 6 - Hyperparameter optimization with Optuna (dissertation, Section 5.4.3,
Table 5.4): 30 trials of 15 epochs each, TPE sampler with a fixed seed, no
pruning, minimizing the MSE on the validation blocks defined in step 5.

Reads   data/processed/<sensor>_dataset.pkl
Writes  models/best_hyperparameters_<sensor>.pkl
        results/optuna_trials_<sensor>.csv

  * The scaler used during tuning is fitted on the inner training blocks only.
  * The sequence length is fixed at 60 windows (selected for the three sensors
    in a preliminary search over 30, 60, and 90; the block split depends on it).
  * Seeds are fixed (Optuna, PyTorch, NumPy). Memory is released after each
    trial on CUDA and on Apple MPS; if a trial still runs out of memory, it is
    repeated on the CPU instead of aborting the study.

Usage: python 06_tune_hyperparameters.py          (all three sensors)
       python 06_tune_hyperparameters.py mpu      (one sensor)
"""

import gc
import pickle
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pipeline_common import (MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, SENSORS, TIME_STEPS, block_mask,
                             get_device, set_seed)
from model_common import CNN_BiLSTM, SequenceDataset

N_TRIALS = 30
EPOCHS = 15


def load(sensor):
    with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "rb") as f:
        ds = pickle.load(f)
    n = len(ds["y"])
    scaler = StandardScaler().fit(ds["X_raw"][block_mask(ds["split"]["inner_train"], n)])
    Xs = scaler.transform(ds["X_raw"]).astype(np.float32)
    tr = SequenceDataset(Xs, ds["y"], ds["starts"]["inner_train"], TIME_STEPS)
    vl = SequenceDataset(Xs, ds["y"], ds["starts"]["inner_val"], TIME_STEPS)
    print(f"\n{sensor.upper()}: {len(tr)} inner training sequences, "
          f"{len(vl)} validation sequences, {Xs.shape[1]} features")
    return tr, vl, Xs.shape[1]


def free_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def objective(trial, tr, vl, input_dim, device):
    try:
        return run_trial(trial, tr, vl, input_dim, device)
    except RuntimeError as e:
        if "out of memory" not in str(e) or device.type == "cpu":
            raise
        print(f"  [trial {trial.number}] out of memory on {device}; repeating on the CPU")
        free_memory(device)
        return run_trial(trial, tr, vl, input_dim, torch.device("cpu"))


def run_trial(trial, tr, vl, input_dim, device):
    params = {
        "cnn_filters": trial.suggest_categorical("cnn_filters", [16, 32, 64]),
        "cnn_kernel": trial.suggest_categorical("cnn_kernel", [3, 5]),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    set_seed(SEED + trial.number)
    g = torch.Generator().manual_seed(SEED + trial.number)
    train_loader = DataLoader(tr, batch_size=256, shuffle=True, generator=g)
    val_loader = DataLoader(vl, batch_size=1024, shuffle=False)

    model = CNN_BiLSTM(input_dim, params["cnn_filters"], params["cnn_kernel"],
                       params["lstm_hidden"], params["lstm_layers"],
                       params["dropout_rate"]).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                            weight_decay=params["weight_decay"])
    for _ in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()

    model.eval()
    loss, n = 0.0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            loss += crit(model(bx), by).item() * bx.size(0)
            n += bx.size(0)
    del model, opt, bx, by, train_loader, val_loader
    free_memory(device)
    return loss / n


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    sensors = sys.argv[1:] or SENSORS
    for sensor in sensors:
        tr, vl, input_dim = load(sensor)
        study = optuna.create_study(direction="minimize", study_name=f"CNN_BiLSTM_{sensor}",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda tl: objective(tl, tr, vl, input_dim, device), n_trials=N_TRIALS)
        best = dict(study.best_trial.params, time_steps=TIME_STEPS)
        print(f"\nBest for {sensor.upper()} (val MSE {study.best_value:.6f}): {best}")
        with open(MODEL_DIR / f"best_hyperparameters_{sensor}.pkl", "wb") as f:
            pickle.dump(best, f)
        RESULTS_DIR.mkdir(exist_ok=True)
        study.trials_dataframe().to_csv(RESULTS_DIR / f"optuna_trials_{sensor}.csv", index=False)
        del tr, vl, study
        free_memory(device)
