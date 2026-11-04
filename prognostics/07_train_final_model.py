"""
Step 7 - Train the final model of each sensor and evaluate it on the test
sequences (dissertation, Sections 5.4.3 and 6.3, Table 6.3).

Reads   data/processed/<sensor>_dataset.pkl, models/scaler_<sensor>.pkl,
        models/best_hyperparameters_<sensor>.pkl
Writes  models/model_<sensor>.pt
        results/metrics.json
        results/figures/prediction_<sensor>_hi_only.pdf, results/figures/loss_<sensor>.png

  * 50 epochs, AdamW, MSE loss, batch size 256, no early stopping or learning
    rate scheduling, seeds of Python, NumPy, and PyTorch fixed.
  * Metrics on the test sequences: RMSE and MAE of the HI, and, after the FPT,
    the RMSE and MAE of the RUL derived from the predicted HI (Equation 5.4).
  * The checkpoint also stores the definition of the labels (t_FPT, t_EoL, L)
    and the ordered list of features, which the online scripts use to convert
    the HI into a RUL and to check the order of the features they compute.

Usage: python 07_train_final_model.py          (all three sensors)
       python 07_train_final_model.py piezo    (one sensor; metrics.json is merged)
"""

import gc
import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline_common import (MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, SENSORS,
                             get_device, hi_to_rul, set_seed)
from model_common import CNN_BiLSTM, SequenceDataset, predict

EPOCHS = 50
FIGURES = RESULTS_DIR / "figures"


def train_sensor(sensor, device):
    with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "rb") as f:
        ds = pickle.load(f)
    with open(MODEL_DIR / f"scaler_{sensor}.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / f"best_hyperparameters_{sensor}.pkl", "rb") as f:
        hp = pickle.load(f)
    T = ds["time_steps"]
    Xs = scaler.transform(ds["X_raw"]).astype(np.float32)
    train_set = SequenceDataset(Xs, ds["y"], ds["starts"]["train"], T)
    test_set = SequenceDataset(Xs, ds["y"], ds["starts"]["test"], T)
    print(f"\n{'=' * 60}\n  {sensor.upper()} | training {len(train_set)} | test {len(test_set)} "
          f"| features {Xs.shape[1]}\n  {hp}\n{'=' * 60}")

    set_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_set, batch_size=256, shuffle=True, generator=g)
    model = CNN_BiLSTM(Xs.shape[1], hp["cnn_filters"], hp["cnn_kernel"], hp["lstm_hidden"],
                       hp["lstm_layers"], hp["dropout_rate"]).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"],
                            weight_decay=hp["weight_decay"])
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        tot = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item() * bx.size(0)
        losses.append(tot / len(train_set))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:>2}/{EPOCHS}  train MSE {losses[-1]:.6f}")

    pred, true = predict(model, test_set, device)
    pred = np.clip(pred, 0.0, 1.0)
    m = {"rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
         "mae": float(np.mean(np.abs(pred - true))),
         "n_test": int(len(true)),
         "n_params": int(sum(p.numel() for p in model.parameters()))}

    # equivalent RUL (h), only after the FPT
    L = ds["label_info"]["L_s"]
    last = ds["starts"]["test"] + T - 1
    post = true < 1.0
    rul_true = ds["rul_s"][last][post] / 3600
    rul_pred = hi_to_rul(pred[post], L) / 3600
    m["rul_rmse_h"] = float(np.sqrt(np.mean((rul_pred - rul_true) ** 2)))
    m["rul_mae_h"] = float(np.mean(np.abs(rul_pred - rul_true)))
    m["n_test_post_fpt"] = int(post.sum())
    print(f"  Test: RMSE {m['rmse']:.4f} | MAE {m['mae']:.4f} | "
          f"RUL MAE {m['rul_mae_h']:.2f} h, RMSE {m['rul_rmse_h']:.2f} h "
          f"({m['n_test_post_fpt']} sequences after the FPT)")

    hours = ds["t"][last] / 3600
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(hours, true, color="black", lw=2, label="Reference HI")
    ax.plot(hours, pred, color="red", lw=1.5, ls="--", label=f"Predicted HI (RMSE={m['rmse']:.4f})")
    ax.set_xlabel("Operating time (h)", fontsize=14)
    ax.set_ylabel("Health Index", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / f"prediction_{sensor}_hi_only.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / f"loss_{sensor}.png", dpi=150)
    plt.close(fig)

    torch.save({"model_state_dict": model.state_dict(), "hparams": hp,
                "input_dim": int(Xs.shape[1]), "time_steps": T, "sensor": sensor,
                "feature_cols": ds["feature_cols"], "target": ds["target"],
                "label_info": ds["label_info"], "metrics": m},
               MODEL_DIR / f"model_{sensor}.pt")
    del model, opt, loader, bx, by
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return m


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    sensors = sys.argv[1:] or SENSORS
    # merge with previous results, so that one sensor can be trained at a time
    mfile = RESULTS_DIR / "metrics.json"
    summary = json.loads(mfile.read_text()) if mfile.exists() else {}
    for s in sensors:
        summary[s] = train_sensor(s, device)
        with open(mfile, "w") as f:
            json.dump(summary, f, indent=2)
    with open(mfile, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSensor   RMSE     MAE      RUL MAE (h)")
    for s, m in summary.items():
        print(f"{s:<8} {m['rmse']:.4f}   {m['mae']:.4f}   {m['rul_mae_h']:.2f}")
