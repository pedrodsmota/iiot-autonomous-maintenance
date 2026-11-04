"""
Step 9 - Evaluation with whole sessions held out (dissertation, Section 6.3,
Table 6.5 and Figure 6.6).

The random split into blocks puts windows of every session in both the training
and the test sets, so a model can reproduce the test labels partly by
recognizing the session. To approximate a new degradation run, the sessions are
divided into K interleaved groups (sessions 1, 6, 11, ...; 2, 7, 12, ...; and so
on), so that each group covers the whole life cycle. For each group, a model is
trained from scratch without any window of those sessions (including its own
scaler), with the same architecture, hyperparameters, optimizer, loss, epochs,
and batch size as step 7, and evaluated on the held-out sessions. In the end,
every session is estimated by a model that never saw it.

Reads   data/processed/<sensor>_dataset.pkl, models/best_hyperparameters_<sensor>.pkl
Writes  results/loso_<sensor>.json        overall metrics, before/after the FPT, per session
        results/loso_<sensor>_pred.csv    out-of-sample predictions (session, t, true HI, predicted HI)
        results/figures/loso_<sensor>.pdf

Usage: python 09_loso_eval.py adxl                      (5 groups, 50 epochs)
       python 09_loso_eval.py adxl --folds 5 --epochs 50 --stride 10
"""

import argparse
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pipeline_common import (MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, contiguity,
                             get_device, sequence_starts, set_seed)
from model_common import CNN_BiLSTM, SequenceDataset, predict

FIGURES = RESULTS_DIR / "figures"


def metrics(true, pred):
    return {"rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
            "mae": float(np.mean(np.abs(pred - true))),
            "n": int(len(true))}


def train_model(Xs, y, starts, hp, T, epochs, device):
    set_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(SequenceDataset(Xs, y, starts, T), batch_size=256, shuffle=True, generator=g)
    model = CNN_BiLSTM(Xs.shape[1], hp["cnn_filters"], hp["cnn_kernel"], hp["lstm_hidden"],
                       hp["lstm_layers"], hp["dropout_rate"]).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item() * bx.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:>2}/{epochs}  training MSE {tot / len(starts):.6f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sensor", choices=["adxl", "piezo", "mpu"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--stride", type=int, default=10,
                    help="Step between the sequences evaluated in the held-out sessions.")
    a = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()

    with open(PROCESSED_DIR / f"{a.sensor}_dataset.pkl", "rb") as f:
        ds = pickle.load(f)
    with open(MODEL_DIR / f"best_hyperparameters_{a.sensor}.pkl", "rb") as f:
        hp = pickle.load(f)
    T = ds["time_steps"]
    X, y = ds["X_raw"], ds["y"]
    session, win = np.asarray(ds["session"]), np.asarray(ds["win"])
    cont = contiguity(session, win)
    sessions = np.unique(session)
    groups = [sessions[k::a.folds] for k in range(a.folds)]
    print(f"Device: {device} | {a.sensor.upper()} | {len(sessions)} sessions in {a.folds} groups: "
          f"{[g.tolist() for g in groups]}")

    rows = []
    for k, held in enumerate(groups):
        t0 = time.time()
        m_test = np.isin(session, held)
        m_train = ~m_test
        tr_starts = sequence_starts(m_train, cont, T)
        te_starts = sequence_starts(m_test, cont, T)[::a.stride]
        scaler = StandardScaler().fit(X[m_train])
        Xs = scaler.transform(X).astype(np.float32)
        print(f"\n  Group {k + 1}/{a.folds}: held-out sessions {held.tolist()} | "
              f"training {len(tr_starts)} seq. | evaluation {len(te_starts)} seq.", flush=True)
        model = train_model(Xs, y, tr_starts, hp, T, a.epochs, device)
        pred, true = predict(model, SequenceDataset(Xs, y, te_starts, T), device)
        pred = np.clip(pred, 0.0, 1.0)
        last = te_starts + T - 1
        rows.append(pd.DataFrame({"fold": k + 1, "session": session[last], "t_s": ds["t"][last],
                                  "hi_true": true, "hi_pred": pred}))
        print(f"    RMSE {metrics(true, pred)['rmse']:.4f} ({time.time() - t0:.0f} s)", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    df = pd.concat(rows).sort_values("t_s")
    df.to_csv(RESULTS_DIR / f"loso_{a.sensor}_pred.csv", index=False)
    pre, post = df.hi_true >= 1.0, df.hi_true < 1.0
    out = {"sensor": a.sensor, "folds": [g.tolist() for g in groups], "epochs": a.epochs,
           "all": metrics(df.hi_true.values, df.hi_pred.values),
           "before_fpt": metrics(df.hi_true[pre].values, df.hi_pred[pre].values),
           "after_fpt": metrics(df.hi_true[post].values, df.hi_pred[post].values),
           "per_session": {}}
    print(f"\n{'session':>7} {'true HI':>8} {'pred. HI':>9} {'p10':>6} {'p90':>6} {'RMSE':>6}")
    for s, g in df.groupby("session"):
        r = {"hi_true_mean": float(g.hi_true.mean()), "hi_pred_mean": float(g.hi_pred.mean()),
             "hi_pred_p10": float(g.hi_pred.quantile(.1)), "hi_pred_p90": float(g.hi_pred.quantile(.9)),
             "rmse": float(np.sqrt(np.mean((g.hi_pred - g.hi_true) ** 2)))}
        out["per_session"][int(s)] = r
        print(f"{int(s):>7} {r['hi_true_mean']:8.3f} {r['hi_pred_mean']:9.3f} "
              f"{r['hi_pred_p10']:6.3f} {r['hi_pred_p90']:6.3f} {r['rmse']:6.3f}")
    for k in ("all", "before_fpt", "after_fpt"):
        m = out[k]
        print(f"{k:<11} RMSE {m['rmse']:.4f} | MAE {m['mae']:.4f} | n {m['n']}")
    (RESULTS_DIR / f"loso_{a.sensor}.json").write_text(json.dumps(out, indent=2))

    h = df.t_s / 3600
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h, df.hi_true, color="black", lw=2, label="Reference HI")
    ax.scatter(h, df.hi_pred, s=4, color="red", alpha=0.5,
               label=f"Predicted HI, held-out sessions (RMSE={out['all']['rmse']:.4f})")
    ax.set_xlabel("Operating time (h)", fontsize=14)
    ax.set_ylabel("Health Index", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / f"loso_{a.sensor}.pdf")
    print(f"\nResults in {RESULTS_DIR}/loso_{a.sensor}.* and {FIGURES}/loso_{a.sensor}.pdf")


if __name__ == "__main__":
    main()
