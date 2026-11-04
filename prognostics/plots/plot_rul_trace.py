"""
Estimated RUL over the whole life cycle, as it would be computed in operation.

For each sensor, every contiguous sequence is passed through the model, the HI
is smoothed exactly as in the online pipeline (rolling median of 60 windows +
EMA with a span of 60, see online/online_common.py), and converted into a RUL
with the definition of the labels, RUL = (1 - log2(2 - HI)) * L (Equation 5.4).
Before the FPT the HI is 1 and the RUL stays at L: "healthy, at least L hours",
as in a piecewise linear RUL.

Note: about 80 % of the windows belong to the training set, so the full curve is
optimistic. The held-out test sequences (unsmoothed, isolated sequences) are the
ones that measure the real error, and are marked separately in the figure.

Reads   data/processed/<sensor>_dataset.pkl, models/scaler_<sensor>.pkl, models/model_<sensor>.pt
Writes  results/figures/rul_trace_<sensor>.pdf and .png

Usage: python prognostics/plots/plot_rul_trace.py [sensor ...]
"""

import pickle
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # prognostics/
from pipeline_common import (MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, SENSORS, contiguity,
                             get_device, hi_to_rul, sequence_starts)
from model_common import CNN_BiLSTM, SequenceDataset, predict

FIGURES = RESULTS_DIR / "figures"
MEDIAN, EMA = 60, 60
T_NEEDED_H = 3.6


def smooth(hi_raw):
    """Same smoothing as the Estimator of online/online_common.py."""
    buf, alpha, ema, out = deque(maxlen=MEDIAN), 2.0 / (EMA + 1), None, np.empty(len(hi_raw))
    for i, h in enumerate(hi_raw):
        buf.append(h)
        med = float(np.median(buf))
        ema = med if ema is None else alpha * med + (1 - alpha) * ema
        out[i] = ema
    return out


def load_model(sensor, device):
    ck = torch.load(MODEL_DIR / f"model_{sensor}.pt", map_location="cpu", weights_only=False)
    hp = ck["hparams"]
    model = CNN_BiLSTM(ck["input_dim"], hp["cnn_filters"], hp["cnn_kernel"], hp["lstm_hidden"],
                       hp["lstm_layers"], hp["dropout_rate"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sensors = sys.argv[1:] or SENSORS
    for sensor in sensors:
        with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "rb") as f:
            ds = pickle.load(f)
        with open(MODEL_DIR / f"scaler_{sensor}.pkl", "rb") as f:
            scaler = pickle.load(f)
        model = load_model(sensor, device)
        T, L = ds["time_steps"], ds["label_info"]["L_s"]
        n = len(ds["y"])
        Xs = scaler.transform(ds["X_raw"]).astype(np.float32)
        cont = contiguity(ds["session"], ds["win"])

        # whole life cycle, as in operation
        starts = sequence_starts(np.ones(n, dtype=bool), cont, T)
        hi_raw, _ = predict(model, SequenceDataset(Xs, ds["y"], starts, T), device)
        last = starts + T - 1
        hours = ds["t"][last] / 3600
        hi_s = smooth(np.clip(hi_raw, 0.0, 1.0))
        rul_pred = hi_to_rul(hi_s, L) / 3600
        rul_true = np.minimum(ds["rul_s"][last], L) / 3600

        # test sequences, unsmoothed
        te = ds["starts"]["test"]
        hi_te, _ = predict(model, SequenceDataset(Xs, ds["y"], te, T), device)
        rul_te = hi_to_rul(np.clip(hi_te, 0.0, 1.0), L) / 3600
        h_te = ds["t"][te + T - 1] / 3600

        t_eol = ds["label_info"]["t_eol_s"] / 3600
        t_fpt = ds["label_info"]["t_fpt_s"] / 3600
        print(f"\n{sensor.upper()}  L = {L / 3600:.2f} h, FPT {t_fpt:.2f} h, EoL {t_eol:.2f} h")
        healthy = hours < t_fpt
        if healthy.any():
            print(f"  before the FPT: estimated RUL between {rul_pred[healthy].min():.2f} and "
                  f"{rul_pred[healthy].max():.2f} h (median {np.median(rul_pred[healthy]):.2f} h)")
        for before in [8, 6, 5, 4, 3, 2, 1]:
            i = np.argmin(np.abs(hours - (t_eol - before)))
            if abs(hours[i] - (t_eol - before)) < 0.1:
                print(f"  {before} h before the EoL: estimated RUL {rul_pred[i]:.2f} h")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, rul_true, color="black", lw=2, label="Reference RUL")
        ax.plot(hours, rul_pred, color="red", lw=1.2, label="Estimated RUL (smoothed, as deployed)")
        ax.scatter(h_te, rul_te, s=10, color="tab:blue", zorder=3,
                   label="Held-out test sequences (unsmoothed)")
        ax.axhline(T_NEEDED_H, color="gray", ls=":", lw=1, label="Time to act (print + replacement)")
        ax.axvline(t_fpt, color="gray", ls="--", lw=1)
        ax.axvline(t_eol, color="black", ls="--", lw=1)
        ax.text(t_fpt, L / 3600 * 1.02, " FPT", fontsize=10)
        ax.text(t_eol, L / 3600 * 1.02, " EoL", fontsize=10)
        ax.set_xlabel("Operating time (h)", fontsize=13)
        ax.set_ylabel("RUL (h)", fontsize=13)
        ax.set_ylim(-0.3, L / 3600 * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="lower left")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(FIGURES / f"rul_trace_{sensor}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  figure: {FIGURES / f'rul_trace_{sensor}.pdf'}")
