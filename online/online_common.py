"""
Code shared by the three online estimation scripts (online_rul_adxl.py,
online_rul_piezo.py, online_rul_mpu.py), which run on the PdM node of each arm
(dissertation, Section 5.5).

Deploy this file in the same folder as the scripts on each PdM node (the scripts
import it directly, since Python adds the script's folder to sys.path).

It reproduces exactly the preprocessing of training
(prognostics/01_sync_data.py + 02_extract_features.py):
  1. resampling to 2800 Hz:
       ADXL345 and MPU9250 - streaming polyphase resampling, whose output is
       identical to resample_poly applied to the whole recording;
       piezoelectric sensor - continuous linear interpolation from the
       reception timestamps;
  2. causal 4th-order Butterworth high-pass at 1 Hz, with its state kept across
     the whole stream and started from the steady state of the first sample;
  3. windows of exactly 2800 samples and the same 11 features per axis, in the
     same order as the training columns (axis by axis);
  4. windows without signal (variance of the first axis <= 1e-6) are discarded
     and the sequence is restarted, as in training (contiguous windows only);
  5. NaN/inf features are set to 0, as in training.
After the model: rolling median + EMA, HI -> RUL conversion with the definition
of the labels (L is stored in the checkpoint), and an atomic write of the file
read by the RUL_ESTIMATION function block.

Format of the file read by RUL_ESTIMATION (same for the three sensors):
  line 1: smoothed HI, between 0 and 1
  line 2: status ("OK" or "WARMUP")
  line 3: estimated RUL in seconds (-1 if the checkpoint has no L)
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.stats import kurtosis, skew

FS = 2800
WINDOW = FS
HP_CUTOFF = 1.0
HP_ORDER = 4
ACTIVITY_THRESHOLD = 1e-6
FREQ_BANDS = [(0, 350), (350, 700), (700, 1050), (1050, 1400)]
FEATURE_KEYS = ["rms", "var", "peak", "crest_factor", "skew", "kurtosis",
                "total_spectral_energy", "energy_band_1", "energy_band_2",
                "energy_band_3", "energy_band_4"]
DEFAULT_MEDIAN = 60
DEFAULT_EMA = 60


# ── model ──────────────────────────────────────────────────────────────────
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters, cnn_kernel, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_filters, cnn_kernel, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        x = self.relu(self.cnn(x.transpose(1, 2)))
        x = self.pool(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def load_model_and_scaler(model_path, scaler_path, expected_cols=None):
    model_path, scaler_path = Path(model_path), Path(scaler_path)
    for p in (model_path, scaler_path):
        if not p.exists():
            sys.exit(f"File not found: {p}")
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    hp = ckpt["hparams"]
    model = CNN_BiLSTM(ckpt["input_dim"], hp["cnn_filters"], hp["cnn_kernel"],
                       hp["lstm_hidden"], hp["lstm_layers"], hp["dropout_rate"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    if getattr(scaler, "n_features_in_", ckpt["input_dim"]) != ckpt["input_dim"]:
        sys.exit("The scaler and the model do not have the same number of features.")
    if expected_cols is not None:
        if ckpt["input_dim"] != len(expected_cols):
            sys.exit(f"The model expects {ckpt['input_dim']} features; the script produces {len(expected_cols)}.")
        trained = ckpt.get("feature_cols")
        if trained is not None and list(trained) != list(expected_cols):
            sys.exit("The order of the features of the model does not match that of the script.")
    L = (ckpt.get("label_info") or {}).get("L_s")
    print(f"[online] model {model_path.name}: {ckpt['input_dim']} features, "
          f"{ckpt['time_steps']} windows, L = {'-' if L is None else f'{L / 3600:.2f} h'}",
          file=sys.stderr)
    return model, scaler, int(ckpt["time_steps"]), L


def feature_columns(axes):
    return [f"{a}_{k}" for a in axes for k in FEATURE_KEYS]


# ── streaming resampling ───────────────────────────────────────────────────
class PolyphaseStream:
    """Streaming resample_poly(up, down). Gives exactly the same output as
    resample_poly applied to the whole signal (including the zero padding at
    the start), except for the last samples, which cannot be emitted yet."""

    def __init__(self, up, down, channels, margin=400):
        self.up, self.down = up, down
        self.M = int(np.ceil(margin / down) * down)
        self.buf = np.zeros((self.M, channels))
        self.base = -self.M
        self.next_m = 0

    def push(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
        self.buf = np.vstack([self.buf, x])
        total = self.base + len(self.buf)
        m_max = int(np.floor((total - 1 - self.M) * self.up / self.down))
        if m_max < self.next_m:
            return np.empty((0, self.buf.shape[1]))
        y = signal.resample_poly(self.buf, self.up, self.down, axis=0)
        off = self.base * self.up // self.down
        out = y[self.next_m - off: m_max + 1 - off]
        self.next_m = m_max + 1
        keep_from = ((self.next_m * self.down) // self.up - self.M) // self.down * self.down
        if keep_from > self.base:
            self.buf = self.buf[keep_from - self.base:]
            self.base = keep_from
        return out


class InterpStream:
    """Continuous linear interpolation onto a uniform 2800 Hz grid from irregular
    timestamps (piezoelectric sensor). Same as np.interp over the grid
    np.arange(t[0], t[-1], 1/2800) of the whole recording."""

    def __init__(self):
        self.t_prev = self.v_prev = self.t_next = None

    def push(self, t, v):
        t = np.asarray(t, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if len(t) == 0:
            return np.empty((0, 1))
        if self.t_prev is not None:
            t = np.concatenate([[self.t_prev], t])
            v = np.concatenate([[self.v_prev], v])
        if self.t_next is None:
            self.t_next = t[0]
        self.t_prev, self.v_prev = t[-1], v[-1]
        dt = 1.0 / FS
        k = int(np.floor((t[-1] - self.t_next) / dt)) + 1
        if k <= 0 or len(t) < 2:
            return np.empty((0, 1))
        grid = self.t_next + dt * np.arange(k)
        self.t_next = grid[-1] + dt
        return np.interp(grid, t, v).reshape(-1, 1)


# ── filter and windows ────────────────────────────────────────────────────────
class CausalHighpass:
    def __init__(self, channels):
        self.sos = signal.butter(HP_ORDER, HP_CUTOFF, btype="high", fs=FS, output="sos")
        self.zi = None
        self.channels = channels

    def push(self, x):
        if len(x) == 0:
            return x
        if self.zi is None:
            z0 = signal.sosfilt_zi(self.sos)
            self.zi = np.stack([z0 * x[0, c] for c in range(self.channels)], axis=-1)
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class Windower:
    def __init__(self, channels):
        self.buf = np.empty((0, channels))

    def push(self, x):
        self.buf = np.vstack([self.buf, x])
        out = []
        while len(self.buf) >= WINDOW:
            out.append(self.buf[:WINDOW].astype(np.float32))
            self.buf = self.buf[WINDOW:]
        return out


def extract_features(window):
    """window: (2800, n_axes). Returns the 11 features of each axis, axis by axis."""
    feats = []
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    for c in range(window.shape[1]):
        w = window[:, c]
        rms = float(np.sqrt(np.mean(w ** 2)))
        peak = float(np.max(np.abs(w)))
        mag2 = np.abs(np.fft.rfft(w)) ** 2
        feats += [rms, float(np.var(w)), peak, peak / (rms + 1e-8), float(skew(w)),
                  float(kurtosis(w)), float(np.sum(mag2) / n)]
        feats += [float(np.sum(mag2[(freqs >= lo) & (freqs < hi)]) / n) for lo, hi in FREQ_BANDS]
    f = np.array(feats, dtype=np.float64)
    return np.where(np.isfinite(f), f, 0.0)


# ── estimator ───────────────────────────────────────────────────────────────
def hi_to_rul(hi, L):
    hi = min(max(float(hi), 0.0), 1.0)
    return (1.0 - np.log2(2.0 - hi)) * L


class Estimator:
    """Filtered window -> features -> scaler -> sequence -> model -> smoothed HI
    -> RUL. push_window returns a dict for each prediction, or None."""

    def __init__(self, model, scaler, time_steps, L, median=DEFAULT_MEDIAN, ema=DEFAULT_EMA,
                 monotone=False, dump_features=None):
        self.model, self.scaler, self.T, self.L = model, scaler, time_steps, L
        self.seq = deque(maxlen=time_steps)
        self.raw = deque(maxlen=max(1, median))
        self.alpha = 2.0 / (max(1, ema) + 1)
        self.ema = None
        self.monotone = monotone
        self.mono = 1.0
        self.n_windows = 0
        self.n_dead = 0
        # line-buffered: each window reaches the disk even if the process is
        # terminated with SIGTERM by the STOP event of the function block
        self.dump = open(dump_features, "w", buffering=1) if dump_features else None

    def push_window(self, w):
        self.n_windows += 1
        if np.var(w[:, 0]) <= ACTIVITY_THRESHOLD:
            self.n_dead += 1
            self.seq.clear()
            return None
        f = extract_features(w)
        if self.dump:
            self.dump.write(f"{time.time():.3f}," + ",".join(f"{v:.10g}" for v in f) + "\n")
        self.seq.append(self.scaler.transform(f.reshape(1, -1)).astype(np.float32)[0])
        if len(self.seq) < self.T:
            return None
        x = torch.tensor(np.stack(self.seq)[None], dtype=torch.float32)
        with torch.no_grad():
            hi_raw = float(np.clip(self.model(x)[0].item(), 0.0, 1.0))
        self.raw.append(hi_raw)
        med = float(np.median(self.raw))
        self.ema = med if self.ema is None else self.alpha * med + (1 - self.alpha) * self.ema
        hi = self.ema
        if self.monotone:
            self.mono = min(self.mono, hi)
            hi = self.mono
        rul = None if self.L is None else float(hi_to_rul(hi, self.L))
        return {"window": self.n_windows, "hi_raw": hi_raw, "hi": hi, "rul_s": rul}


def write_atomic(path, hi, rul_s, status="OK"):
    if not path:
        return
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as fh:
            fh.write(f"{hi:.6f}\n{status}\n{-1.0 if rul_s is None else rul_s:.1f}\n")
        os.replace(tmp, path)
    except OSError:
        pass


def report(result, quiet, tag):
    if quiet or result is None:
        return
    rul = "-" if result["rul_s"] is None else f"{result['rul_s'] / 3600:.2f} h"
    print(f"[{tag}] window {result['window']:>7} | raw HI {result['hi_raw']:.4f} | "
          f"HI {result['hi']:.4f} | RUL {rul}", flush=True)


def add_common_args(ap):
    ap.add_argument("--model", required=False)
    ap.add_argument("--scaler", required=False)
    ap.add_argument("--rul-file", default=None,
                    help="File read by the RUL_ESTIMATION function block (HI 0-1, status, RUL in s).")
    ap.add_argument("--median", type=int, default=DEFAULT_MEDIAN)
    ap.add_argument("--ema", type=int, default=DEFAULT_EMA)
    ap.add_argument("--monotone", action="store_true",
                    help="Prevent the smoothed HI from increasing (off by default).")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dump-features", default=None,
                    help="Write, for each window, its time (unix) and features to a CSV "
                         "(to compare with training and to analyze live runs).")
