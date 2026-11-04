#!/usr/bin/env python3
"""
online_rul_adxl.py — Real-time Remaining Useful Life estimator for ADXL345
===========================================================================
Implements the same preprocessing chain as the training pipeline, adapted
for continuous online operation:

  Samples (2800 Hz, native) ─► HP filter (stateful) ─► feature extraction
  ─► StandardScaler (training scaler) ─► sequence buffer (TIME_STEPS windows)
  ─► CNN-BiLSTM inference ─► rolling-median smoother ─► Health Index + RUL estimate

Output smoothing — rolling median over 50–100 predicted HI windows:
  The raw model output is noisy because a single 1-second window during a
  pick-and-place or other transient operation can look like a health shock.
  A rolling median of N windows is immune to these spikes by construction:
  a handful of outlier windows cannot move the median of 50–100 values.
  This is the same technique applied in construct_hs_hi.py before the GMM
  and is physically justified for the same reason — genuine degradation is
  a slow, monotonic process, not a 1-second event.

  EWM (exponential weighted mean) was considered but rejected: the mean is
  still pulled by outlier values; only the median is truly spike-immune.

Because the ADXL345 already samples at 2800 Hz, the resampling step from
sync_data.py is not needed here.  The Butterworth HP filter is kept because
it was applied during training; running it with persistent filter state
(sosfilt + zi) avoids edge artefacts at window boundaries.

Data source modes (--mode):
  stdin   — lines piped from adxl345spi  (default)
  serial  — direct UART/SPI-bridge serial port
  file    — replay a previously recorded CSV (offline validation)

Expected line format (same as adxl345spi.c output):
  x,y,z                    e.g.  0.031,-0.012,1.002
  timestamp,x,y,z          e.g.  1234567,0.031,-0.012,1.002

Units must match training data (raw ADC counts or g — whichever was used).

Usage examples:
  # Pipe live data from the C collector
  ./adxl345spi | python3 online_rul_adxl.py

  # Read from serial port directly
  python3 online_rul_adxl.py --mode serial --port /dev/ttyUSB0 --baud 921600

  # Replay a recorded file (useful for validation)
  python3 online_rul_adxl.py --mode file --file trial_22.csv
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy import signal
from scipy.stats import skew, kurtosis
from pathlib import Path

# ── Constants (must match training pipeline) ──────────────────────────────────
FS          = 2800      # native sampling rate — no resampling needed
HP_CUTOFF   = 1.0      # high-pass cut-off (Hz), same as sync_data.py
WINDOW_SIZE = FS        # 1-second feature window = 2800 samples

# Explicit feature column order — must match extract_features.py exactly.
# Built by iterating columns=['x','y','z'] then the feature dict keys in
# insertion order (Python 3.7+ guarantees dict insertion order).
_AXES       = ['x', 'y', 'z']
_TIME_FEATS = ['rms', 'var', 'peak', 'crest_factor', 'skew', 'kurtosis']
_FREQ_FEATS = ['total_spectral_energy',
               'energy_band_1', 'energy_band_2',
               'energy_band_3', 'energy_band_4']
FEATURE_COLS = [f"{ax}_{feat}"
                for ax in _AXES
                for feat in _TIME_FEATS + _FREQ_FEATS]   # 33 features
assert len(FEATURE_COLS) == 33


# ── Model architecture (must be identical to train_final_model.py) ────────────
class CNN_BiLSTM(nn.Module):
    """
    1D-CNN auxiliary layer compresses local spectral patterns; BiLSTM
    captures the temporal degradation trajectory and regresses HI ∈ [0, 1].
    """
    def __init__(self, input_dim, cnn_filters, cnn_kernel,
                 lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.cnn     = nn.Conv1d(input_dim, cnn_filters, cnn_kernel, padding='same')
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(2)
        self.lstm    = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers,
                               batch_first=True,
                               dropout=dropout if lstm_layers > 1 else 0,
                               bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden * 2, 1)  # ×2 — bidirectional

    def forward(self, x):
        # x: (batch, time_steps, features)
        x = self.relu(self.cnn(x.transpose(1, 2)))   # → (B, filters, T)
        x = self.pool(x).transpose(1, 2)              # → (B, T/2, filters)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze()


# ── Feature extraction — identical to extract_features.py ────────────────────
def _extract_axis(data: np.ndarray, fs: int = FS) -> dict:
    """Time + frequency features for a single 1-D axis window."""
    feat = {}
    feat['rms']          = float(np.sqrt(np.mean(data ** 2)))
    feat['var']          = float(np.var(data))
    feat['peak']         = float(np.max(np.abs(data)))
    feat['crest_factor'] = feat['peak'] / (feat['rms'] + 1e-8)
    feat['skew']         = float(skew(data))
    feat['kurtosis']     = float(kurtosis(data))

    n        = len(data)
    fft_mag  = np.abs(np.fft.rfft(data))
    fft_freq = np.fft.rfftfreq(n, d=1.0 / fs)

    feat['total_spectral_energy'] = float(np.sum(fft_mag ** 2) / n)
    for i, (lo, hi) in enumerate([(0, 350), (350, 700), (700, 1050), (1050, 1400)]):
        idx = np.where((fft_freq >= lo) & (fft_freq < hi))[0]
        feat[f'energy_band_{i + 1}'] = float(np.sum(fft_mag[idx] ** 2) / n)
    return feat


def window_to_vector(window_xyz: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    window_xyz : (WINDOW_SIZE, 3) float32 — HP-filtered x, y, z samples

    Returns
    -------
    np.ndarray of shape (33,) in the same column order as the training CSV
    """
    row = {}
    for ax, col_data in zip(_AXES, window_xyz.T):
        for k, v in _extract_axis(col_data).items():
            row[f'{ax}_{k}'] = v

    # Build vector in the guaranteed training column order
    return np.array([row[c] for c in FEATURE_COLS], dtype=np.float32)


# ── RUL helper ────────────────────────────────────────────────────────────────
def hi_to_rul_fraction(hi: float) -> float:
    """
    Inverts the exponential decay from construct_hs_hi.py:
        HI(t) = 2 − exp(ln 2 · t_norm),   t_norm ∈ [0, 1]
    ↔  t_norm = log(2 − HI) / log 2
    ↔  RUL fraction = 1 − t_norm   (1.0 = fully healthy, 0.0 = end of life)
    """
    hi = float(np.clip(hi, 1e-6, 1.0))
    if hi >= 1.0:
        return 1.0
    arg = 2.0 - hi
    if arg <= 0.0:
        return 0.0
    return float(np.clip(1.0 - np.log(arg) / np.log(2.0), 0.0, 1.0))


# ── Core estimator ────────────────────────────────────────────────────────────
class OnlineRULEstimator:
    """
    Stateful online estimator.  Call push_sample() for every incoming
    x,y,z triple.  Returns a result dict once per completed second window
    (after the initial TIME_STEPS warm-up period).
    """

    def __init__(self,
                 model_path:    Path,
                 scaler_path:   Path,
                 median_window: int = 75):
        """
        Parameters
        ----------
        model_path    : path to model_adxl.pt (torch checkpoint)
        scaler_path   : path to scaler_adxl.pkl (fitted StandardScaler)
        median_window : rolling-median window size applied to raw HI predictions.
                        50–100 windows (seconds) is recommended — large enough
                        that transient operational shocks (pick-and-place, etc.)
                        cannot move the median, small enough to still track slow
                        real degradation.  Set to 1 to disable.
        """
        # ── Load model ─────────────────────────────────────────────────────
        ckpt            = torch.load(model_path, map_location='cpu',
                                     weights_only=False)
        hp              = ckpt['hparams']
        self.time_steps = int(ckpt.get('time_steps', 60))
        input_dim       = int(ckpt['input_dim'])

        if input_dim != len(FEATURE_COLS):
            raise ValueError(
                f"Model expects {input_dim} features but this script produces "
                f"{len(FEATURE_COLS)}.  Check FEATURE_COLS ordering.")

        self.model = CNN_BiLSTM(
            input_dim   = input_dim,
            cnn_filters = hp['cnn_filters'],
            cnn_kernel  = hp['cnn_kernel'],
            lstm_hidden = hp['lstm_hidden'],
            lstm_layers = hp['lstm_layers'],
            dropout     = hp['dropout_rate'],
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # ── Load scaler ─────────────────────────────────────────────────────
        with open(scaler_path, 'rb') as fh:
            self.scaler = pickle.load(fh)

        # ── HP filter — SOS form for numerical stability ─────────────────────
        # Using sosfilt with persistent zi avoids the Gibbs-like edge artefacts
        # that filtfilt would introduce if applied window-by-window.
        # Note: filtfilt was used during training (offline), but for causal
        # online use, sosfilt with matched zi is the correct equivalent.
        self.sos = signal.butter(4, HP_CUTOFF / (0.5 * FS),
                                 btype='high', output='sos')
        zi_proto = signal.sosfilt_zi(self.sos)          # (n_sections, 2)
        # Persistent state for 3 channels — shape: (n_sections, 2, 3)
        self.zi  = np.stack([zi_proto] * 3, axis=-1)

        # ── Buffers ──────────────────────────────────────────────────────────
        self.sample_buf  = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
        self.sample_idx  = 0
        self.feat_buffer = deque(maxlen=self.time_steps)  # ring buffer of feature vectors

        # ── Rolling-median buffer for spike-immune HI smoothing ──────────────
        # A rolling median over N windows is unaffected by individual shock
        # windows (pick-and-place, etc.).  np.median of a deque is O(N) but
        # N≤100 makes this negligible compared to model inference.
        self._hi_raw_buf = deque(maxlen=max(1, median_window))

        # ── Counters ─────────────────────────────────────────────────────────
        self.window_count = 0   # completed 1-second windows

        print(f"[RUL] Model loaded         input_dim={input_dim}, "
              f"time_steps={self.time_steps}, device={self.device}")
        print(f"[RUL] HP filter            4th-order Butterworth @ {HP_CUTOFF} Hz  "
              f"(stateful sosfilt)")
        print(f"[RUL] HI smoother          rolling median, window={median_window} windows "
              f"(~{median_window}s lag)\n")

    # ── Public API ────────────────────────────────────────────────────────────
    def push_sample(self, x: float, y: float, z: float):
        """
        Ingest one raw sample.  Call at 2800 Hz.

        Returns
        -------
        dict  with keys  hi, hi_raw, rul_fraction, rul_pct, window_idx, status
              when a complete inference step is ready, otherwise None.
        """
        self.sample_buf[self.sample_idx] = (x, y, z)
        self.sample_idx += 1

        if self.sample_idx < WINDOW_SIZE:
            return None   # window not yet full

        # ── Reset accumulator ────────────────────────────────────────────────
        self.sample_idx  = 0
        self.window_count += 1
        window = self.sample_buf.copy()   # (2800, 3)

        # ── HP filter with persistent state ─────────────────────────────────
        filtered = np.empty_like(window)
        for ch in range(3):
            filtered[:, ch], self.zi[:, :, ch] = signal.sosfilt(
                self.sos, window[:, ch], zi=self.zi[:, :, ch])

        # ── Feature extraction → (33,) ───────────────────────────────────────
        feat_raw = window_to_vector(filtered)

        # Guard against degenerate windows (all-zero / NaN / Inf segments)
        if not np.all(np.isfinite(feat_raw)):
            feat_raw = np.where(np.isfinite(feat_raw), feat_raw, 0.0)

        # ── Scale (transform only — scaler was fitted on training blocks) ────
        feat_scaled = self.scaler.transform(
            feat_raw.reshape(1, -1))[0].astype(np.float32)

        self.feat_buffer.append(feat_scaled)

        if len(self.feat_buffer) < self.time_steps:
            return None   # still warming up

        # ── Build sequence tensor and run inference ──────────────────────────
        seq = np.stack(list(self.feat_buffer), axis=0)            # (T, 33)
        x_t = torch.tensor(seq, dtype=torch.float32) \
                    .unsqueeze(0).to(self.device)                  # (1, T, 33)

        with torch.no_grad():
            hi_raw = float(self.model(x_t).item())

        hi_raw = float(np.clip(hi_raw, 0.0, 1.0))

        # ── Rolling-median smoothing — spike-immune ──────────────────────────
        # Append raw prediction and take the median of the current window.
        # Until the buffer is full, min_periods=1 behaviour applies naturally
        # (median of however many values are available so far).
        self._hi_raw_buf.append(hi_raw)
        hi = float(np.clip(np.median(self._hi_raw_buf), 0.0, 1.0))
        rul_frac = hi_to_rul_fraction(hi)

        return {
            'window_idx':   self.window_count,
            'hi_raw':       hi_raw,
            'hi':           hi,
            'rul_fraction': rul_frac,
            'rul_pct':      rul_frac * 100.0,
            'status':       _health_label(hi),
        }


# ── Display helper ────────────────────────────────────────────────────────────
def _health_label(hi: float) -> str:
    if hi >= 0.85:
        return "HEALTHY"
    elif hi >= 0.60:
        return "EARLY DEGRADATION"
    elif hi >= 0.35:
        return "MODERATE DEGRADATION"
    else:
        return "CRITICAL — REPLACE SOON"


# ── Data source iterators ─────────────────────────────────────────────────────
def _parse_stream(stream):
    """
    Yields (x, y, z) from a line-oriented text stream.
    Accepted formats:
      x,y,z
      timestamp,x,y,z
    Header / comment lines (non-numeric first field) are silently skipped.
    """
    for raw in stream:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        try:
            if len(parts) == 3:
                yield float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) >= 4:
                # timestamp in first column
                yield float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue   # skip header rows or malformed lines


def _iter_serial(port: str, baud: int):
    try:
        import serial
    except ImportError:
        sys.exit("pyserial not installed — run:  pip install pyserial")
    with serial.Serial(port, baud, timeout=5) as ser:
        yield from _parse_stream(
            ser.readline().decode('utf-8', errors='ignore')
            for _ in iter(int, 1)
        )


def _iter_file(path: str):
    with open(path) as fh:
        yield from _parse_stream(fh)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Online ADXL345 RUL estimator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model',      default='model_adxl.pt',
                    help='Path to model checkpoint')
    ap.add_argument('--scaler',     default='scaler_adxl.pkl',
                    help='Path to fitted StandardScaler')
    ap.add_argument('--mode',       default='stdin',
                    choices=['stdin', 'serial', 'file'],
                    help='Data source')
    ap.add_argument('--port',       default='/dev/ttyUSB0',
                    help='Serial port  (mode=serial)')
    ap.add_argument('--baud',       default=921600, type=int,
                    help='Baud rate   (mode=serial)')
    ap.add_argument('--file',       default=None,
                    help='CSV path    (mode=file)')
    ap.add_argument('--median-window', default=75, type=int,
                    help='Rolling-median window (windows) for HI smoothing. '
                         '50-100 recommended. 1 = off.')
    ap.add_argument('--rul-file', default=None,
                    help='Path to a file where the latest smoothed RUL%% (0-100) '
                         'is written atomically after each inference step.  '
                         'The function block reads this file on every POLL event.')
    args = ap.parse_args()

    estimator = OnlineRULEstimator(
        model_path    = Path(args.model),
        scaler_path   = Path(args.scaler),
        median_window = args.median_window,
    )

    # ── Select data source ────────────────────────────────────────────────────
    if args.mode == 'serial':
        sample_iter = _iter_serial(args.port, args.baud)
    elif args.mode == 'file':
        if not args.file:
            sys.exit('--file is required when --mode=file')
        sample_iter = _iter_file(args.file)
    else:
        sample_iter = _parse_stream(sys.stdin)

    # ── Header ────────────────────────────────────────────────────────────────
    print('═' * 70)
    print(f'  ADXL345 Online RUL Estimator  |  mode={args.mode}')
    print('═' * 70)
    print(f'  {"Window":>7}  │  {"HI (raw)":>8}  │  {"HI (med)":>8}  │  '
          f'{"RUL%":>5}  │  Status')
    print(f'  {"─"*7}  │  {"─"*8}  │  {"─"*8}  │  {"─"*5}  │  {"─"*24}')

    # ── Main loop ─────────────────────────────────────────────────────────────
    warmup_reported = False
    rul_file = args.rul_file   # None if not supplied

    for x, y, z in sample_iter:
        result = estimator.push_sample(x, y, z)

        if result is None:
            # Print warm-up progress once per completed second window
            n_buf = len(estimator.feat_buffer)
            ts    = estimator.time_steps
            if n_buf > 0 and not warmup_reported:
                print(f'  Warming up … {n_buf}/{ts} windows buffered …',
                      end='\r', flush=True)
            if n_buf >= ts:
                warmup_reported = True
            continue

        print(f"  {result['window_idx']:>7}  │  "
              f"{result['hi_raw']:>8.4f}  │  "
              f"{result['hi']:>8.4f}  │  "
              f"{result['rul_pct']:>5.1f}  │  "
              f"{result['status']}")

        # ── Write latest smoothed RUL to file for the function block ──────────
        # Written atomically (rename) so the FB never reads a partial value.
        if rul_file:
            tmp = rul_file + '.tmp'
            try:
                with open(tmp, 'w') as fh:
                    fh.write(f"{result['rul_pct']:.2f}\n{result['status']}\n")
                os.replace(tmp, rul_file)
            except OSError:
                pass   # non-fatal — FB will simply read the previous value


if __name__ == '__main__':
    main()
