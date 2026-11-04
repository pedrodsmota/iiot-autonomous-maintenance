#!/usr/bin/env python3
"""
Online Remaining Useful Life (RUL) Estimator — MPU9250 / CNN-BiLSTM
=====================================================================

Usage:
  # ── Live mode via netcat (sensor Pi → inference Pi over Ethernet) ──────────
  #
  #  On the SENSOR Pi  (runs the C driver, streams CSV rows over the wire):
  #    ./mpu9250                                        # starts writing accel_data.csv
  #    tail -f -n +1 accel_data.csv | nc <INFERENCE_IP> 5555
  #
  #  On the INFERENCE Pi  (receives the stream, feeds it to this script):
  #    nc -l -p 5555 | python rul_estimator.py live \\
  #                        --model model_mpu.pt --scaler scaler_mpu.pkl
  #
  #  The script reads CSV rows from stdin line by line. The optional --skip-header
  #  flag (default: on) discards the first "Sample,Accel_X,..." header line.

  # ── Offline test — single CSV ───────────────────────────────────────────────
  python rul_estimator.py csv --input trial_22.csv \\
                              --model model_mpu.pt --scaler scaler_mpu.pkl

  # ── Offline test — ordered folder of CSVs (trial_1.csv … trial_N.csv) ──────
  python rul_estimator.py folder --input /path/to/csvs \\
                                 --model model_mpu.pt --scaler scaler_mpu.pkl

Pipeline (mirrors training exactly):
  Raw 4 kHz samples (Sample, Accel_X, Accel_Y, Accel_Z)
    → High-pass filter (Butterworth 4th-order, 1 Hz cutoff)
    → resample_poly  (4000 → 2800 Hz,  up=7, down=10)
    → 2800-sample windows (1 second each)
    → 33 features  (11 per axis × 3 axes)
    → StandardScaler
    → CNN-BiLSTM  →  Health Index HI ∈ [0.0, 1.0]
    → RUL estimate
"""

import argparse
import collections
import gc
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from scipy import signal
from scipy.stats import skew, kurtosis

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match training pipeline
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_SR = 4000          # MPU9250 native rate (after C-driver SW decimation)
TARGET_SR   = 2800          # Resampled rate used in training
WINDOW_SIZE = TARGET_SR     # 1-second window = 2800 samples at target rate
HP_CUTOFF   = 1.0           # High-pass cutoff (Hz)
HP_ORDER    = 4             # Butterworth filter order
AXES        = ['Accel_X', 'Accel_Y', 'Accel_Z']
FREQ_BANDS  = [(0, 350), (350, 700), (700, 1050), (1050, 1400)]

# Feature column order — must exactly match extract_features.py / dataset_builder.py
FEAT_KEYS  = ['rms', 'var', 'peak', 'crest_factor', 'skew', 'kurtosis',
              'total_spectral_energy',
              'energy_band_1', 'energy_band_2', 'energy_band_3', 'energy_band_4']
FEAT_COLS  = [f'{ax}_{k}' for ax in AXES for k in FEAT_KEYS]  # 33 columns
N_FEATURES = len(FEAT_COLS)  # 33


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture (identical to train_final_model.py)
# ─────────────────────────────────────────────────────────────────────────────
class CNN_BiLSTM(nn.Module):
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
        self.fc      = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (batch, time_steps, features)
        x = self.relu(self.cnn(x.transpose(1, 2)))   # → (B, C, T)
        x = self.pool(x).transpose(1, 2)              # → (B, T/2, C)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (identical to extract_features.py)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_axis(window: np.ndarray, fs: int = TARGET_SR) -> dict:
    """Extract 11 features from a 1-D window. Matches training code exactly."""
    f = {}
    f['rms']          = float(np.sqrt(np.mean(window ** 2)))
    f['var']          = float(np.var(window))
    f['peak']         = float(np.max(np.abs(window)))
    f['crest_factor'] = f['peak'] / (f['rms'] + 1e-8)
    f['skew']         = float(skew(window))
    f['kurtosis']     = float(kurtosis(window))

    n         = len(window)
    fft_vals  = np.abs(np.fft.rfft(window))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    f['total_spectral_energy'] = float(np.sum(fft_vals ** 2) / n)
    for i, (lo, hi) in enumerate(FREQ_BANDS):
        idx = np.where((fft_freqs >= lo) & (fft_freqs < hi))[0]
        f[f'energy_band_{i + 1}'] = float(np.sum(fft_vals[idx] ** 2) / n)
    return f


def extract_features(window_2800: np.ndarray) -> np.ndarray:
    """
    window_2800 : (2800, 3) float32 — HP-filtered & resampled.
    Returns     : (33,)  float64 feature vector in the exact column order
                  used by dataset_builder.py / StandardScaler.
    """
    row = {}
    for col_idx, ax in enumerate(AXES):
        feats = _extract_axis(window_2800[:, col_idx], fs=TARGET_SR)
        for k, v in feats.items():
            row[f'{ax}_{k}'] = v
    return np.array([row[c] for c in FEAT_COLS], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
_GCD  = math.gcd(TARGET_SR, ORIGINAL_SR)
_UP   = TARGET_SR  // _GCD   # 7
_DOWN = ORIGINAL_SR // _GCD  # 10


def make_hp_sos():
    """SOS coefficients for online (causal) filtering."""
    nyq = 0.5 * ORIGINAL_SR
    return signal.butter(HP_ORDER, HP_CUTOFF / nyq, btype='high', output='sos')


def stream_windows_from_csv(
        csv_path: Path,
        sos: np.ndarray,
        zi: np.ndarray,
        read_chunk: int = ORIGINAL_SR,
):
    """
    RAM-safe streaming preprocessor.

    Reads csv_path in chunks of read_chunk rows (default 4000 = 1 s of raw
    data), applies causal sosfilt with state carried across chunks, resamples
    to TARGET_SR, and yields one complete non-overlapping 2800-sample window
    at a time together with the updated filter state.

    Peak memory per iteration: O(read_chunk) rows -- a few KB, not hundreds of MB.

    Yields
    ------
    window : (WINDOW_SIZE, 3) float32  -- HP-filtered and resampled
    zi     : updated sosfilt state     -- pass back in when calling for the next file
    """
    carry = np.empty((0, 3), dtype=np.float32)   # resampled tail from last chunk

    reader = pd.read_csv(
        csv_path,
        usecols=['Accel_X', 'Accel_Y', 'Accel_Z'],
        dtype=np.float32,
        chunksize=read_chunk,
    )

    for chunk_df in reader:
        raw = chunk_df.values                              # (<= 4000, 3) float32

        # Causal HP filter -- state zi is carried across chunks and files
        filtered, zi = signal.sosfilt(sos, raw, axis=0, zi=zi)

        # Resample this small chunk: 4000 -> 2800 Hz
        resampled = signal.resample_poly(
            filtered.astype(np.float32), _UP, _DOWN, axis=0
        )

        # Prepend leftover samples from the previous chunk
        if carry.shape[0]:
            resampled = np.concatenate([carry, resampled], axis=0)

        # Yield complete non-overlapping 1-second windows
        n_full = resampled.shape[0] // WINDOW_SIZE
        for w in range(n_full):
            yield resampled[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE], zi

        # Carry the sub-window tail into the next chunk
        used  = n_full * WINDOW_SIZE
        carry = resampled[used:]


# ─────────────────────────────────────────────────────────────────────────────
# RUL Estimator core
# ─────────────────────────────────────────────────────────────────────────────
class RULEstimator:
    """
    Wraps model + scaler + sliding sequence buffer.

    Call push_window(feat_vec) once per 1-second window.
    Returns HI once the warm-up buffer (time_steps windows) is filled.
    """

    def __init__(self, model_path: str, scaler_path: str, device: str = 'cpu', smooth_span: int = 20):
        self.device = torch.device(device)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        hp   = ckpt['hparams']
        self.time_steps = ckpt['time_steps']
        self.input_dim  = ckpt['input_dim']

        self.model = CNN_BiLSTM(
            input_dim   = self.input_dim,
            cnn_filters = hp['cnn_filters'],
            cnn_kernel  = hp['cnn_kernel'],
            lstm_hidden = hp['lstm_hidden'],
            lstm_layers = hp['lstm_layers'],
            dropout     = hp['dropout_rate'],
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.seq_buf     = collections.deque(maxlen=self.time_steps)
        self.hi_history  = []   # raw model outputs
        self.hi_smooth   = []   # EMA-smoothed (used for display and RUL)
        self.window_idx  = 0
        # EMA: alpha = 2/(span+1). span=1 disables smoothing (alpha=1.0).
        self._ema_alpha  = 2.0 / (max(smooth_span, 1) + 1)
        self._ema_val    = None  # seeded on first prediction

        print(f"\n{'─'*60}")
        print(f"  Model loaded: {Path(model_path).name}")
        print(f"  Input features : {self.input_dim}")
        print(f"  Sequence length: {self.time_steps} windows  ({self.time_steps}s warm-up)")
        print(f"  Architecture   : CNN({hp['cnn_filters']}f k={hp['cnn_kernel']}) + "
              f"BiLSTM({hp['lstm_hidden']}h {hp['lstm_layers']}L)")
        print(f"  Device         : {self.device}")
        print(f"  EMA smooth span: {smooth_span}s  (alpha={self._ema_alpha:.3f})")
        print(f"{'─'*60}\n")

    def push_window(self, feat_vec: np.ndarray) -> float | None:
        """
        Push one 33-feature vector.
        Returns HI ∈ [0.0, 1.0] once warmed up, else None.
        """
        scaled = self.scaler.transform(feat_vec.reshape(1, -1)).astype(np.float32).ravel()
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)

        self.seq_buf.append(scaled)
        self.window_idx += 1

        if len(self.seq_buf) < self.time_steps:
            return None  # Still warming up

        seq = np.stack(list(self.seq_buf), axis=0)            # (T, F)
        x   = torch.tensor(seq, dtype=torch.float32) \
                    .unsqueeze(0).to(self.device)              # (1, T, F)

        with torch.no_grad():
            hi = float(self.model(x).cpu().item())
        hi = float(np.clip(hi, 0.0, 1.0))
        self.hi_history.append(hi)
        # EMA smoothing — seed with first value to avoid initial step
        if self._ema_val is None:
            self._ema_val = hi
        else:
            self._ema_val = self._ema_alpha * hi + (1.0 - self._ema_alpha) * self._ema_val
        hi_s = float(np.clip(self._ema_val, 0.0, 1.0))
        self.hi_smooth.append(hi_s)
        return hi_s  # callers always receive the smoothed value

    # ── Derived metrics ───────────────────────────────────────────────────────

    @staticmethod
    def hi_to_stage(hi: float) -> str:
        if   hi > 0.85: return "Stage 0 — Healthy"
        elif hi > 0.60: return "Stage 1 — Early Degradation"
        elif hi > 0.30: return "Stage 2 — Moderate Degradation"
        else:           return "Stage 3 — Critical / Imminent Failure"

    def estimate_rul(self, hi: float, slope_window: int = 60) -> str:
        """
        Linearly extrapolate recent HI trend to estimate windows until HI→0.
        Each window = 1 second of data.
        """
        if len(self.hi_smooth) < slope_window:
            return "N/A (warming up)"
        recent = self.hi_smooth[-slope_window:]
        # Fit slope via least squares
        xs = np.arange(len(recent), dtype=np.float64)
        slope = float(np.polyfit(xs, recent, 1)[0])  # dHI / window

        if slope >= -1e-6:
            return "Stable / not degrading"
        windows_left = int(hi / abs(slope))
        mins = windows_left // 60
        secs = windows_left % 60
        return f"≈ {mins}m {secs:02d}s  ({windows_left} windows)"

    # ── Printing ──────────────────────────────────────────────────────────────

    def print_status(self, hi: float, window_idx: int, elapsed_sec: float):
        bar_len   = 40
        filled    = int(hi * bar_len)
        bar       = '█' * filled + '░' * (bar_len - filled)
        stage     = self.hi_to_stage(hi)
        rul       = self.estimate_rul(hi)

        # Colour codes (works on most terminals)
        colour = '\033[92m' if hi > 0.60 else ('\033[93m' if hi > 0.30 else '\033[91m')
        reset  = '\033[0m'

        raw_hi = self.hi_history[-1] if self.hi_history else hi
        print(f"\r[W{window_idx:>5} | t={elapsed_sec:>7.1f}s]  "
              f"HI: {colour}{bar}{reset}  {hi:.4f} (raw {raw_hi:.4f})  "
              f"|  {stage:35s}  |  RUL {rul}",
              end='', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV reading helpers
# ─────────────────────────────────────────────────────────────────────────────
def sort_trial_key(p: Path):
    m = re.search(r'trial_(\d+)', p.name)
    return int(m.group(1)) if m else 9999


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────

def _run_streaming(
        csvs: list,
        estimator: "RULEstimator",
        label: str,
        carry_zi_across_files: bool,
):
    """
    Core streaming loop shared by csv and folder modes.

    Reads each CSV in csvs one chunk at a time via stream_windows_from_csv,
    so peak RAM is O(4000 rows) regardless of file size.

    carry_zi_across_files=True  -> filter state is preserved across files
                                   (folder mode: treats all trials as one
                                   continuous recording)
    carry_zi_across_files=False -> filter state resets between files
                                   (csv mode: single isolated file)
    """
    sos = make_hp_sos()
    zi  = np.zeros((sos.shape[0], 2, 3), dtype=np.float64)  # (n_sections, 2, n_channels) -- scipy axis=0 convention

    total_window_idx = 0
    t0 = time.perf_counter()

    print(f"\n  {label}")
    print(f"  Streaming chunk-by-chunk -- peak RAM: O(4000 rows) per iteration")
    print(f"  Warm-up: {estimator.time_steps} windows\n")

    for file_num, csv_path in enumerate(csvs, start=1):
        if len(csvs) > 1:
            print(f"\n  [{file_num:>2}/{len(csvs)}] {csv_path.name}", flush=True)

        if not carry_zi_across_files:
            # Reset filter state for an isolated single-file run
            zi = np.zeros((sos.shape[0], 2, 3), dtype=np.float64)

        for window, zi in stream_windows_from_csv(csv_path, sos, zi):
            feat = extract_features(window)        # (33,) float64
            hi   = estimator.push_window(feat)
            total_window_idx += 1
            elapsed = time.perf_counter() - t0

            if hi is None:
                pct = int(len(estimator.seq_buf) / estimator.time_steps * 100)
                print(f"\r  [W{total_window_idx:>5}]  Warming up ... {pct:3d}%  "
                      f"({len(estimator.seq_buf)}/{estimator.time_steps} windows)",
                      end='', flush=True)
            else:
                estimator.print_status(hi, total_window_idx, elapsed)

    wall = time.perf_counter() - t0
    print(f"\n\n  Done -- {total_window_idx} windows in {wall:.1f}s  "
          f"({total_window_idx / max(wall, 1e-6):.0f}x real-time)")
    if estimator.hi_history:
        final_hi = estimator.hi_history[-1]
        min_hi   = min(estimator.hi_history)
        print(f"  Final HI  : {final_hi:.4f}  ({estimator.hi_to_stage(final_hi)})")
        print(f"  Minimum HI: {min_hi:.4f}")


def run_csv_mode(args, estimator: "RULEstimator"):
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")
    _run_streaming([path], estimator,
                   label=f"Test mode -- single CSV: {path}",
                   carry_zi_across_files=False)


def run_folder_mode(args, estimator: "RULEstimator"):
    folder = Path(args.input)
    if not folder.is_dir():
        sys.exit(f"ERROR: not a directory: {folder}")
    csvs = sorted(folder.glob("trial_*.csv"), key=sort_trial_key)
    if not csvs:
        sys.exit(f"ERROR: no trial_*.csv files found in {folder}")
    _run_streaming(csvs, estimator,
                   label=f"Folder mode -- {len(csvs)} files in {folder}",
                   carry_zi_across_files=True)


def run_live_mode(args, estimator: RULEstimator, rul_file: str | None = None):
    """
    Read CSV rows from stdin in real-time.

    Intended pipeline:
      Sensor Pi :  tail -f -n +1 accel_data.csv | nc <INFERENCE_IP> 5555
      Infer  Pi :  nc -l -p 5555 | python rul_estimator.py live ...

    Each stdin line is expected to be in the format written by the C driver:
      <sample_int>,<Accel_X_float>,<Accel_Y_float>,<Accel_Z_float>

    Uses causal sosfilt so no look-ahead is needed — suitable for true
    real-time operation. Accumulates exactly ORIGINAL_SR (4000) raw samples
    per processing chunk (= 1 second), then resamples to TARGET_SR (2800)
    and extracts features.
    """
    print(f"\nLive mode — reading from stdin")
    print(f"  Sensor Pi command : tail -f -n +1 accel_data.csv | nc <THIS_IP> 5555")
    print(f"  This Pi command   : nc -l -p 5555 | python {sys.argv[0]} live ...")
    print(f"  (Waiting for data. Press Ctrl-C to stop.)\n")

    sos = make_hp_sos()
    # sosfilt state: (n_sections, n_channels=3, 2)
    zi  = np.zeros((sos.shape[0], 2, 3), dtype=np.float64)

    # We accumulate raw 4 kHz samples in a plain list.
    # Once we have CHUNK_SIZE samples we process them as one 1-second block.
    # Any remainder (from the resampled output not filling a full window) is
    # carried into the next chunk via `resample_carry`.
    CHUNK_SIZE    = ORIGINAL_SR       # 4000 raw samples = 1 second
    raw_buf       = []                # list of [x, y, z] float rows
    resample_carry = np.empty((0, 3), dtype=np.float32)  # leftover resampled samples
    header_skipped = not args.skip_header  # if flag off, skip nothing
    window_idx    = 0
    lines_read    = 0
    t0            = time.perf_counter()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        # ── Header detection ────────────────────────────────────────────────
        if not header_skipped:
            # The C driver writes "Sample,Accel_X,Accel_Y,Accel_Z" as the first line.
            # tail -f -n +1 forwards it verbatim. We discard it once.
            if line.lower().startswith('sample') or not line[0].lstrip('-').isdigit():
                header_skipped = True
                continue
            header_skipped = True   # first line was already data — fall through

        # ── Parse one sample ────────────────────────────────────────────────
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue

        raw_buf.append([x, y, z])
        lines_read += 1

        # ── Process every full 1-second chunk ───────────────────────────────
        if len(raw_buf) < CHUNK_SIZE:
            # Print a lightweight progress tick so the user sees data arriving
            if lines_read % 400 == 0:
                elapsed = time.perf_counter() - t0
                print(f"\r  [LIVE | t={elapsed:>7.1f}s]  "
                      f"Buffering … {len(raw_buf)}/{CHUNK_SIZE} samples",
                      end='', flush=True)
            continue

        # Consume exactly CHUNK_SIZE samples; keep any overflow for next round
        chunk          = np.array(raw_buf[:CHUNK_SIZE], dtype=np.float32)
        raw_buf        = raw_buf[CHUNK_SIZE:]

        # Causal HP-filter + resample — carries filter state across chunks
        filtered, zi   = signal.sosfilt(sos, chunk, axis=0, zi=zi)
        resampled      = signal.resample_poly(filtered.astype(np.float32), _UP, _DOWN, axis=0)

        # Prepend any leftover samples from the previous chunk
        if len(resample_carry) > 0:
            resampled = np.concatenate([resample_carry, resampled], axis=0)

        # Slide non-overlapping 1-second windows through the resampled data
        n_full = len(resampled) // WINDOW_SIZE
        for w in range(n_full):
            s   = w * WINDOW_SIZE
            win = resampled[s: s + WINDOW_SIZE]           # (2800, 3)
            feat = extract_features(win)                  # (33,)
            hi   = estimator.push_window(feat)
            window_idx += 1
            elapsed = time.perf_counter() - t0

            if hi is None:
                pct = int(len(estimator.seq_buf) / estimator.time_steps * 100)
                print(f"\r  [LIVE | t={elapsed:>7.1f}s]  "
                      f"Warming up … {pct:3d}%  "
                      f"({len(estimator.seq_buf)}/{estimator.time_steps} windows)",
                      end='', flush=True)
            else:
                estimator.print_status(hi, window_idx, elapsed)
                if rul_file:
                    try:
                        hi_pct = round(hi * 100, 4)
                        stage  = estimator.hi_to_stage(hi)
                        with open(rul_file, 'w') as _fh:
                            _fh.write(f"{hi_pct}\nOK\n")
                    except OSError:
                        pass

        # Keep the tail that didn't fill a full window
        used            = n_full * WINDOW_SIZE
        resample_carry  = resampled[used:]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Online RUL Estimator — MPU9250 / CNN-BiLSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode — sensor Pi streams, inference Pi receives via nc
  #   Sensor Pi:    tail -f -n +1 accel_data.csv | nc <INFERENCE_IP> 5555
  #   Inference Pi: nc -l -p 5555 | python rul_estimator.py --mode live \\
  #                     --model model_mpu.pt --scaler scaler_mpu.pkl

  # If the stream contains no header line (raw data only):
  #   nc -l -p 5555 | python rul_estimator.py --mode live --no-skip-header \\
  #                     --model model_mpu.pt --scaler scaler_mpu.pkl

  # Test with a single trial CSV
  python rul_estimator.py --mode csv --input trial_22.csv \\
                          --model model_mpu.pt --scaler scaler_mpu.pkl

  # Test with a folder of ordered trial CSVs (trial_1.csv … trial_N.csv)
  python rul_estimator.py --mode folder --input /path/to/csvs \\
                          --model model_mpu.pt --scaler scaler_mpu.pkl
        """
    )

    p.add_argument('--mode', default='live', choices=['live', 'csv', 'folder'],
                   help='Operation mode: live (stdin via nc), csv, or folder (default: live)')
    p.add_argument('--model',  required=True, help='Path to model_mpu.pt')
    p.add_argument('--scaler', required=True, help='Path to scaler_mpu.pkl')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                   help='PyTorch device (default: cpu)')
    p.add_argument('--smooth', type=int, default=20, metavar='N',
                   help='EMA smoothing span in seconds (default: 20; use 1 to disable)')
    # live-mode options
    p.add_argument('--no-skip-header', dest='skip_header',
                   action='store_false', default=True,
                   help='Disable automatic header-line skipping '
                        '(use if the stream has no CSV header)')
    p.add_argument('--rul-file', type=str, default=None, metavar='FILE',
                   help=('Path to a file where the latest HI smooth estimate will be '
                         'written after every prediction (line 1: HI%% as float, '
                         'line 2: status string). Intended for external pollers such '
                         'as a function block.'))
    # csv / folder options
    p.add_argument('--input', default=None,
                   help='Path to a trial CSV file (csv mode) or folder (folder mode)')

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ('csv', 'folder') and not args.input:
        import sys as _sys
        _sys.exit(f"ERROR: --input is required for --mode {args.mode}")

    estimator = RULEstimator(
        model_path  = args.model,
        scaler_path = args.scaler,
        device      = args.device,
        smooth_span = args.smooth,
    )

    try:
        if args.mode == 'live':
            run_live_mode(args, estimator, rul_file=args.rul_file)
        elif args.mode == 'csv':
            run_csv_mode(args, estimator)
        elif args.mode == 'folder':
            run_folder_mode(args, estimator)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        if estimator.hi_history:
            final_hi = estimator.hi_history[-1]
            print(f"Last HI: {final_hi:.4f}  —  {estimator.hi_to_stage(final_hi)}")
            print(f"RUL estimate: {estimator.estimate_rul(final_hi)}")


if __name__ == '__main__':
    main()
