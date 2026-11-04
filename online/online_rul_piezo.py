#!/usr/bin/env python3
"""
online_rul_piezo.py — Real-time Remaining Useful Life estimation for the
                       piezo vibration sensor (ARM2).

Pipeline (mirrors the offline training pipeline exactly):

  Pico USB Serial
       │
       ▼ raw ADC integers (0-65535) + PC timestamps
       │
       ▼ × SENSOR_CALIBRATION_MULTIPLIER          ← sensor-swap correction
         (piezo_data_repair.py computed std(trial_14)/std(trial_15) and
          scaled trials 15-22 before feature extraction. The scaler and
          model were therefore trained on calibrated amplitudes. Apply the
          same multiplier here so live data is on the same scale.)
       │
  1-second accumulation window
       │
       ▼ scipy.interpolate → uniform 2800 Hz grid
  4th-order Butterworth high-pass filter @ 1 Hz
       │
       ▼ same 11 features as extract_features.py
  StandardScaler  (scaler_piezo.pkl — fit on training data)
       │
       ▼ rolling deque of 60 feature vectors  (TIME_STEPS from dataset_builder)
  CNN-BiLSTM  (model_piezo.pt)
       │
       ▼ Health Index  HI ∈ [0, 1]   (1 = healthy, 0 = failed)
  RUL estimate  via linear regression on recent HI history

Usage:
    # Live sensor
    python online_rul_piezo.py --multiplier 1.8342
    python online_rul_piezo.py --multiplier 1.8342 --plot
    python online_rul_piezo.py --port /dev/ttyACM1 --multiplier 1.8342

    # Replay a single recorded CSV
    python online_rul_piezo.py --csv path/to/piezo_data.csv

    # Replay an entire folder of trial CSVs (sorted by trial number)
    python online_rul_piezo.py --csv-dir path/to/ARM2_piezo/
    python online_rul_piezo.py --csv-dir path/to/ARM2_piezo/ --plot

    Files are streamed one row at a time — RAM stays flat no matter how large
    the folder is.  --csv and --csv-dir are mutually exclusive.
"""

import argparse
import csv
import sys
import time
import threading
import queue
import pickle
import signal
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis

# ─── Optional live plot ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('TkAgg')          # works headless-free; falls back silently
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed — run:  pip install pyserial")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration (must match training pipeline)
# ══════════════════════════════════════════════════════════════════════════════

SERIAL_PORT    = '/dev/ttyACM0'
BAUD_RATE      = 115200
TARGET_SR      = 2800            # Hz — sync_data.py target_sr
WINDOW_SEC     = 1.0             # seconds per feature window
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SEC)   # 2800 samples
HP_CUTOFF_HZ   = 1.0            # high-pass filter cutoff
HP_ORDER       = 4              # Butterworth order
TIME_STEPS     = 60             # look-back sequence length (dataset_builder.py)
MODEL_PATH     = Path('model_reconstruction/results/model_piezo.pt')
SCALER_PATH    = Path('model_reconstruction/scaler_piezo.pkl')

# ── Sensor-swap calibration ────────────────────────────────────────────────────
# piezo_data_repair.py scaled trials 15-22 by std(trial_14)/std(trial_15) so
# all training data is on a single amplitude reference (the original sensor).
# The scaler and model therefore expect calibrated amplitudes.
# Set this default to the "Calibration multiplier" printed by that script.
# Can also be overridden at runtime via --multiplier.
# If you are using the ORIGINAL sensor (trials 1-14), use 1.0.
#SENSOR_CALIBRATION_MULTIPLIER: float = 1.9169561438   # ← replace with your printed value
SENSOR_CALIBRATION_MULTIPLIER: float = 1.0

# ── Post-prediction smoothing ─────────────────────────────────────────────────
# Raw model outputs are noisy (one prediction per 1-second window).
# We apply the same two-stage filter construct_hs_hi.py used on training data:
#   1. Rolling median  — kills transient spikes completely
#   2. EMA             — smooths the remaining trend
# Then we enforce monotonicity: smoothed HI can only decrease (bearings don't heal).
HI_MEDIAN_WINDOW = 30           # rolling median over last N raw predictions
HI_EMA_SPAN      = 60           # EMA span in predictions (alpha = 2/(span+1))
HI_EMA_ALPHA     = 2.0 / (HI_EMA_SPAN + 1)
MONOTONE_WARMUP  = HI_MEDIAN_WINDOW   # don't enforce cummin until we have enough history

# RUL trend estimation
RUL_TREND_WINDOW = 7200          # use last 3600 smoothed predictions (~60 minutes of trend)
                                # longer window → regression slope is stable

# Frequency bands for spectral energy features (Hz) — must match extract_features.py
FREQ_BANDS = [(0, 350), (350, 700), (700, 1050), (1050, 1400)]


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture (identical to train_final_model.py / tune_hyperparameters.py)
# ══════════════════════════════════════════════════════════════════════════════

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
        x = x.transpose(1, 2)             # → (batch, features, time_steps)
        x = self.relu(self.cnn(x))
        x = self.pool(x)
        x = x.transpose(1, 2)             # → (batch, time_steps/2, cnn_filters)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (mirrors extract_features.py → extract_features_from_window)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(window: np.ndarray, fs: int = TARGET_SR) -> np.ndarray:
    """
    Extract the same 11 features used during training from a 1D float32 array.

    Returns a 1D numpy array of length 11 in this exact order:
        rms, var, peak, crest_factor, skew, kurtosis,
        total_spectral_energy,
        energy_band_1, energy_band_2, energy_band_3, energy_band_4
    """
    # ── Time-domain ──────────────────────────────────────────────────────────
    rms          = float(np.sqrt(np.mean(window ** 2)))
    var          = float(np.var(window))
    peak         = float(np.max(np.abs(window)))
    crest_factor = peak / (rms + 1e-8)
    sk           = float(skew(window))
    kurt         = float(kurtosis(window))

    # ── Frequency-domain (rFFT) ───────────────────────────────────────────────
    n         = len(window)
    fft_vals  = np.abs(np.fft.rfft(window))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total_spectral_energy = float(np.sum(fft_vals ** 2) / n)

    band_energies = []
    for low, high in FREQ_BANDS:
        idx = np.where((fft_freqs >= low) & (fft_freqs < high))[0]
        band_energies.append(float(np.sum(fft_vals[idx] ** 2) / n))

    return np.array(
        [rms, var, peak, crest_factor, sk, kurt, total_spectral_energy] + band_energies,
        dtype=np.float32
    )


# ══════════════════════════════════════════════════════════════════════════════
# High-pass filter (mirrors sync_data.py → process_file_piezo)
# ══════════════════════════════════════════════════════════════════════════════

def build_hp_filter(fs: int = TARGET_SR, cutoff: float = HP_CUTOFF_HZ,
                    order: int = HP_ORDER):
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, cutoff / nyq, btype='high')
    return b, a


# ══════════════════════════════════════════════════════════════════════════════
# RUL estimation from HI history
# ══════════════════════════════════════════════════════════════════════════════

def estimate_rul(hi_history: list[float], window_sec: float = WINDOW_SEC) -> tuple[float, str]:
    """
    Fit a linear trend to the most recent HI values and extrapolate to HI = 0.

    Returns:
        rul_seconds  — estimated seconds to failure (or inf if trending upward)
        rul_label    — human-readable string (e.g. "14m 23s", "Stable", "Imminent")
    """
    n = len(hi_history)
    if n < 3:
        return float('inf'), "Collecting data…"

    use = hi_history[-RUL_TREND_WINDOW:]
    t   = np.arange(len(use), dtype=np.float32) * window_sec   # seconds

    # Linear regression: HI = slope * t + intercept
    slope, intercept = np.polyfit(t, use, 1)

    # HI(t*) = 0  ⟹  t* = -intercept / slope
    if slope >= 0:
        return float('inf'), "Stable / Recovering"

    t_failure    = -intercept / slope          # seconds from start of 'use' window
    t_now        = t[-1]                       # seconds elapsed in 'use' window
    rul_seconds  = max(t_failure - t_now, 0.0)

    if rul_seconds < 60:
        return rul_seconds, f"⚠  {rul_seconds:.0f}s  — IMMINENT FAILURE"

    minutes, secs = divmod(int(rul_seconds), 60)
    hours, mins   = divmod(minutes, 60)
    if hours > 0:
        label = f"{hours}h {mins}m"
    else:
        label = f"{mins}m {secs:02d}s"

    return rul_seconds, label


# ══════════════════════════════════════════════════════════════════════════════
# Serial reader thread
# ══════════════════════════════════════════════════════════════════════════════

class SerialReader(threading.Thread):
    """
    Background thread: reads raw integer lines from the Pico, timestamps them
    on the PC, and pushes (timestamp, value) tuples into a queue.
    """

    def __init__(self, port: str, baud: int,
                 sample_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.port         = port
        self.baud         = baud
        self.sample_queue = sample_queue
        self.stop_event   = stop_event
        self.error: str | None = None

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as exc:
            self.error = str(exc)
            return

        ser.reset_input_buffer()
        t0 = time.perf_counter()

        while not self.stop_event.is_set():
            try:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
            except serial.SerialException:
                self.error = "Serial port disconnected."
                break

            if raw.isdigit():
                ts  = time.perf_counter() - t0
                val = int(raw)
                self.sample_queue.put((ts, val))

        ser.close()


class CSVReader(threading.Thread):
    """
    Background thread: replays a recorded piezo CSV file (columns: Time_Sec,
    Vibration_Val) into the same sample_queue used by SerialReader.

    Memory strategy: the file is opened once and iterated one row at a time
    using csv.reader with a small OS read buffer (buffering=65536 bytes).
    Python never accumulates the full file in RAM — only one row is live at
    any moment. The queue's maxsize acts as natural backpressure, so this
    thread blocks cheaply whenever the main loop falls behind rather than
    racing ahead and allocating unbounded memory.
    """

    # How many bytes the OS read buffer is allowed to hold for this file.
    # 64 KiB is large enough for throughput but tiny relative to the files.
    READ_BUFFER = 65_536

    def __init__(self, csv_path: Path, sample_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.csv_path     = csv_path
        self.sample_queue = sample_queue
        self.stop_event   = stop_event
        self.error: str | None = None
        self.rows_read    = 0

    def run(self):
        try:
            # buffering= controls the OS-level read buffer size, NOT whether
            # the whole file is read; the file object stays lazy.
            with self.csv_path.open("r", newline="", encoding="utf-8",
                                    buffering=self.READ_BUFFER) as fh:
                reader = csv.reader(fh)

                header = next(reader, None)
                if header is None:
                    self.error = f"Empty CSV file: {self.csv_path}"
                    return

                # Locate columns by name (case-insensitive, strip whitespace)
                header_clean = [h.strip().lower() for h in header]
                try:
                    t_col = header_clean.index("time")
                    v_col = header_clean.index("vibration_val")
                except ValueError:
                    self.error = (
                        f"CSV must have 'Time_Sec' and 'Vibration_Val' columns. "
                        f"Found: {header}"
                    )
                    return

                for row in reader:
                    if self.stop_event.is_set():
                        break
                    try:
                        ts  = float(row[t_col])
                        val = float(row[v_col])
                    except (ValueError, IndexError):
                        continue   # skip malformed rows silently

                    # Block here if the main loop is busy — this is the
                    # backpressure that keeps memory flat.
                    while not self.stop_event.is_set():
                        try:
                            self.sample_queue.put((ts, val), timeout=0.1)
                            self.rows_read += 1
                            break
                        except queue.Full:
                            continue   # queue full → wait, don't accumulate

        except OSError as exc:
            self.error = str(exc)




class FolderCSVReader(threading.Thread):
    """
    Background thread: streams all trial_*.csv files in a folder in sorted
    trial order (same sort key as extract_features.py) into sample_queue.

    Memory guarantees
    -----------------
    * Only ONE file is open at any moment — previous files are fully closed
      and released before the next one is opened.
    * Each file is read one row at a time with a 64 KiB OS buffer; the
      entire file is never accumulated in RAM.
    * The queue's maxsize provides backpressure: when the main loop is busy
      this thread blocks cheaply instead of racing ahead and buffering rows.

    Timestamp continuity
    --------------------
    Each CSV's Time_Sec column starts near 0.  To give the main loop a
    monotonically increasing timestamp stream across files we keep a running
    offset that is bumped by (last_ts_of_previous_file + 1/TARGET_SR) at
    every file boundary.  The main loop's window logic (raw_ts[-1] - raw_ts[0])
    therefore works identically to the single-file case.
    """

    READ_BUFFER = 65_536   # 64 KiB OS read buffer per file

    def __init__(self, folder: Path, sample_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.folder       = folder
        self.sample_queue = sample_queue
        self.stop_event   = stop_event
        self.error: str | None = None
        self.rows_read    = 0
        self.current_file = ""          # main loop can read this for display
        self.files_done   = 0
        self._files       = self._sorted_trials()

    def _sorted_trials(self) -> list[Path]:
        import re
        pattern = re.compile(r"trial_(\d+)(?:_(\d+))?\.csv$")
        found: list[tuple[int, int, Path]] = []
        for p in self.folder.glob("trial_*.csv"):
            m = pattern.search(p.name)
            if m:
                found.append((int(m.group(1)), int(m.group(2) or 0), p))
        if not found:
            self.error = f"No trial_*.csv files found in {self.folder}"
        found.sort()
        return [p for _, _, p in found]

    def _push(self, ts: float, val: float) -> bool:
        """Put one sample into the queue with backpressure. Returns False if stopped."""
        while not self.stop_event.is_set():
            try:
                self.sample_queue.put((ts, val), timeout=0.1)
                self.rows_read += 1
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        if not self._files:
            return   # error already set in _sorted_trials

        t_offset = 0.0   # cumulative timestamp shift across file boundaries

        for csv_path in self._files:
            if self.stop_event.is_set():
                break

            self.current_file = csv_path.name
            last_ts_in_file   = 0.0

            try:
                with csv_path.open("r", newline="", encoding="utf-8",
                                   buffering=self.READ_BUFFER) as fh:
                    reader = csv.reader(fh)

                    header = next(reader, None)
                    if header is None:
                        self.error = f"Empty file: {csv_path}"
                        return

                    header_clean = [h.strip().lower() for h in header]
                    try:
                        t_col = header_clean.index("time")
                        v_col = header_clean.index("vibration_val")
                    except ValueError:
                        # Try "time_sec" as fallback
                        try:
                            t_col = header_clean.index("time_sec")
                            v_col = header_clean.index("vibration_val")
                        except ValueError:
                            self.error = (
                                f"Cannot find Time_Sec / Vibration_Val columns "
                                f"in {csv_path.name}. Header: {header}"
                            )
                            return

                    for row in reader:
                        if self.stop_event.is_set():
                            break
                        try:
                            ts  = float(row[t_col])
                            val = float(row[v_col])
                        except (ValueError, IndexError):
                            continue

                        last_ts_in_file = ts
                        if not self._push(ts + t_offset, val):
                            return   # stop requested

            except OSError as exc:
                self.error = str(exc)
                return

            # Bump offset so next file's t=0 sits just after this file's last sample
            t_offset += last_ts_in_file + (1.0 / TARGET_SR)
            self.files_done += 1

        self.current_file = "(done)"

# ══════════════════════════════════════════════════════════════════════════════
# Model + scaler loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_scaler(model_path: Path, scaler_path: Path,
                           device: torch.device) -> tuple:
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        sys.exit(f"Scaler file not found: {scaler_path}")

    checkpoint = torch.load(model_path, map_location=device)
    hp         = checkpoint['hparams']

    model = CNN_BiLSTM(
        input_dim   = checkpoint['input_dim'],
        cnn_filters = hp['cnn_filters'],
        cnn_kernel  = hp['cnn_kernel'],
        lstm_hidden = hp['lstm_hidden'],
        lstm_layers = hp['lstm_layers'],
        dropout     = hp['dropout_rate'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"  Model   : {model_path.name}  |  input_dim={checkpoint['input_dim']}, "
          f"time_steps={checkpoint['time_steps']}")
    print(f"  Scaler  : {scaler_path.name}")
    print(f"  Device  : {device}")
    print(f"  Hparams : {hp}")
    return model, scaler, checkpoint['time_steps']


# ══════════════════════════════════════════════════════════════════════════════
# Live plot (optional)
# ══════════════════════════════════════════════════════════════════════════════

class LivePlot:
    """Simple two-panel scrolling plot: HI over time + raw waveform."""

    MAX_POINTS = 300   # ~5 minutes of predictions at 1/s

    def __init__(self):
        self.hi_raw_history: list[float] = []
        self.hi_history:     list[float] = []
        self.rul_history:    list[float] = []
        self._lock = threading.Lock()

        self.fig, (self.ax_hi, self.ax_rul) = plt.subplots(
            2, 1, figsize=(10, 6), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('Online RUL — Piezo')

        for ax in (self.ax_hi, self.ax_rul):
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#e0e0e0')
            for sp in ax.spines.values():
                sp.set_color('#444')

        self.line_hi_raw,    = self.ax_hi.plot([], [], lw=0.8, color='#005f7a',
                                                alpha=0.5, label='raw')
        self.line_hi_smooth, = self.ax_hi.plot([], [], lw=2.0, color='#00d4ff',
                                                label='smoothed (monotone)')
        self.line_rul, = self.ax_rul.plot([], [], lw=1.5, color='#ff6b6b')

        self.ax_hi.set_ylim(-0.05, 1.05)
        self.ax_hi.set_ylabel('Health Index (HI)', color='#e0e0e0')
        self.ax_hi.axhline(0.5, ls='--', lw=0.8, color='#ffaa00', alpha=0.6,
                           label='50% health')
        self.ax_hi.axhline(0.2, ls='--', lw=0.8, color='#ff4444', alpha=0.6,
                           label='Critical threshold')
        self.ax_hi.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                          labelcolor='#e0e0e0')
        self.ax_hi.set_title('Real-time Health Index', color='#e0e0e0', pad=6)
        self.ax_hi.legend(loc='upper right', fontsize=7,
                          facecolor='#1a1a2e', labelcolor='#e0e0e0')

        self.ax_rul.set_ylabel('Estimated RUL (s)', color='#e0e0e0')
        self.ax_rul.set_xlabel('Prediction index', color='#e0e0e0')
        self.ax_rul.set_title('Estimated Remaining Useful Life', color='#e0e0e0', pad=6)

        self.fig.tight_layout(pad=2.0)

    def update(self, hi_raw: float, hi_smooth: float, rul_s: float):
        with self._lock:
            self.hi_raw_history.append(hi_raw)
            self.hi_history.append(hi_smooth)
            self.rul_history.append(min(rul_s, 7200))   # cap at 2 h for display
            if len(self.hi_history) > self.MAX_POINTS:
                self.hi_raw_history.pop(0)
                self.hi_history.pop(0)
                self.rul_history.pop(0)

    def _animate(self, _frame):
        with self._lock:
            hi_raw    = list(self.hi_raw_history)
            hi_smooth = list(self.hi_history)
            rul       = list(self.rul_history)
        xs = list(range(len(hi_smooth)))

        self.line_hi_raw.set_data(list(range(len(hi_raw))), hi_raw)
        self.line_hi_smooth.set_data(xs, hi_smooth)
        self.ax_hi.set_xlim(0, max(len(hi_smooth), 2))

        self.line_rul.set_data(xs, rul)
        self.ax_rul.set_xlim(0, max(len(rul), 2))
        self.ax_rul.set_ylim(0, max(max(rul, default=60) * 1.1, 60))

        return self.line_hi_raw, self.line_hi_smooth, self.line_rul

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig, self._animate, interval=1000, blit=True, cache_frame_data=False)
        plt.show(block=False)


# ══════════════════════════════════════════════════════════════════════════════
# Terminal display helpers
# ══════════════════════════════════════════════════════════════════════════════

BAR_WIDTH = 30

def _hi_bar(hi: float) -> str:
    filled = int(round(hi * BAR_WIDTH))
    bar    = '█' * filled + '░' * (BAR_WIDTH - filled)
    if hi > 0.6:
        color = '\033[92m'    # green
    elif hi > 0.3:
        color = '\033[93m'    # yellow
    else:
        color = '\033[91m'    # red
    return f"{color}[{bar}]\033[0m"


def display(hi_raw: float, hi_smooth: float, rul_s: float, rul_label: str,
            n_windows: int, n_predictions: int, sample_rate: float,
            current_file: str = "", files_done: int = 0, total_files: int = 0):
    sys.stdout.write('\033[2J\033[H')   # clear screen, move cursor home
    sys.stdout.write(
        "┌─────────────────────────────────────────────────────┐\n"
        "│       Online RUL Estimation — Piezo (ARM2)          │\n"
        "└─────────────────────────────────────────────────────┘\n"
        f"\n  Health Index  raw     {_hi_bar(hi_raw)}  {hi_raw:.4f}"
        f"\n  Health Index  smooth  {_hi_bar(hi_smooth)}  {hi_smooth:.4f}"
        f"\n  Remaining Useful Life   :  {rul_label}"
        f"\n  RUL (seconds)           :  {rul_s:.1f}"
        f"\n"
        f"\n  Windows processed       :  {n_windows}"
        f"\n  Predictions made        :  {n_predictions}"
        f"\n  Estimated sample rate   :  {sample_rate:.0f} Hz"
        + (f"\n  File                    :  [{files_done}/{total_files}] {current_file}" if current_file else "")
        + "\n"
        + "\n  Press Ctrl+C to stop."
        + "\n"
    )
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run(port: str, live_plot: bool, calibration_multiplier: float = SENSOR_CALIBRATION_MULTIPLIER,
        csv_path: Path | None = None, csv_dir: Path | None = None,
        model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH,
        rul_file: str | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model and scaler…")
    model, scaler, time_steps = load_model_and_scaler(model_path, scaler_path, device)
    print(f"  TIME_STEPS used for inference: {time_steps}")
    if calibration_multiplier != 1.0:
        print(f"  Sensor calibration multiplier: {calibration_multiplier:.6f}  "
              f"(new sensor → raw ADC values scaled to training amplitude)")
    else:
        print("  Sensor calibration multiplier: 1.0  (original sensor / passthrough)")

    # Build the high-pass filter coefficients once
    b_hp, a_hp = build_hp_filter()

    # Rolling buffer of scaled feature vectors  (shape: [time_steps, n_features])
    feature_queue: deque[np.ndarray] = deque(maxlen=time_steps)

    # Raw HI predictions from the model
    hi_raw_history:    list[float] = []
    # Smoothed, monotone HI used for display and RUL regression
    hi_smooth_history: list[float] = []
    # Running EMA state (initialise at 1.0 = healthy)
    _hi_ema      = 1.0
    _hi_monotone = 1.0

    # Accumulators for the current raw window
    raw_ts:  list[float] = []    # PC timestamps (seconds since start)
    raw_val: list[float] = []    # ADC values (0-65535)

    # Stats
    n_windows     = 0
    n_predictions = 0

    # Set up live plot (optional)
    plotter: LivePlot | None = None
    if live_plot:
        if not MATPLOTLIB_AVAILABLE:
            print("WARNING: matplotlib not available — --plot ignored.")
        else:
            plotter = LivePlot()
            plotter.start()

    # Source reader — serial or CSV
    stop_event   = threading.Event()
    # Queue holds (timestamp_sec, raw_value) tuples.
    # 50 000 entries ≈ ~17 seconds at 2800 Hz — large enough to absorb bursts,
    # small enough that backpressure kicks in before RAM pressure does.
    sample_queue: queue.Queue[tuple[float, float]] = queue.Queue(maxsize=50_000)

    if csv_dir is not None:
        reader: SerialReader | CSVReader | FolderCSVReader = FolderCSVReader(csv_dir, sample_queue, stop_event)
    elif csv_path is not None:
        reader = CSVReader(csv_path, sample_queue, stop_event)
    else:
        reader = SerialReader(port, BAUD_RATE, sample_queue, stop_event)

    def _shutdown(sig, frame):
        print("\nShutting down…")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if csv_dir is not None:
        files = sorted(csv_dir.glob('trial_*.csv'))
        print(f"\nReplaying folder: {csv_dir}")
        print(f"  Found {len(files)} trial file(s) — streaming one at a time, RAM stays flat.\n")
    elif csv_path is not None:
        print(f"\nReplaying CSV: {csv_path}")
        print("(streaming row-by-row — RAM usage stays flat regardless of file size)\n")
    else:
        print(f"\nConnecting to {port} …")
    reader.start()
    if csv_path is None and csv_dir is None:
        time.sleep(1.5)   # let the Pico's 3-blink startup finish

    if reader.error:
        sys.exit(f"Reader error: {reader.error}")

    print("Collecting data…\n")

    # ── Initialise display state ──────────────────────────────────────────────
    hi_display  = 1.0
    rul_display = float('inf')
    rul_label   = "Collecting data…"
    t_start     = time.perf_counter()
    samples_total = 0

    while not stop_event.is_set():
        # ── Drain the queue ───────────────────────────────────────────────────
        drained = 0
        while drained < 2000:
            try:
                ts, val = sample_queue.get_nowait()
                raw_ts.append(ts)
                raw_val.append(float(val) * calibration_multiplier)
                samples_total += 1
                drained += 1
            except queue.Empty:
                break

        if reader.error:
            print(f"\nReader error: {reader.error}")
            stop_event.set()
            break

        # ── Decide if we have a full 1-second window ──────────────────────────
        # Use the data's own timestamps rather than wall-clock time.
        # This is correct for both modes:
        #   serial → timestamps are wall-clock offsets (≈ real elapsed time)
        #   CSV    → timestamps are the original Time_Sec values (data duration)
        # Wall-clock would break CSV mode entirely because the reader can push
        # thousands of samples before a single second of wall time elapses.
        now = time.perf_counter()

        if (len(raw_ts) < (WINDOW_SAMPLES // 2) or
                (raw_ts[-1] - raw_ts[0]) < WINDOW_SEC):
            time.sleep(0.005)
            continue

        # ── Build the uniform 2800 Hz grid for this window ────────────────────
        t_raw  = np.array(raw_ts,  dtype=np.float64)
        v_raw  = np.array(raw_val, dtype=np.float32)

        t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0 / TARGET_SR)
        if len(t_uniform) < 2:
            # Window too short to interpolate — reset
            raw_ts.clear()
            raw_val.clear()
            continue

        v_interp = np.interp(t_uniform, t_raw, v_raw).astype(np.float32)

        # ── High-pass filter  (mirrors process_file_piezo) ────────────────────
        v_filtered = sp_signal.filtfilt(b_hp, a_hp, v_interp).astype(np.float32)

        # ── Extract features ──────────────────────────────────────────────────
        feat_raw = extract_features(v_filtered, fs=TARGET_SR)   # shape: (11,)

        # ── Scale  (same scaler.transform used in dataset_builder.py) ─────────
        feat_scaled = scaler.transform(feat_raw.reshape(1, -1)).astype(np.float32)[0]

        # ── Push into rolling sequence buffer ─────────────────────────────────
        feature_queue.append(feat_scaled)
        n_windows += 1

        # Reset raw accumulators for the next window
        raw_ts.clear()
        raw_val.clear()

        # ── Inference (only once we have a full sequence) ─────────────────────
        if len(feature_queue) == time_steps:
            # Build tensor: (1, time_steps, n_features)
            seq     = np.stack(list(feature_queue), axis=0)           # (T, F)
            x_in    = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)

            with torch.no_grad():
                hi_raw = model(x_in).item()

            hi_raw_val = float(np.clip(hi_raw, 0.0, 1.0))
            hi_raw_history.append(hi_raw_val)
            n_predictions += 1

            # Stage 1 — rolling median (robust spike removal)
            median_window = hi_raw_history[-HI_MEDIAN_WINDOW:]
            hi_median = float(np.median(median_window))

            # Stage 2 — EMA on the median output
            _hi_ema = HI_EMA_ALPHA * hi_median + (1.0 - HI_EMA_ALPHA) * _hi_ema

            # Stage 3 — monotone decreasing enforcement (skip during warmup)
            if len(hi_raw_history) > MONOTONE_WARMUP:
                _hi_monotone = min(_hi_ema, _hi_monotone)
            else:
                _hi_monotone = _hi_ema   # warmup: track freely

            hi_smooth_history.append(_hi_monotone)
            rul_display, rul_label = estimate_rul(hi_smooth_history)

            # Write latest RUL to file so external pollers (e.g. a function block)
            # can read it without coupling to this process's stdout.
            # Format mirrors the ADXL worker:
            #   line 1 — RUL in seconds as a float (-1.0 when not yet estimable)
            #   line 2 — operational status string ("OK", or the label when trending stable/infinite)
            if rul_file:
                try:
                    hi_pct     = round(_hi_monotone * 100, 4)
                    rul_status = rul_label if rul_display == float('inf') else "OK"
                    with open(rul_file, 'w') as _fh:
                        _fh.write(f"{hi_pct}\n{rul_status}\n")
                except OSError:
                    pass

            if plotter:
                plotter.update(hi_raw_val, _hi_monotone, rul_display)

        # ── Update terminal display ───────────────────────────────────────────
        elapsed_wall = time.perf_counter() - t_start
        sr_estimate  = samples_total / max(elapsed_wall, 1e-6)
        _cur_file   = getattr(reader, 'current_file', '')
        _files_done = getattr(reader, 'files_done',   0)
        _total_files= getattr(reader, '_files',       [])
        _hi_raw_disp    = hi_raw_history[-1]    if hi_raw_history    else 1.0
        _hi_smooth_disp = hi_smooth_history[-1] if hi_smooth_history else 1.0
        display(_hi_raw_disp, _hi_smooth_disp, rul_display, rul_label,
                n_windows, n_predictions,
                sr_estimate, _cur_file, _files_done, len(_total_files))

        # In CSV mode run as fast as the CPU allows (no real-time pacing).
        # In serial mode pace to ~1 iteration per second so the terminal
        # is readable and the CPU isn't busy-looping.
        if csv_path is None and csv_dir is None:
            time.sleep(max(0, WINDOW_SEC - (time.perf_counter() - now)))

    # ── Shutdown ──────────────────────────────────────────────────────────────
    stop_event.set()
    reader.join(timeout=3.0)

    if csv_dir is not None and hasattr(reader, 'rows_read'):
        print(f"\nDone.  Folder replay complete: {reader.files_done} file(s), "
              f"{reader.rows_read:,} rows, "
              f"{n_windows} windows, {n_predictions} predictions.")
    elif csv_path is not None and hasattr(reader, 'rows_read'):
        print(f"\nDone.  Read {reader.rows_read:,} CSV rows, "
              f"processed {n_windows} windows, made {n_predictions} predictions.")
    else:
        print(f"\nDone.  Processed {n_windows} windows, made {n_predictions} predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Online RUL estimation for the piezo vibration sensor.')
    parser.add_argument('--port', default=SERIAL_PORT,
                        help=f'Serial port (default: {SERIAL_PORT})')
    parser.add_argument('--plot', action='store_true',
                        help='Show a live matplotlib plot (requires display)')
    parser.add_argument(
        '--multiplier', type=float, default=SENSOR_CALIBRATION_MULTIPLIER,
        metavar='M',
        help=('Sensor calibration multiplier printed by piezo_data_repair.py '
              '(std_trial14 / std_trial15). Use 1.0 for the original sensor. '
              f'Default: {SENSOR_CALIBRATION_MULTIPLIER}'))
    parser.add_argument(
        '--csv', type=Path, default=None, metavar='FILE',
        help=('Path to a single recorded piezo CSV (Time_Sec, Vibration_Val). '
              'Streams row-by-row; RAM stays flat. Mutually exclusive with --csv-dir.'))
    parser.add_argument(
        '--csv-dir', type=Path, default=None, metavar='DIR',
        help=('Path to a folder containing trial_*.csv files. '
              'Files are sorted by trial number and streamed one at a time — '
              'only one file is open at any moment so RAM stays flat even for '
              'very large datasets. Mutually exclusive with --csv.'))
    parser.add_argument(
        '--model', type=Path, default=MODEL_PATH, metavar='FILE',
        help=f'Path to the trained model .pt file (default: {MODEL_PATH})')
    parser.add_argument(
        '--scaler', type=Path, default=SCALER_PATH, metavar='FILE',
        help=f'Path to the fitted scaler .pkl file (default: {SCALER_PATH})')
    parser.add_argument(
        '--rul-file', type=str, default=None, metavar='FILE',
        help=('Path to a file where the latest RUL estimate will be written '
              'after every prediction (line 1: seconds as float, line 2: label). '
              'Intended for external pollers such as a function block.'))
    args = parser.parse_args()

    if args.csv is not None and args.csv_dir is not None:
        sys.exit("--csv and --csv-dir are mutually exclusive.")
    if args.csv is not None and not args.csv.exists():
        sys.exit(f"CSV file not found: {args.csv}")
    if args.csv_dir is not None and not args.csv_dir.is_dir():
        sys.exit(f"Not a directory: {args.csv_dir}")

    run(args.port, args.plot, args.multiplier,
        csv_path=args.csv, csv_dir=args.csv_dir,
        model_path=args.model, scaler_path=args.scaler,
        rul_file=args.rul_file)
