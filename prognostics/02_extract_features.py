"""
Step 2 - Extract eleven features from each one-second window of each channel
(dissertation, Section 5.4.1).

Reads   data/synchronized/<sensor>/trial_*.csv
Writes  data/features/<sensor>_features.csv

Features per window and channel
  time domain:      RMS, variance, peak, crest factor, skewness, kurtosis
  frequency domain: total spectral energy and the energy in four 350 Hz bands
                    up to the Nyquist frequency of 1400 Hz (real FFT)
This gives 33 features per window for the three-axis ADXL345 and MPU9250, and
11 for the single-channel piezoelectric sensor.
"""

import gc
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from pipeline_common import FEAT_DIR, FS, SYNC_DIR


def sort_key(filepath):
    match = re.search(r"trial_(\d+)(?:_(\d+))?\.csv", filepath.name)
    if match:
        return (int(match.group(1)), int(match.group(2)) if match.group(2) else 0)
    return (999, 0)


def extract_features_from_window(window_data, fs=FS):
    features = {}
    features["rms"] = np.sqrt(np.mean(window_data ** 2))
    features["var"] = np.var(window_data)
    features["peak"] = np.max(np.abs(window_data))
    features["crest_factor"] = features["peak"] / (features["rms"] + 1e-8)
    features["skew"] = skew(window_data)
    features["kurtosis"] = kurtosis(window_data)

    n = len(window_data)
    fft_vals = np.abs(np.fft.rfft(window_data))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    features["total_spectral_energy"] = np.sum(fft_vals ** 2) / n
    bands = [(0, 350), (350, 700), (700, 1050), (1050, 1400)]
    for i, (low, high) in enumerate(bands):
        idx = np.where((fft_freqs >= low) & (fft_freqs < high))[0]
        features[f"energy_band_{i + 1}"] = np.sum(fft_vals[idx] ** 2) / n
    return features


def process_sensor_features(sensor_name, input_dir, output_dir, columns, window_size=FS):
    print(f"Extracting features for {sensor_name}...")
    files = sorted(input_dir.glob("trial_*.csv"), key=sort_key)
    rows = []
    for f in files:
        print(f"  Reading {f.name}...")
        df = pd.read_csv(f, usecols=columns)
        n_windows = len(df) // window_size
        for w in range(n_windows):
            s, e = w * window_size, (w + 1) * window_size
            feat = {"trial": f.name, "window_idx": w, "time_start_sec": w * (window_size / FS)}
            for col in columns:
                for k, v in extract_features_from_window(
                        df[col].iloc[s:e].values.astype(np.float32)).items():
                    feat[f"{col}_{k}"] = v
            rows.append(feat)
        del df
        gc.collect()
    if rows:
        out = output_dir / f"{sensor_name}_features.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {len(rows)} feature vectors to {out}")


if __name__ == "__main__":
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    if (SYNC_DIR / "piezo").exists():
        process_sensor_features("piezo", SYNC_DIR / "piezo", FEAT_DIR, ["Vibration_Val"])
    if (SYNC_DIR / "adxl345").exists():
        process_sensor_features("adxl345", SYNC_DIR / "adxl345", FEAT_DIR, ["x", "y", "z"])
    if (SYNC_DIR / "mpu9250").exists():
        process_sensor_features("mpu9250", SYNC_DIR / "mpu9250", FEAT_DIR,
                                ["Accel_X", "Accel_Y", "Accel_Z"])
    print("Feature extraction complete.")
