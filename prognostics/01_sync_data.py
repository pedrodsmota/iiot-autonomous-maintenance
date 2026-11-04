"""
Step 1 - Resample every recording to 2800 Hz and apply the high-pass filter
(dissertation, Section 5.4.1).

Reads   data/raw/<sensor folder>/trial_*.parquet|csv and data/metadata/sessions.csv
Writes  data/synchronized/<sensor>/trial_*.csv

Design notes
  * Order of operations, identical for the three sensors and identical to the
    online pipeline (online/): first resample to 2800 Hz, then filter.
      - ADXL345 (3200 Hz) and MPU9250 (4000 Hz): polyphase resampling
        (ratios 7/8 and 7/10), whose anti-aliasing filter keeps the content
        above 1400 Hz from folding into the retained band.
      - Piezoelectric sensor (non-uniform rate, ~2800 Hz): linear interpolation
        onto a uniform 2800 Hz grid, from the reception timestamps.
  * Causal 4th-order Butterworth high-pass at 1 Hz, in second-order sections
    (the (b, a) form is numerically ill-conditioned for such a low cutoff),
    starting from the steady state of the first sample (no start-up transient).
    The online pipeline keeps the filter state across windows, so the training
    and the live features are computed from identical signals.
  * Piezoelectric sensor:
      - readings above the maximum of the ADC are discarded; they appear when
        two lines of the serial stream are merged into one value;
      - the sensor was replaced after session 14; the recorded values are
        multiplied by the per-session calibration factor of
        data/metadata/sessions.csv (1.0 for sessions 1-14).
"""

import gc
import math

import numpy as np
import pandas as pd
from scipy import signal

from pipeline_common import (ADC_MAX, FS, HP_CUTOFF, HP_ORDER, SENSORS, SYNC_DIR,
                             load_sessions, parse_trial, raw_files, read_raw)

SOS = signal.butter(HP_ORDER, HP_CUTOFF, btype="high", fs=FS, output="sos")


def causal_highpass(x):
    """Causal high-pass, column by column, starting from the steady state."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    out = np.empty_like(x, dtype=np.float32)
    zi0 = signal.sosfilt_zi(SOS)
    for c in range(x.shape[1]):
        y, _ = signal.sosfilt(SOS, x[:, c].astype(np.float64), zi=zi0 * x[0, c])
        out[:, c] = y
    return out


def resample_poly_to_fs(data, original_sr):
    g = math.gcd(FS, original_sr)
    return signal.resample_poly(data, FS // g, original_sr // g, axis=0).astype(np.float32)


def process_adxl(path, sessions, original_sr=3200):
    df = read_raw(path)
    df.columns = df.columns.str.strip()
    time = df["time"].values
    data = causal_highpass(resample_poly_to_fs(df[["x", "y", "z"]].values.astype(np.float32),
                                               original_sr))
    t = np.linspace(time[0], time[-1], len(data), endpoint=False)
    out = pd.DataFrame(data, columns=["x", "y", "z"])
    out.insert(0, "time", t)
    return out


def process_mpu(path, sessions, original_sr=4000):
    df = read_raw(path)
    df.columns = df.columns.str.strip()
    time = df["Sample"].values / original_sr
    data = causal_highpass(resample_poly_to_fs(
        df[["Accel_X", "Accel_Y", "Accel_Z"]].values.astype(np.float32), original_sr))
    t = np.linspace(time[0], time[-1], len(data), endpoint=False)
    out = pd.DataFrame(data, columns=["Accel_X", "Accel_Y", "Accel_Z"])
    out.insert(0, "time", t)
    return out


def process_piezo(path, sessions):
    df = read_raw(path)
    df.columns = df.columns.str.strip()
    session, _ = parse_trial(path.name)
    valid = df["Vibration_Val"].values <= ADC_MAX
    if (~valid).any():
        print(f"    {(~valid).sum()} readings above the maximum of the ADC discarded")
    time = df["Time_Sec"].values[valid]
    factor = float(sessions.loc[session, "piezo_calibration_factor"])
    data = (df["Vibration_Val"].values[valid].astype(np.float64) * factor).astype(np.float32)
    t = np.arange(time[0], time[-1], 1.0 / FS)
    data = np.interp(t, time, data).astype(np.float32)
    data = causal_highpass(data[:, None])[:, 0]
    return pd.DataFrame({"time": t, "Vibration_Val": data})


PROCESSORS = {"adxl": ("adxl345", process_adxl),
              "piezo": ("piezo", process_piezo),
              "mpu": ("mpu9250", process_mpu)}

if __name__ == "__main__":
    sessions = load_sessions()
    for sensor in SENSORS:
        out_name, fn = PROCESSORS[sensor]
        dst = SYNC_DIR / out_name
        dst.mkdir(parents=True, exist_ok=True)
        print(f"Processing {sensor}...")
        for path in raw_files(sensor):
            print(f"  {path.name}")
            fn(path, sessions).to_csv(dst / (path.stem + ".csv"), index=False)
            gc.collect()
    print("Done.")
