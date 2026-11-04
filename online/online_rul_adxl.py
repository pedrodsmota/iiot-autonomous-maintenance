#!/usr/bin/env python3
"""
online_rul_adxl.py - online HI and RUL estimation for the ADXL345 (Arm #1).

  adxl345spi -f 3200 (sampling device) --MQTT--> stdin
      -> streaming resample_poly 3200 -> 2800 Hz (identical to training)
      -> causal 1 Hz high-pass, with state
      -> windows of 2800 samples -> 33 features -> scaler -> sequence of 60
      -> CNN-BiLSTM -> HI -> median + EMA -> RUL -> file read by RUL_ESTIMATION

Notes
  * The ADXL345 must be sampled at 3200 Hz (adxl345spi -f 3200, FREQ = 3200 in
    the IMU_STREAM function block), as during data collection. Sampling directly
    at 2800 Hz would pick the SPI reading closest to each instant, without an
    anti-aliasing filter, and would not match the training data.
  * Same smoothing for the three sensors (median of 60 + EMA with a span of 60).
  * The order of the features is checked against the checkpoint.

Usage (on the PdM node, fed by the MQTT stream):
  mosquitto_sub -t <topic> | python3 online_rul_adxl.py --rul-file /tmp/online_rul.txt
Offline replay of a recording:
  python3 online_rul_adxl.py --mode file --file trial_12.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from online_common import (CausalHighpass, Estimator, PolyphaseStream, Windower,
                           add_common_args, feature_columns, load_model_and_scaler,
                           report, write_atomic)

AXES = ["x", "y", "z"]
CHUNK = 320          # samples per processing block (0.1 s at 3200 Hz)


def parse_lines(stream):
    """Accepts 'x,y,z' or 't,x,y,z' (adxl345spi format). Ignores status lines of
    adxl345spi and corrupted lines."""
    for raw in stream:
        parts = raw.strip().split(",")
        try:
            if len(parts) == 3:
                yield float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 4:
                yield float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue


def main():
    ap = argparse.ArgumentParser(description="Online HI/RUL estimation - ADXL345")
    ap.add_argument("--mode", default="stdin", choices=["stdin", "file"])
    ap.add_argument("--file", default=None, help="CSV recorded by adxl345spi (file mode)")
    ap.add_argument("--input-rate", type=int, default=3200, choices=[2800, 3200],
                    help="Sampling rate of adxl345spi (-f). 3200 reproduces training.")
    add_common_args(ap)
    a = ap.parse_args()
    a.model = a.model or "model_adxl.pt"
    a.scaler = a.scaler or "scaler_adxl.pkl"

    model, scaler, T, L = load_model_and_scaler(a.model, a.scaler, feature_columns(AXES))
    if a.input_rate == 2800:
        print("[adxl] WARNING: 2800 Hz input without resampling; does not match training.",
              file=sys.stderr)
    resampler = PolyphaseStream(7, 8, 3) if a.input_rate == 3200 else None
    hp, win = CausalHighpass(3), Windower(3)
    est = Estimator(model, scaler, T, L, a.median, a.ema, a.monotone, a.dump_features)

    src = open(a.file) if a.mode == "file" else sys.stdin
    block = []
    try:
        for s in parse_lines(src):
            block.append(s)
            if len(block) < CHUNK:
                continue
            x = np.asarray(block)
            block = []
            if resampler is not None:
                x = resampler.push(x)
            for w in win.push(hp.push(x)):
                r = est.push_window(w)
                if r is not None:
                    write_atomic(a.rul_file, r["hi"], r["rul_s"])
                    report(r, a.quiet, "adxl")
    except KeyboardInterrupt:
        pass
    print(f"[adxl] end: {est.n_windows} windows ({est.n_dead} without signal)", file=sys.stderr)


if __name__ == "__main__":
    main()
