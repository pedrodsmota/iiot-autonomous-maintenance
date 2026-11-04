#!/usr/bin/env python3
"""
online_rul_mpu.py - online HI and RUL estimation for the MPU9250 (Arm #3).

  MPU9250 C sampling utility (4000 Hz) --MQTT--> stdin  (lines Sample,X,Y,Z)
      -> streaming resample_poly 4000 -> 2800 Hz (identical to training)
      -> causal 1 Hz high-pass, with state
      -> windows of 2800 samples -> 33 features -> scaler -> sequence of 60
      -> CNN-BiLSTM -> HI -> median + EMA -> RUL -> file read by RUL_ESTIMATION

Notes
  * The streaming resampler gives the same output as resample_poly applied to
    the whole recording in training (no transients at the block boundaries).
  * Same order as training: resample first, then filter at 2800 Hz.

Usage (on the PdM node, fed by the MQTT stream):
  mosquitto_sub -t <topic> | python3 online_rul_mpu.py --rul-file /tmp/online_rul.txt
Offline replay:
  python3 online_rul_mpu.py --mode csv --input trial_12.csv
  python3 online_rul_mpu.py --mode folder --input data/raw/ARM3_mpu9250
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from online_common import (CausalHighpass, Estimator, PolyphaseStream, Windower,
                           add_common_args, feature_columns, load_model_and_scaler,
                           report, write_atomic)

AXES = ["Accel_X", "Accel_Y", "Accel_Z"]
CHUNK = 400          # 0.1 s at 4000 Hz


def parse_stdin(stream, skip_header=True):
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            yield float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue      # header "Sample,Accel_X,..." or corrupted line


def iter_csv(paths):
    for p in paths:
        print(f"[mpu] {p.name}", file=sys.stderr)
        for chunk in pd.read_csv(p, usecols=lambda c: c.strip() in AXES, chunksize=40000):
            chunk.columns = chunk.columns.str.strip()
            yield from chunk[AXES].itertuples(index=False, name=None)


def main():
    ap = argparse.ArgumentParser(description="Online HI/RUL estimation - MPU9250")
    ap.add_argument("--mode", default="live", choices=["live", "csv", "folder"])
    ap.add_argument("--input", default=None, help="CSV (csv mode) or folder (folder mode)")
    ap.add_argument("--no-skip-header", dest="skip_header", action="store_false", default=True)
    ap.add_argument("--device", default="cpu", help="Ignored; kept for compatibility.")
    add_common_args(ap)
    a = ap.parse_args()
    a.model = a.model or "model_mpu.pt"
    a.scaler = a.scaler or "scaler_mpu.pkl"

    model, scaler, T, L = load_model_and_scaler(a.model, a.scaler, feature_columns(AXES))
    resampler, hp, win = PolyphaseStream(7, 10, 3), CausalHighpass(3), Windower(3)
    est = Estimator(model, scaler, T, L, a.median, a.ema, a.monotone, a.dump_features)

    if a.mode == "live":
        samples = parse_stdin(sys.stdin, a.skip_header)
    elif a.mode == "csv":
        samples = iter_csv([Path(a.input)])
    else:
        key = lambda p: [int(n) for n in re.findall(r"\d+", p.name)]
        samples = iter_csv(sorted(Path(a.input).glob("trial_*.csv"), key=key))

    block = []
    try:
        for s in samples:
            block.append(s)
            if len(block) < CHUNK:
                continue
            x = resampler.push(np.asarray(block))
            block = []
            for w in win.push(hp.push(x)):
                r = est.push_window(w)
                if r is not None:
                    write_atomic(a.rul_file, r["hi"], r["rul_s"])
                    report(r, a.quiet, "mpu")
    except KeyboardInterrupt:
        pass
    print(f"[mpu] end: {est.n_windows} windows ({est.n_dead} without signal)", file=sys.stderr)


if __name__ == "__main__":
    main()
