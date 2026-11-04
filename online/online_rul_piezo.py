#!/usr/bin/env python3
"""
online_rul_piezo.py - online HI and RUL estimation for the piezoelectric
sensor (Arm #2).

  Pico W (USB serial): ADC integers 0-65535, timestamped on reception by the Pi
      -> readings > 65535 discarded (two serial lines merged into one value)
      -> x calibration factor of the replacement sensor (see below)
      -> continuous linear interpolation onto a 2800 Hz grid (identical to training)
      -> causal 1 Hz high-pass, with state
      -> windows of 2800 samples -> 11 features -> scaler -> sequence of 60
      -> CNN-BiLSTM -> HI -> median + EMA -> RUL -> file read by RUL_ESTIMATION

Calibration
  The piezoelectric sensor was replaced after session 14 of the data collection.
  Sessions 15-22 were brought to the amplitude of the original sensor by the
  factor std(session 14) / std(session 15) = 1.9169561438, so the model expects
  amplitudes on that scale. Live, with the replacement sensor, the same factor
  is applied. When replaying recordings of the dataset (--csv, --csv-dir), whose
  values are stored as recorded, the factor is applied per session (1.0 for
  sessions 1-14), as in prognostics/01_sync_data.py.

Pause (to introduce faults without unplugging the Pico or restarting inference)
  touch /tmp/piezo_pause   -> pause: the serial port is still read, but the
                              samples are discarded
  rm /tmp/piezo_pause      -> resume: the paused interval is removed from the
                              time scale, and the stream continues as if
                              uninterrupted (as the sessions of the training data
                              were chained with last + 1/2800)
  The state of the estimator (sequence of 60, median, EMA) is kept, as for the
  other two sensors when IMU_STREAM is stopped. A pause file left over from a
  previous run is deleted at start-up.

The smoothed HI may increase again (e.g. after a replacement), which lets
RUL_THRESHOLD re-arm; --monotone prevents it.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from online_common import (CausalHighpass, Estimator, InterpStream, Windower,
                           add_common_args, feature_columns, load_model_and_scaler,
                           report, write_atomic)

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
ADC_MAX = 65535
SENSOR_CALIBRATION_MULTIPLIER = 1.9169561438
FIRST_NEW_SENSOR_SESSION = 15
AXES = ["Vibration_Val"]
PAUSE_FILE = "/tmp/piezo_pause"
PAUSE_CHECK_S = 0.25     # interval between checks of the pause file
FS = 2800


def serial_reader(port, q, stop, stats, pause_file):
    try:
        import serial
    except ImportError:
        stats["error"] = "pyserial is not installed (pip install pyserial)"
        return
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
    except Exception as exc:
        stats["error"] = str(exc)
        return
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    shift = 0.0          # total paused time removed from the time scale
    last_t = None        # last instant delivered to the pipeline (pauses removed)
    paused = False
    paused_at = 0.0
    next_check = 0.0
    while not stop.is_set():
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            stats["error"] = "Serial port disconnected."
            break
        now = time.perf_counter() - t0

        if now >= next_check:
            next_check = now + PAUSE_CHECK_S
            want_pause = os.path.exists(pause_file)
            if want_pause and not paused:
                paused, paused_at = True, now
                stats["pauses"] += 1
                print(f"[piezo] paused ({pause_file} exists)", file=sys.stderr, flush=True)
            elif paused and not want_pause:
                paused = False
                if last_t is not None:
                    # The next sample is placed 1/2800 s after the last one before the pause.
                    shift = now - (last_t + 1.0 / FS)
                print(f"[piezo] resumed after {now - paused_at:.0f} s of pause",
                      file=sys.stderr, flush=True)

        if paused or not raw.isdigit():
            continue          # while paused the port is still read, so that old data does not pile up
        v = int(raw)
        if v > ADC_MAX:
            stats["rejected"] += 1
            continue
        last_t = now - shift
        q.put((last_t, v))
    ser.close()


def csv_samples(paths, stats, multiplier=None):
    """Replay recordings of the dataset (Time_Sec, Vibration_Val, values as
    recorded), with the timestamps chained between files. Each value is
    multiplied by the calibration factor of its session, or by `multiplier`."""
    offset = 0.0
    for p in paths:
        m = re.search(r"trial_(\d+)", p.name)
        if multiplier is not None:
            factor = multiplier
        else:
            factor = (SENSOR_CALIBRATION_MULTIPLIER
                      if m and int(m.group(1)) >= FIRST_NEW_SENSOR_SESSION else 1.0)
        last = 0.0
        with p.open(newline="") as fh:
            r = csv.reader(fh)
            header = [h.strip().lower() for h in next(r)]
            ti, vi = header.index("time_sec"), header.index("vibration_val")
            for row in r:
                try:
                    t, v = float(row[ti]), float(row[vi])
                except (ValueError, IndexError):
                    continue
                if v > ADC_MAX:
                    stats["rejected"] += 1
                    continue
                last = t
                yield t + offset, v * factor
        offset += last + 1.0 / FS


def main():
    ap = argparse.ArgumentParser(description="Online HI/RUL estimation - piezoelectric sensor")
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--mode", default=None, help="Ignored; kept for compatibility with the function block.")
    ap.add_argument("--multiplier", type=float, default=None,
                    help=f"Calibration factor. Default: {SENSOR_CALIBRATION_MULTIPLIER} live, "
                         "and the per-session factor with --csv/--csv-dir.")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--csv-dir", type=Path, default=None)
    ap.add_argument("--pause-file", default=PAUSE_FILE,
                    help=f"While this file exists, the live stream is paused "
                         f"(default {PAUSE_FILE}).")
    add_common_args(ap)
    a = ap.parse_args()
    a.model = a.model or "model_piezo.pt"
    a.scaler = a.scaler or "scaler_piezo.pkl"
    if a.csv is not None and a.csv_dir is not None:
        sys.exit("--csv and --csv-dir are mutually exclusive.")

    model, scaler, T, L = load_model_and_scaler(a.model, a.scaler, feature_columns(AXES))
    replay = a.csv is not None or a.csv_dir is not None
    # live: one factor for the whole stream; replay: the factor is applied per session in csv_samples
    mult = 1.0 if replay else (a.multiplier if a.multiplier is not None else SENSOR_CALIBRATION_MULTIPLIER)
    print(f"[piezo] calibration factor: "
          f"{'per session' if replay and a.multiplier is None else (a.multiplier if replay else mult)}",
          file=sys.stderr)

    interp, hp, win = InterpStream(), CausalHighpass(1), Windower(1)
    est = Estimator(model, scaler, T, L, a.median, a.ema, a.monotone, a.dump_features)
    stats = {"rejected": 0, "error": None, "pauses": 0}

    def process(ts, vs):
        for w in win.push(hp.push(interp.push(ts, np.asarray(vs) * mult))):
            r = est.push_window(w)
            if r is not None:
                write_atomic(a.rul_file, r["hi"], r["rul_s"])
                report(r, a.quiet, "piezo")

    try:
        if replay:
            key = lambda p: [int(n) for n in re.findall(r"\d+", p.name)]
            paths = [a.csv] if a.csv else sorted(a.csv_dir.glob("trial_*.csv"), key=key)
            ts, vs = [], []
            for t, v in csv_samples(paths, stats, a.multiplier):
                ts.append(t)
                vs.append(v)
                if len(ts) >= FS:
                    process(ts, vs)
                    ts, vs = [], []
            if ts:
                process(ts, vs)
        else:
            if os.path.exists(a.pause_file):
                os.remove(a.pause_file)
                print(f"[piezo] {a.pause_file} left over from a previous run deleted; starting unpaused",
                      file=sys.stderr)
            q, stop = queue.Queue(maxsize=50_000), threading.Event()
            th = threading.Thread(target=serial_reader,
                                  args=(a.port, q, stop, stats, a.pause_file), daemon=True)
            th.start()
            while th.is_alive() or not q.empty():
                ts, vs = [], []
                try:
                    while len(ts) < 5000:
                        t, v = q.get(timeout=0.05)
                        ts.append(t)
                        vs.append(v)
                except queue.Empty:
                    pass
                if ts:
                    process(ts, vs)
            if stats["error"]:
                print(f"[piezo] error: {stats['error']}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    print(f"[piezo] end: {est.n_windows} windows ({est.n_dead} without signal), "
          f"{stats['rejected']} readings rejected, {stats['pauses']} pauses", file=sys.stderr)


if __name__ == "__main__":
    main()
