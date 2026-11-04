"""
Shared definitions for the whole offline pipeline (dissertation, Sections 5.3-5.4).

What lives here, and why:
  - all paths, in a single place;
  - window keys that are unique per session, (session, win), which also handle
    sessions recorded in several files (e.g. trial_16.csv + trial_16_1.csv);
  - the real operating time line (seconds of operation), rather than the window
    index, so that the labels are not compressed where a sensor was disconnected;
  - the HI <-> RUL conversion of Equation 5.4;
  - sequences built only from windows that are contiguous in time;
  - a fixed random seed.

Paths
-----
By default, the data is expected in ``<repository>/data``:

    data/raw/<sensor folder>/trial_<n>[_<part>].parquet   (or .csv)
    data/metadata/sessions.csv   (falls back to dataset/metadata/sessions.csv)
    data/synchronized/      written by 01_sync_data.py
    data/features/          written by 02 and 03
    data/processed/         written by 04_build_dataset.py

Set the environment variable ``IIOT_DATA_DIR`` to use another location.
Trained models go to ``<repository>/models`` and results to ``<repository>/results``.
"""

import json
import os
import random
import re
from pathlib import Path

import numpy as np

# ── paths ───────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", REPO_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
# per-session metadata: from the downloaded dataset if present, otherwise the copy in dataset/metadata
METADATA_DIR = (DATA_DIR / "metadata" if (DATA_DIR / "metadata" / "sessions.csv").exists()
                else REPO_DIR / "dataset" / "metadata")
SYNC_DIR = DATA_DIR / "synchronized"
FEAT_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = REPO_DIR / "models"
RESULTS_DIR = REPO_DIR / "results"

RAW_SUBDIRS = {"adxl": "ARM1_adxl345", "piezo": "ARM2_piezo", "mpu": "ARM3_mpu9250"}
FEATURE_FILES = {"adxl": "adxl345_features.csv",
                 "piezo": "piezo_features.csv",
                 "mpu": "mpu9250_features.csv"}
SENSORS = ["adxl", "piezo", "mpu"]
EXCLUDED_SESSIONS = {"adxl": {8}}  # no data recorded (sensor disconnected)

# ── piezoelectric sensor replacement ───────────────────────────────────────
# The piezoelectric sensor was replaced after session 14. The dataset keeps the
# values as recorded (raw ADC integers); the factor that brings sessions 15-22
# to the amplitude of the original sensor, std(session 14) / std(session 15),
# is stored per session in data/metadata/sessions.csv
# (column piezo_calibration_factor) and applied in 01_sync_data.py.
# The same factor is applied to the live signal in online/online_rul_piezo.py.
ADC_MAX = 65535

# ── common parameters ──────────────────────────────────────────────────────
FS = 2800                 # common sampling rate (Hz)
HP_CUTOFF = 1.0           # high-pass cutoff (Hz)
HP_ORDER = 4
ACTIVITY_THRESHOLD = 1e-6 # windows with a lower variance are discarded as stationary
TIME_STEPS = 60           # length of the sequences and of the blocks of the split
TRAIN_RATIO = 0.80
INNER_VAL_RATIO = 0.20    # fraction of the training blocks used for validation in Optuna
SEED = 42

# End of life used in the labels:
#   "failure_onset"    -> start of session 22 (the grippers had already failed completely)
#   "end_of_recording" -> end of session 22
EOL_MODE = "failure_onset"
FAILURE_SESSION = 22


# ── raw files and session metadata ─────────────────────────────────────────
def raw_files(sensor, raw_dir=RAW_DIR):
    """Raw recordings of a sensor, Parquet or CSV, sorted by (session, part)."""
    folder = raw_dir / RAW_SUBDIRS[sensor]
    files = [p for p in folder.iterdir()
             if p.suffix in (".parquet", ".csv") and p.name.startswith("trial_")]
    return sorted(files, key=lambda p: parse_trial(p.name))


def read_raw(path):
    """Read one raw recording, Parquet or CSV."""
    import pandas as pd
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_sessions(metadata_dir=METADATA_DIR):
    """Per-session metadata (duration, faults, piezo calibration factor, ...)."""
    import pandas as pd
    return pd.read_csv(metadata_dir / "sessions.csv").set_index("session")


def piezo_factor(session, sessions=None):
    """Calibration factor of the piezoelectric signal of a session (1.0 before
    the sensor was replaced)."""
    sessions = load_sessions() if sessions is None else sessions
    return float(sessions.loc[session, "piezo_calibration_factor"])


# ── window keys ─────────────────────────────────────────────────────────────
def parse_trial(name):
    """'trial_16_1.csv' -> (16, 1); 'trial_8.parquet' -> (8, 0)."""
    m = re.search(r"trial_(\d+)(?:_(\d+))?", str(name))
    if not m:
        raise ValueError(f"Unexpected session file name: {name}")
    return int(m.group(1)), int(m.group(2) or 0)


def add_window_keys(df):
    """Add 'session', 'part', and 'win' (index of the window within the session).
    Sessions split into several files are chained: the windows of part 1 start
    where those of part 0 end."""
    parsed = df["trial"].map(parse_trial)
    df = df.copy()
    df["session"] = parsed.map(lambda x: x[0]).astype(int)
    df["part"] = parsed.map(lambda x: x[1]).astype(int)
    n_per_part = df.groupby(["session", "part"])["window_idx"].max() + 1
    offsets = {}
    for (s, p), _ in n_per_part.sort_index().items():
        prev = [n_per_part[(s, q)] for q in range(p) if (s, q) in n_per_part.index]
        offsets[(s, p)] = int(sum(prev))
    df["win"] = df["window_idx"] + [offsets[(s, p)] for s, p in zip(df["session"], df["part"])]
    return df


# ── real operating time line ───────────────────────────────────────────────
TIMELINE_FILE = "session_timeline.json"


def build_session_timeline(feat_dir=FEAT_DIR):
    """Duration of each session (s) = maximum, over the three sensors, of the
    number of windows recorded in that session (the raw features still include
    the windows without signal, which are only removed later). The three sensors
    recorded simultaneously, so they share the same time line."""
    import pandas as pd
    dur = {}
    for sensor in SENSORS:
        df = pd.read_csv(feat_dir / FEATURE_FILES[sensor], usecols=["trial", "window_idx"])
        df = add_window_keys(df)
        for s, n in (df.groupby("session")["win"].max() + 1).items():
            dur[int(s)] = max(dur.get(int(s), 0), int(n))
    sessions = sorted(dur)
    offsets, acc = {}, 0
    for s in sessions:
        offsets[s] = acc
        acc += dur[s]
    timeline = {"duration_s": dur, "offset_s": offsets, "total_s": acc}
    with open(feat_dir / TIMELINE_FILE, "w") as f:
        json.dump({k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v)
                   for k, v in timeline.items()}, f, indent=2)
    return timeline


def load_session_timeline(feat_dir=FEAT_DIR):
    with open(feat_dir / TIMELINE_FILE) as f:
        raw = json.load(f)
    return {"duration_s": {int(k): v for k, v in raw["duration_s"].items()},
            "offset_s": {int(k): v for k, v in raw["offset_s"].items()},
            "total_s": raw["total_s"]}


def elapsed_time(df, timeline):
    """Seconds of operation since the start of the data collection."""
    return np.array([timeline["offset_s"][s] for s in df["session"]]) + df["win"].values


# ── HI <-> RUL (Equation 5.4) ───────────────────────────────────────────────
# HI(t') = 2 - 2^t', t' = (t - t_FPT) / L, L = t_EoL - t_FPT
#   => RUL = t_EoL - t = (1 - log2(2 - HI)) * L, valid for HI < 1 (after the FPT).
def hi_to_rul(hi, L):
    hi = np.clip(np.asarray(hi, dtype=float), 0.0, 1.0)
    return (1.0 - np.log2(2.0 - hi)) * L   # HI = 1 gives L (a lower bound before the FPT)


def rul_to_hi(rul, L):
    rul = np.clip(np.asarray(rul, dtype=float), 0.0, L)
    return 2.0 - 2.0 ** (1.0 - rul / L)


def load_label_info(sensor, feat_dir=FEAT_DIR):
    with open(feat_dir / f"{sensor}_label_info.json") as f:
        return json.load(f)


# ── contiguous sequences ───────────────────────────────────────────────────
def contiguity(session, win):
    """cont[i] = True if window i immediately follows window i-1."""
    cont = np.zeros(len(win), dtype=bool)
    cont[1:] = (session[1:] == session[:-1]) & (win[1:] == win[:-1] + 1)
    return cont


def sequence_starts(mask, cont, T=TIME_STEPS):
    """Start indices of all the sequences of T contiguous windows whose rows are
    all in `mask`. The target of each sequence is its last window."""
    starts, run_start = [], None
    for i in range(len(mask)):
        if not mask[i]:
            run_start = None
            continue
        if run_start is None or not cont[i]:
            run_start = i
        if i - run_start + 1 >= T:
            starts.append(i - T + 1)
    return np.array(starts, dtype=np.int64)


def block_split(n_rows, T=TIME_STEPS, seed=SEED):
    """Blocks of T consecutive rows randomly assigned to training and test, with
    a fraction of the training blocks set aside for validation in Optuna."""
    n_blocks = n_rows // T
    idx = np.arange(n_blocks)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(idx)
    sp = int(n_blocks * TRAIN_RATIO)
    train_b, test_b = np.sort(idx[:sp]), np.sort(idx[sp:])
    rng_val = np.random.default_rng(seed=seed + 1)
    perm = rng_val.permutation(train_b)
    nv = int(len(train_b) * INNER_VAL_RATIO)
    val_b, inner_b = np.sort(perm[:nv]), np.sort(perm[nv:])
    return {"n_blocks": n_blocks, "train": train_b, "test": test_b,
            "inner_train": inner_b, "inner_val": val_b}


def block_mask(blocks, n_rows, T=TIME_STEPS):
    m = np.zeros(n_rows, dtype=bool)
    for b in blocks:
        m[b * T:(b + 1) * T] = True
    return m


def test_starts(test_blocks, cont, n_rows, T=TIME_STEPS):
    """One sequence per test block, only if its T windows are contiguous."""
    starts = []
    for b in test_blocks:
        s, e = b * T, (b + 1) * T
        if e <= n_rows and cont[s + 1:e].all():
            starts.append(s)
    return np.array(starts, dtype=np.int64)


# ── reproducibility ────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
