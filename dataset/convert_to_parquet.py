"""
Build the published dataset from the original CSV recordings.

Converts every recording to Parquet (same file names, same columns, values as
recorded), adds the column `sensor_unit` to the piezoelectric recordings
(A = original sensor, sessions 1-14; B = replacement sensor, sessions 15-22),
and copies the session metadata.

Usage:
  python dataset/convert_to_parquet.py --src <folder with the sensor folders> --dst <output folder> \
                                       [--piezo-src ARM2_piezo_raw]

--piezo-src names the source folder of the piezoelectric recordings. It must hold
the values AS RECORDED (raw ADC integers), not a copy already multiplied by the
calibration factor.

Output layout (the layout expected by the pipeline in data/):
  <dst>/raw/ARM1_adxl345/trial_<n>.parquet
  <dst>/raw/ARM2_piezo/trial_<n>[_<part>].parquet
  <dst>/raw/ARM3_mpu9250/trial_<n>.parquet
  <dst>/metadata/sessions.csv
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SENSOR_DIRS = ["ARM1_adxl345", "ARM2_piezo", "ARM3_mpu9250"]
DTYPES = {"ARM1_adxl345": {"time": "float64", "x": "float64", "y": "float64", "z": "float64"},
          "ARM2_piezo": {"Time_Sec": "float64", "Vibration_Val": "int32"},
          "ARM3_mpu9250": {"Sample": "int64", "Accel_X": "float32", "Accel_Y": "float32",
                           "Accel_Z": "float32"}}


def session_of(name):
    return int(re.search(r"trial_(\d+)", name).group(1))


def main():
    ap = argparse.ArgumentParser(description="Convert the original CSV recordings to Parquet.")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--piezo-src", default="ARM2_piezo",
                    help="Source folder of the piezoelectric recordings (values as recorded)")
    a = ap.parse_args()
    sessions = pd.read_csv(HERE / "metadata" / "sessions.csv").set_index("session")

    for sensor in SENSOR_DIRS:
        out = a.dst / "raw" / sensor
        out.mkdir(parents=True, exist_ok=True)
        src = a.src / (a.piezo_src if sensor == "ARM2_piezo" else sensor)
        files = sorted(p for p in src.glob("trial_*.csv"))
        for p in files:
            df = pd.read_csv(p, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            df = df.astype(DTYPES[sensor])
            if sensor == "ARM2_piezo":
                unit = sessions.loc[session_of(p.name), "piezo_sensor_unit"]
                df["sensor_unit"] = pd.Categorical([unit] * len(df), categories=["A", "B"])
            df.to_parquet(out / (p.stem + ".parquet"), index=False, compression="zstd")
            print(f"{sensor}/{p.stem}.parquet  ({len(df):,} rows)")

    (a.dst / "metadata").mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "metadata" / "sessions.csv", a.dst / "metadata" / "sessions.csv")
    print("Done.")


if __name__ == "__main__":
    main()
