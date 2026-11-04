# Run-to-failure vibration dataset of 3D-printed robotic grippers

Vibration recordings of the complete life cycle of three 3D-printed robotic grippers
under accelerated degradation, each monitored by a different sensor, collected on a
laboratory production line at the Faculty of Engineering of the University of Porto
(FEUP) in 2026.

- **Code that uses this dataset:** https://github.com/pedrodsmota/iiot-autonomous-maintenance
- **DOI:** [10.5281/zenodo.23900129](https://doi.org/10.5281/zenodo.23900129) (Zenodo) · mirror on [Kaggle](https://www.kaggle.com/datasets/pedrodesousamota/run-to-failure-vibration-of-3d-printed-grippers)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Version:** 1.0

This README describes everything needed to use the data on its own, without the code.

---

## 1. Setup

Three [EEZYbotARM Mk2](https://www.instructables.com/EEZYbotARM-Mk2-3D-Printed-Robot/)
robotic arms, made largely of 3D-printed PLA, move a piece back and forth along a small
line (a production cycle of six movements, ~43.7 s). The original claw of the arms was
replaced by the ["Parallel Gripper for EEZYbotARM MK2" by sthone](https://www.thingiverse.com/thing:4362304),
also printed in PLA. This gripper, the component most subject to wear, is the one that
degrades in the recordings; each of the three carries one vibration sensor:

| Folder | Arm | Sensor | Sampling rate | Columns | Unit |
|---|---|---|---|---|---|
| `raw/ARM1_adxl345/` | #1 | ADXL345, 3-axis digital accelerometer, SPI, +/-16 g | 3200 Hz | `time` (s), `x`, `y`, `z` | g |
| `raw/ARM2_piezo/` | #2 | DFRobot DFR0052 piezoelectric vibration sensor (analog), read by the 12-bit ADC of a Raspberry Pi Pico W | ~2800 Hz, not uniform | `Time_Sec` (s), `Vibration_Val`, `sensor_unit` | ADC counts (0-65535) |
| `raw/ARM3_mpu9250/` | #3 | MPU9250, 3-axis digital accelerometer, SPI, +/-16 g, digital low-pass filter bypassed | 4000 Hz | `Sample` (index), `Accel_X`, `Accel_Y`, `Accel_Z` | g |

Notes on the time columns:
- ADXL345: `time` is the time of each sample from the start of the session.
- Piezoelectric: `Time_Sec` is the time at which the Raspberry Pi received each value
  over USB serial (the Pico W has no clock in the stream), so the rate is not uniform.
- MPU9250: there is no time column; the time of a sample is `Sample / 4000` s.

The Pico W returns 16-bit values (`read_u16`) from a 12-bit converter.

## 2. Protocol

The arms ran in sessions of about one hour, one file per session and sensor
(`trial_<session>.parquet`). Faults were introduced to accelerate degradation: the
holes of the screws were progressively loosened and holes were made in the gripper
parts, at the start of a session and simultaneously in the three grippers.

| Sessions | Condition |
|---|---|
| 1-16 | Normal operation, no introduced fault (slight natural wear noted from session 10) |
| 17-21 | One new fault stage introduced at the start of each session (5 stages) |
| 22 | Complete failure: at the start of this session the grippers no longer worked (30 min recorded) |

- Session 9 was interrupted after 34 minutes.
- In total, about 21 h of operation, of which 20 h 34 min until the complete failure.
- In the accompanying work, the **end of life** is the start of session 22.

`metadata/sessions.csv` has one row per session: duration, condition, number of fault
stages present, piezoelectric sensor unit and calibration factor, ADXL345 data
availability and fraction of windows without signal, and notes.

## 3. Known issues - read before using the data

### 3.1 Piezoelectric sensor replaced after session 14
The piezoelectric sensor was replaced between sessions 14 and 15. The values are
stored **as recorded**, so sessions 15-22 (sensor unit `B`) have a different amplitude
scale from sessions 1-14 (unit `A`). To put them on the scale of the original sensor,
multiply `Vibration_Val` by the factor of the session in `metadata/sessions.csv`
(column `piezo_calibration_factor`):

| Sessions | `sensor_unit` | `piezo_calibration_factor` |
|---|---|---|
| 1-14 | A | 1.0 |
| 15-22 | B | 1.9169561438 |

The factor is std(session 14) / std(session 15), computed over all the readings,
including the one invalid reading of session 14 (see 3.2); without it, the ratio would
be 1.892. A single factor also removes any genuine change in amplitude between those
two sessions.

The factor equalizes the standard deviation of the whole signal, which is dominated by
the bursts of the movements, **not the level of the quiet windows**: the median RMS of
the 1-s windows goes from 240 in session 14 to 559 in session 15, after the factor (292
without it). This step comes from the sensor, not from wear, and a model can learn it.

```python
import pandas as pd
sessions = pd.read_csv("metadata/sessions.csv").set_index("session")
df = pd.read_parquet("raw/ARM2_piezo/trial_18.parquet")
df = df[df["Vibration_Val"] <= 65535]                                    # see 3.2
df["vibration"] = df["Vibration_Val"] * sessions.loc[18, "piezo_calibration_factor"]
```

### 3.2 Invalid piezoelectric readings
Occasionally two lines of the serial stream were merged into a single value above
the maximum of the converter (e.g. 544544). Discard values above 65535.

### 3.3 Missing ADXL345 data
The ADXL345 was disconnected during part of the collection. Its files keep everything
the sampling utility wrote, including those periods, in which there is no vibration
signal (near-zero variance after high-pass filtering, with occasional spurious
values). Column `adxl345_windows_without_signal` of `metadata/sessions.csv` gives,
per session, the fraction of 1-s windows without signal:

| Session | 6 | 7 | 8 | 9 | 10 | 11 | 22 | others |
|---|---|---|---|---|---|---|---|---|
| Windows without signal | 91% | 47% | 82% | 37% | 100% | 82% | 59% | <= 0.1% |

Session 8 has no usable signal and was excluded in the accompanying work; about 4.5 h
of the life cycle of the ADXL345 is missing in total. Discard windows with near-zero
variance (the accompanying work uses variance <= 1e-6 g^2 after the 1 Hz high-pass
filter).

### 3.4 Session 16 of the piezoelectric sensor
Recorded in two files, `trial_16.parquet` and `trial_16_1.parquet`; concatenate them in
this order.

### 3.5 Outliers in the accelerometers
Some isolated readings are far outside the range of normal operation (up to about
+/-102 g in the ADXL345). They are kept as recorded.

### 3.6 Isolated impulses in the MPU9250
The MPU9250 recordings have many isolated single-sample impulses: one reading far from
both of its neighbors, sometimes reaching the +/-16 g limit (e.g. `-0.024, -15.626,
-0.033`). In every session, 4-10% of the samples of each axis are more than 4 g away from
the median of their two neighbors, against less than 0.1% in the ADXL345 (sessions 1, 14,
and 21). The sample indices have no gaps. The cause was not established, and they are
unlikely to be vibration. They are kept as recorded, and they dominate the RMS and the
spectrum of this sensor; a 3-point median filter removes them. The features in `features/`
and the models of the accompanying work were computed without removing them.

### 3.7 Labels
The dataset contains **no failure labels** other than the protocol above. Health
labels (health stages, Health Index, RUL) were constructed in the accompanying work
(dissertation, Section 5.4.2); they are not part of the dataset, because they depend on
modeling choices, and they encode elapsed time as much as condition.

## 4. Files

```
raw/
  ARM1_adxl345/trial_1.parquet ... trial_22.parquet
  ARM2_piezo/trial_1.parquet ... trial_22.parquet  (+ trial_16_1.parquet)
  ARM3_mpu9250/trial_1.parquet ... trial_22.parquet
metadata/
  sessions.csv
features/                    derived; the features used to train the published models
  adxl345_features.csv
  piezo_features.csv
  mpu9250_features.csv
```

### Feature files
One row per 1-s window (2800 samples) of each recording, computed after resampling
to 2800 Hz and a causal 4th-order Butterworth high-pass filter at 1 Hz (code:
`prognostics/01_sync_data.py` and `02_extract_features.py`). The piezoelectric
features are computed after the calibration factor of Section 3.1.

| Column | Meaning |
|---|---|
| `trial` | source file (`trial_16_1.csv` is the second part of session 16) |
| `window_idx` | index of the window within the file |
| `time_start_sec` | start of the window, s from the start of the file |
| `<channel>_rms`, `_var`, `_peak`, `_crest_factor`, `_skew`, `_kurtosis` | time-domain features |
| `<channel>_total_spectral_energy` | energy of the real FFT of the window |
| `<channel>_energy_band_1` ... `_4` | energy in 0-350, 350-700, 700-1050, 1050-1400 Hz |

Channels: `x`, `y`, `z` (ADXL345), `Vibration_Val` (piezoelectric), `Accel_X`,
`Accel_Y`, `Accel_Z` (MPU9250). Windows without signal (e.g. while the ADXL345 was
disconnected) are still included; discard those with near-zero variance.

On Zenodo, each folder is a zip file (`ARM1_adxl345.zip`, `ARM2_piezo.zip`,
`ARM3_mpu9250.zip`, `features.zip`) and `sessions.csv` is a separate file; unzipping
the zip files in the same folder and moving `sessions.csv` to `metadata/` gives the
layout above.

Parquet files can be read with pandas (`pd.read_parquet`, requires `pyarrow`), Polars,
Spark, DuckDB, or R (`arrow::read_parquet`). To get CSV:
`python parquet_to_csv.py --src raw --dst raw_csv` (script in the code repository,
folder `dataset/`).

## 5. Collection hardware and software

Raspberry Pi 3 Model B boards (one per sensor), sampling with C utilities over SPI
(ADXL345: modified [adxl345spi](https://github.com/nagimov/adxl345spi); MPU9250:
custom utility) and MicroPython on the Pico W. The code is in the folder
`acquisition/` of the code repository.

## 6. Citation

The dataset:

> P. Mota, "Run-to-failure vibration dataset of 3D-printed robotic grippers," version 1.0,
> Zenodo, 2026. https://doi.org/10.5281/zenodo.23900129

The work in which it was collected and used:

> P. Mota, "Leveraging IEC 61499 for Autonomous Maintenance: Digital Twin Integration
> and 3D Printing in Distributed Control Systems," MSc dissertation, Faculty of
> Engineering, University of Porto, 2026.

> P. Mota and R. Pinto, "An Industrial IoT Architecture for Autonomous Maintenance:
> Integrating Digital Twins, Distributed Control, and On-Demand Additive
> Manufacturing," *Applied Sciences*, 2026 (accepted for publication).

Questions and problems: open an issue in the
[code repository](https://github.com/pedrodsmota/iiot-autonomous-maintenance/issues),
or contact Pedro Mota (pedro_mota@outlook.pt).
