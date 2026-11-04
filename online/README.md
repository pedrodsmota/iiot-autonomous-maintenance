# online/ - online health estimation on the PdM nodes

Scripts that run on the PdM node of each arm (dissertation, Section 5.5). They read
the live vibration stream, reproduce the training preprocessing exactly, run the
model once per second, smooth the HI, convert it into a RUL, and write the result to
a small file read by the `RUL_ESTIMATION` function block.

| Script | Sensor | Input |
|---|---|---|
| `online_rul_adxl.py` | ADXL345 (Arm #1) | MQTT stream from the sampling device (`adxl345spi -f 3200`), on stdin |
| `online_rul_piezo.py` | Piezoelectric (Arm #2) | USB serial from the Pico W (`/dev/ttyACM0`) |
| `online_rul_mpu.py` | MPU9250 (Arm #3) | MQTT stream from the sampling device (4000 Hz), on stdin |
| `online_common.py` | shared | streaming resampler, causal filter, windows, features, model, smoothing |
| `check_online_features.py` | tool | compares the features of a live/replayed run with the training features |

The scripts are launched by the `RUL_ESTIMATION` function block (see `../iec61499/`),
in a separate Python virtual environment (DINASORE runs on an older Python than
PyTorch requires). Copy the scripts, `online_common.py`, and the model and scaler of
the sensor into the folder of the function blocks on the node.

Output file (`--rul-file`), rewritten atomically after every estimate:
```
0.734512      <- smoothed HI (0-1)
OK            <- status
21874.3       <- estimated RUL in seconds (-1 if unknown)
```

## Replaying a recording (no hardware needed)

```bash
python online_rul_piezo.py --csv ../data/raw/ARM2_piezo/trial_15.csv \
    --model ../models/model_piezo.pt --scaler ../models/scaler_piezo.pkl \
    --dump-features /tmp/dump.csv
python check_online_features.py /tmp/dump.csv ../data/features/piezo_features.csv trial_15.csv
```
The median relative difference should be at float precision (~1e-7 to 1e-8).
(The replay modes read CSV; convert a Parquet session with `../dataset/parquet_to_csv.py` first.)

## Pausing the stream to introduce faults

Arms #1 and #3: stop the `IMU_STREAM` function block. Arm #2: `touch /tmp/piezo_pause`
(resume with `rm /tmp/piezo_pause`). The state of the estimator is kept, and the
paused interval is removed from the time scale.

On the testbed, the PdM nodes ran an earlier `online_common.py` that differs only in
the optional feature dump (`--dump-features`): no time stamp column and no line
buffering. The estimates are identical; `check_online_features.py` reads both formats.
