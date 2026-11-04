# online/

Online estimation scripts used in the end-to-end trials of the article, launched by the `HI_ESTIMATION` function block in their own virtual environment (see [`../environment/requirements-inference.txt`](../environment/requirements-inference.txt)).

| Script | Sensor | Input |
|---|---|---|
| `online_rul_adxl.py` | ADXL345 (Arm #1) | MQTT stream, on stdin |
| `online_rul_piezo.py` | Piezoelectric (Arm #2) | USB serial from the Pico W |
| `online_rul_mpu.py` | MPU9250 (Arm #3) | MQTT stream, on stdin |

Usage is at the top of each script. The later version, on the `main` branch, reproduces the training preprocessing exactly and should be preferred for new work.
