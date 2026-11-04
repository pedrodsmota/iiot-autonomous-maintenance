# acquisition/ - sensor sampling

How the vibration data was collected (dissertation, Section 5.3). Each gripper was
fitted with a different sensor:

| Arm | Sensor | Rate | Interface | Code |
|---|---|---|---|---|
| #1 | ADXL345 (3-axis accelerometer) | 3200 Hz | SPI, Raspberry Pi 3 | [`adxl345/`](adxl345/) |
| #2 | DFRobot DFR0052 piezoelectric vibration sensor (analog) | ~2800 Hz | ADC of a Raspberry Pi Pico W, USB serial to the PdM node | [`piezo_pico/`](piezo_pico/) |
| #3 | MPU9250 (3-axis accelerometer, +/-16 g, DLPF bypassed) | 4000 Hz | SPI, Raspberry Pi 3, FIFO burst reads | [`mpu9250/`](mpu9250/) |

All three write in the same way: samples are accumulated in alternating buffers of
about one second and written by a separate thread (pinned to another CPU core in
the C utilities), so that sampling is never blocked by storage.

## Wiring

| Sensor | Connection |
|---|---|
| ADXL345, MPU9250 | SPI0 of the Raspberry Pi: CS to CE0 (pin 24), SDO/MISO to pin 21, SDA/MOSI to pin 19, SCL/SCLK to pin 23, plus 3V3 and GND. Enable SPI first (`raspi-config`). |
| Piezoelectric sensor | Analog output to GP26 (ADC0) of the Pico W |

The MPU9250 utility uses `/dev/spidev0.0` at 4 MHz. Known issues of the recordings,
including isolated impulses in the MPU9250, are listed in the
[data card](https://github.com/pedrodsmota/iiot-autonomous-maintenance/blob/main/dataset/README.md#3-known-issues---read-before-using-the-data).

## adxl345/
Modified version of [adxl345spi](https://github.com/nagimov/adxl345spi) (GPL-3.0,
see `adxl345/LICENSE` and the header of `adxl345spi.c` for the changes). Build on the
Raspberry Pi (requires `pigpio`):
```bash
gcc -Wall -pthread -o adxl345spi adxl345spi.c -lpigpio -lrt -lm
sudo ./adxl345spi -t 3600 -f 3200 -s trial_1.csv
```

## mpu9250/
SPI utility written for this work (spidev, FIFO, writer thread):
```bash
gcc -O2 -Wall -pthread -o mpu9250 mpu9250.c
./mpu9250 -t 3600 -s trial_1.csv     # -f is accepted but ignored (fixed 4 kHz)
```

## piezo_pico/
- `main_collection.py`: MicroPython firmware used to record the dataset; reads the
  ADC (pin 26) in a tight loop and prints each 16-bit value on its own line. The rate
  is set by the USB serial link (about 2760-2900 Hz, depending on the session).
- `main_online.py`: the same, paced at about 2800 Hz; used with the online estimation
  in the end-to-end trials.

- `capture_serial.py`: runs on the Raspberry Pi; timestamps each value on reception
  and writes `Time_Sec,Vibration_Val` to CSV with triple buffering. The port, duration,
  and output file are set at the top of the script. During operation, the online script
  (`online/online_rul_piezo.py`) reads the serial port directly instead.

Copy one of the firmwares to the Pico W as `main.py` (it runs at power-up), e.g. with
[mpremote](https://docs.micropython.org/en/latest/reference/mpremote.html):
`mpremote fs cp main_collection.py :main.py`.

Occasionally two serial lines are merged into a single value above 65535; these
values are discarded downstream (`prognostics/01_sync_data.py`, `online/online_rul_piezo.py`).
