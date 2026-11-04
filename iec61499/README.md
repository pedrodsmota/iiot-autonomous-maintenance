# iec61499/ - distributed control application

The IEC 61499 side of the system (dissertation, Sections 4.3, 5.2, and 5.6): custom
function blocks for the [DINASORE](https://github.com/DIGI2-FEUP/dinasore) runtime,
the system configuration developed in the Eclipse 4diac IDE, and deployment notes.

```
IIOT_AM/IIOT_AM.sys                    system configuration (9 devices) for the 4diac IDE
IIOT_AM/function_blocks/               function blocks of the system (.fbt interface + .py implementation)
IIOT_AM/function_blocks/arm_config/    movement sequences of the arms, read by ARM_CONTROL
replacement_part/gripper.gcode         sliced replacement gripper, printed by the AdM node
```

Screenshots of the applications, as in the appendix of the dissertation, so that they can
be read without installing the IDE: [system configuration](../docs/figures/4diac/system_full.png);
control and sampling of Arm [#1](../docs/figures/4diac/pipeline_arm1.png),
[#2](../docs/figures/4diac/pipeline_arm2.png), [#3](../docs/figures/4diac/pipeline_arm3.png);
PdM node of Arm [#1](../docs/figures/4diac/pipeline_arm1dt.png),
[#2](../docs/figures/4diac/pipeline_arm2dt.png), [#3](../docs/figures/4diac/pipeline_arm3dt.png);
[AdM node](../docs/figures/4diac/pipeline_adm.png).

## Function blocks

| FB | Device | Interface (events / data) | Function |
|---|---|---|---|
| `ARM_CONTROL` | control device | RUN / FILE | Executes the movements described in `arm_config/<FILE>.xml` when triggered (Adafruit ServoKit, PCA9685) |
| `IMU_STREAM` | sampling device | START, STOP / EXEC_PATH, FREQ, DURATION, TARGET_IP, TARGET_PORT, TOPIC | Runs the C sampling utility and streams the samples over MQTT to the PdM node |
| `RUL_ESTIMATION` | PdM node | START, STOP, POLL -> DATA / ENV_PATH, SCRIPT_PATH, PORT, MODE, MODEL_PATH, SCALER_PATH, TOPIC -> HI, RUL | Launches the online estimation script (`../online/`) in its own virtual environment and exposes the HI and the RUL (hours) |
| `RUL_THRESHOLD` | PdM node | RUN -> PRINT / VALUE, THRESHOLD | Emits the print order once when the RUL falls below THRESHOLD; re-arms when it rises above THRESHOLD + 0.5 h |
| `PRINT_QUEUE` | AdM node | REQUEST, CHECK -> START_PRINT, QUEUED, DROPPED / REQUEST_FILE, JOB_STATE | Forwards print orders when the printer is idle, holds them otherwise |
| `PRINT_START` | AdM node | START_PRINT -> STARTED, START_FAILED / OCTO_URL, LOCATION, FILE_PATH | Selects and starts the stored G-code file through the OctoPrint REST API |
| `PRINT_CHECK` | AdM node | READ -> PRINT_DONE, PROGRESS_UPDATE / OCTO_URL -> JOB_STATE, COMPLETION | Queries the state and progress of the print job once per second |

MQTT communication uses three more blocks, adapted from the
[DINASORE function block repository](https://github.com/DIGI2-FEUP/dinasore_function_blocks)
(`FBs/MQTT`). They are included here because the versions on the testbed differ from
those in that repository, and in the interface of `READ_MQTT` and `WRITE_MQTT`:

| FB | Interface (events / data) | Function |
|---|---|---|
| `CONNECT_MQTT` | INIT, READ -> INIT_O, READ_O / ADDRESS, PORT, TIMEOUT -> CLIENT, MESSAGES | Connects to a broker; on READ, waits for new messages and outputs them (topic -> last payload) |
| `READ_MQTT` | INIT, SUBS_TOPIC, RUN -> RUN_O / TOPIC, CLIENT, MESSAGES -> VALUE | Subscribes to TOPIC; on RUN, outputs the message of TOPIC if one arrived |
| `WRITE_MQTT` | INIT, RUN -> RUN_O / TOPIC, MESSAGE, CLIENT | Publishes MESSAGE on TOPIC |

They are the versions that ran on the arm devices, and they are the ones to use with
`IIOT_AM.sys`. The broker connection is shared:
`CONNECT_MQTT` outputs the client object, which `READ_MQTT` and `WRITE_MQTT` receive.

### Arm movements (`IIOT_AM/function_blocks/arm_config/`)

`arm<A>_<from><to>.xml` moves the piece from station `<from>` to station `<to>`
(`arm1_12`: Arm #1, station 1 -> 2). Each file lists named positions (angles in
degrees of the base, left and right servos, and gripper) and the sequence of
positions, with the time to wait after commanding each one, in seconds. The FILE parameter of each
`ARM_CONTROL` instance in `IIOT_AM.sys` names one of these files.

### Thresholds used in the dissertation (`RUL_THRESHOLD.THRESHOLD`, hours)

| Arm | Sensor | Target RUL |
|---|---|---|
| #1 | ADXL345 | 4.47 |
| #2 | Piezoelectric | 3.93 |
| #3 | MPU9250 | 4.63 |

Time to act (3.6 h: 3 h 26 min print + ~10 min replacement) plus a margin from the
95th percentile of the optimistic RUL error on test data; see
`../prognostics/10_choose_thresholds.py` and `../results/thresholds.json`.

## The system in `IIOT_AM.sys`

| Application | Device in the system | Runs on | Address on the testbed |
|---|---|---|---|
| `ARM1`, `ARM2`, `ARM3` | `ARM1`, `ARM2`, `ARM3` | Control devices | 10.227.18.20, .22, .21 |
| (part of `ARM1`, `ARM3`) | `ARM1_SENSOR`, `ARM3_SENSOR` | Sampling devices | 10.227.18.15, .12 |
| `PsM1`, `PsM2`, `PsM3` | `ARM1_DT`, `ARM2_DT`, `ARM3_DT` | PdM nodes | 10.227.18.27, .211, .236 |
| `PRINTER` | `PRINTER` | AdM node | 10.227.155.198 |

The addresses are private addresses of the laboratory network: replace them with those
of your own devices.

| MQTT topic | From -> to | Payload |
|---|---|---|
| `/ARM1_12/RUN`, `/ARM2_23/RUN`, `/ARM3_23/RUN`, `/ARM3_32/RUN`, `/ARM2_21/RUN`, `/ARM1_21/RUN` | Control device -> control device of the next movement | Trigger of the movement |
| `ARM1`, `ARM3` | Sampling device (`IMU_STREAM`) -> broker of its PdM node, over Ethernet (10.0.0.1) | One sample per message |
| `/PRINT` | PdM node (`RUL_THRESHOLD`) -> broker of the AdM node | Print order |

## Opening the system in the 4diac IDE

With the 4diac IDE 1.11.0:

1. When the IDE asks for a workspace, choose an empty folder **outside this repository**.
2. On the welcome page, press **Continue to 4diac IDE**.
3. Choose **File > Import > General > Existing Projects into Workspace**.
4. In *Select root directory*, choose `iec61499/IIOT_AM`, tick **Copy projects into
   workspace**, and press **Finish**.

Do not use **File > Import > 4diac IDE > System Import**. In this version it does not load
the function blocks of the project: every block is left with "type not set", and if it
runs over this folder, every application appears twice.

The project contains only the types the system uses. To add other standard blocks, copy
them from the `Tool Library` project that the IDE creates in the workspace.

## Deployment (outline)

On every Raspberry Pi:
1. Install DINASORE (see [DINASORE versions](#dinasore-versions)) and, on the nodes that
   receive messages, the Mosquitto broker (each receiving node runs its own; see
   [Mosquitto](#mosquitto)).
2. Copy the `.fbt` and `.py` files of the function blocks used by that device, from
   `IIOT_AM/function_blocks/`, to `dinasore/resources/function_blocks/`.
3. Start DINASORE on each device and deploy the applications from the 4diac IDE:
   `python core/main.py -a <device-ip> -p 61499`, from the `dinasore` folder.

Control devices, additionally:
- Enable I2C and install `adafruit-circuitpython-servokit` (PCA9685).
- Copy `IIOT_AM/function_blocks/arm_config/` to `dinasore/resources/function_blocks/arm_config/`.

Sampling devices, additionally:
- Enable SPI, install `pigpio` (ADXL345) and `mosquitto-clients`, and build the sampling
  utility (see [`../acquisition/`](../acquisition/)) in the `dinasore` folder:
  `EXEC_PATH` is `./adxl345spi` or `./mpu9250`. `IMU_STREAM` runs it with `sudo`, so the
  user needs passwordless `sudo`, as on Raspberry Pi OS by default.

PdM nodes, additionally:
- Install `mosquitto-clients` and `pv`, used by the command that `RUL_ESTIMATION` launches
  for the MQTT streams (Arms #1 and #3).
- Create a virtual environment for the online estimation with
  `../environment/requirements-pdm-online.txt`, in the `dinasore` folder (`ENV_PATH` is
  relative to it: `myenv`/`my_env` in `IIOT_AM.sys`).
- Copy `../online/online_rul_<sensor>.py`, `../online/online_common.py`, and the model and
  scaler of the sensor (`../models/`) to `dinasore/resources/function_blocks/`
  (`SCRIPT_PATH`, `MODEL_PATH`, and `SCALER_PATH` are relative to that folder).
  `PORT` and `TOPIC` are the MQTT port and topic of the stream, or the serial port for Arm #2.

AdM node:
- OctoPi image (OctoPrint) plus DINASORE `v0.1`, in a virtual environment with
  `../environment/requirements-adm-dinasore.txt`. `PRINT_START` and `PRINT_CHECK` read the
  OctoPrint API key from the environment variable `API_KEY` (never hard-coded):
  `export API_KEY=<your OctoPrint API key>` before starting DINASORE.
- Upload [`replacement_part/gripper.gcode`](replacement_part/gripper.gcode) to OctoPrint
  (`FILE_PATH` = `gripper.gcode`, `LOCATION` = `local` in `IIOT_AM.sys`). It was sliced for
  an Ender-3 S1 Pro with a 0.6 mm nozzle and PLA; for another printer, slice the
  [gripper](https://www.thingiverse.com/thing:4362304) again.

`RUL_ESTIMATION` logs every estimate (and the features of every window) to
`dinasore/resources/logs/`, so live runs can be analyzed afterwards.

## DINASORE versions

Two versions of DINASORE ran on the testbed, and the choice matters:

| Nodes | Version | Why |
|---|---|---|
| Control devices, sampling devices, AdM node | tag [`v0.1`](https://github.com/DIGI2-FEUP/dinasore/tree/v0.1) (`3357668`, February 2021) | `CONNECT_MQTT` and `READ_MQTT`, which wait for messages, work with this version |
| PdM nodes | `master`, commit `d428b74` (July 2023) | Installed later; they only publish (`WRITE_MQTT`, the print order), which works with both versions |

```bash
git clone https://github.com/DIGI2-FEUP/dinasore.git
git -C dinasore checkout v0.1        # control, sampling, and AdM nodes
```

With the newer versions (from `v1.0`, September 2021, which changed 14 files of the
runtime core, including how each function block thread runs), `CONNECT_MQTT` and
`READ_MQTT` stopped receiving messages on the AdM node. Replacing the runtime with the
one of the arm devices, `v0.1`, solved it. The runtime core of the arm devices
(`dinasore/core/*.py`) was confirmed to be identical to `v0.1`.

The applications were developed and deployed with the Eclipse 4diac IDE **1.11.0**,
the version indicated in the DINASORE documentation.

## Mosquitto

The brokers ran with the configuration installed by the package, unchanged.
Mosquitto 1.x (e.g. 1.5.7, in Raspberry Pi OS Buster) accepts anonymous connections
from the network by default. **Mosquitto 2.0 and later** only accept local
connections unless configured otherwise. On those versions, create
`/etc/mosquitto/conf.d/testbed.conf` with:

```
listener 1883
allow_anonymous true
```

This is only acceptable on an isolated laboratory network.
