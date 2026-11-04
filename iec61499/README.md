# iec61499/ - function blocks of the article version

The function blocks of the system evaluated in the article, for the
[DINASORE](https://github.com/DIGI2-FEUP/dinasore) runtime.

**The 4diac system file (`.sys`) of this version was not preserved.** It was later edited
into the dissertation version. The applications can be read in the screenshots below, and
the system file of the later version, with a similar structure, is on the
[`main` branch](https://github.com/pedrodsmota/iiot-autonomous-maintenance/tree/main/iec61499).

| Function block | Device | Function |
|---|---|---|
| `ARM_CONTROL` | Control device | Executes the movements in `arm_config/<FILE>.xml` when triggered |
| `IMU_STREAM` | Sampling device | Runs the C sampling utility and streams the samples over MQTT |
| `HI_ESTIMATION` | PdM node | Launches the online script in its own virtual environment and exposes the estimates |
| `HI_THRESHOLD` | PdM node | Emits the print order when the estimated HI falls below the threshold |
| `PRINT_QUEUE`, `PRINT_START`, `PRINT_CHECK` | AdM node | Queue the orders, start the print, and monitor it through OctoPrint |
| `CONNECT_MQTT`, `READ_MQTT`, `WRITE_MQTT` | All | Adapted from the [DINASORE function blocks](https://github.com/DIGI2-FEUP/dinasore_function_blocks) (`FBs/MQTT`) |

## Thresholds

The thresholds are on the Health Index, selected empirically from the training life cycle:

| Arm | Sensor | HI threshold |
|---|---|---|
| #1 | ADXL345 | 0.4 |
| #2 | Piezoelectric | 0.3 |
| #3 | MPU9250 | 0.3 |

## Screenshots of the applications

- [System configuration](../docs/figures/4diac/system_full.png)
- Control and sampling: [Arm #1](../docs/figures/4diac/pipeline_arm1.png), [Arm #2](../docs/figures/4diac/pipeline_arm2.png), [Arm #3](../docs/figures/4diac/pipeline_arm3.png)
- PdM nodes: [Arm #1](../docs/figures/4diac/pipeline_arm1dt.png), [Arm #2](../docs/figures/4diac/pipeline_arm2dt.png), [Arm #3](../docs/figures/4diac/pipeline_arm3dt.png)
- [AdM node](../docs/figures/4diac/pipeline_adm.png)
