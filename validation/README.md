# validation/ - validation tests

Scripts and procedures of the validation (dissertation, Chapter 6). The overview of
all tests, their metrics, and their success criteria is in Section 6.1 (Table 6.1).

| Folder | Test | Dissertation |
|---|---|---|
| `control_acquisition/` | One hour of operation: production cycles, MQTT round trip, continuity of the sensor streams | §6.2 |
| `trigger_test/` | 30 print orders sent from the PdM node of each arm; round-trip latency to the acknowledgment of the AdM node | §6.4 |
| `e2e_latency/` | Response time in the end-to-end trials: publication of the print order -> printer printing | §6.1.2, §6.5 |

## One hour of operation (`control_acquisition/`)

The line ran for one hour with the three arms streaming to their PdM nodes, while
three measurements ran in parallel:

| Script | Where | Measures (Table 6.2) |
|---|---|---|
| `cycle_duration.py --broker <arm1-control-ip>` | any computer on the testbed network | Production cycles: interval between consecutive messages on `/ARM1_12/RUN` (number of cycles, mean duration, jitter as max - min) |
| `mqtt_round_trip.py --broker <arm-control-ip>` | same, once per arm | Round trip of 1000 MQTT messages to the broker of the arm (rows "MQTT jitter": mean +/- sd and maximum) |
| `record_arrivals.c` | PdM node of each arm | Arrival time of every sample of the sensor stream (plus the sensor time stamp or sample index, with `-s`) |

The stream logs are then analyzed with `analyze_streams.py`, which computes the sensor
jitter (standard deviation of the intervals between arrivals), the longest gap, and
the samples lost, and draws Figures 6.1 and 6.2:

```bash
python analyze_streams.py --arm 1 arm1_stream.csv --plot mqtt_inter_arrival.pdf
python analyze_streams.py --arm 2 arm2_stream.csv
python analyze_streams.py --arm 3 arm3_stream.csv --plot mqtt_inter_arrival_arm3.pdf
```

Results of the dissertation:

| | Arm #1 (ADXL345, MQTT) | Arm #2 (piezo, serial) | Arm #3 (MPU9250, MQTT) |
|---|---|---|---|
| Production cycles | 84 of 84 completed, 43.65 +/- 0.02 s, max - min 0.14 s | | |
| MQTT round trip, mean +/- sd (max), ms | 5.70 +/- 1.70 (24.21) | 5.81 +/- 2.23 (28.57) | 6.08 +/- 4.57 (97.74) |
| Samples received | 7,826,751 | 20,168,058 | 14,328,000 |
| Sensor jitter (ms) | 3.96 | - | 13.41 |
| Longest gap (s) | 0.246 | 0.695 | 0.910 |
| Samples lost (%) | 0.54 | - | 0.00 |

The stream logs (about 1 GB) are not published. The coordination latency between the
arms reported in Table 6.2 (12.4 +/- 3.8 ms, maximum 114 ms) was measured during the
same test, but the script that measured it was not preserved.

Note on Arm #1: this test predates the revision of the online pipeline (Section 6.2),
when the ADXL345 stream was configured at 2800 Hz; the log has 2174 samples per
second. The samples-lost metric counts the intervals between sensor time stamps
longer than 1.5 nominal periods, so it does not reflect this lower rate.

## Trigger of the replacement part (`trigger_test/`)

`print_validation.py` runs on the PdM node of each arm (usage in its docstring) and
writes one CSV per arm; it needs only the Python standard library and `paho-mqtt`.
The scripts run on the testbed were not preserved; this one reproduces their
procedure and output format.
`trigger_test/results.csv` has the 90 trials of the dissertation (Table 6.6): all
successful, mean round-trip latency 350 / 339 / 388 ms (sigma 98 / 83 / 109 ms) for
Arms #1 / #2 / #3.

`IIOT_AM.sys` has the final applications, which do not publish `/PRINT_ACK`. To repeat this
test, connect `/PRINT` directly to `PRINT_START`, without `PRINT_QUEUE`, and publish on
`/PRINT_ACK` once `PRINT_START` has requested the print.

## End-to-end response time (procedure)

The PdM node and the AdM node were in different laboratories, and their applications
could not be modified or stopped, so the measurement used observers outside them:

1. **Clock synchronization with chrony**: the PdM node of the arm under test acts as
   the time server of the AdM node (only the relative offset matters). The uncertainty
   is logged every 2 s as `|offset| + delay/2 + dispersion` (from `chronyc tracking`
   and `chronyc ntpdata`); the worst value of each run is reported.
2. **Timestamps**: publication of the order - kernel timestamp of the outgoing MQTT
   PUBLISH packet (tshark) on the PdM node; arrival - a separate MQTT client on the
   AdM node; printer state transitions - `octoprint.log` (1 ms resolution).
3. **Cross-check**: the observer polls `GET /api/job` every 20 ms after the arrival;
   the transition from the log must fall between the last poll before and the first
   poll after it.

### Running it (`e2e_latency/`)

| Where | Script | Output |
|---|---|---|
| AdM node | `log_sync_uncertainty.sh <pdm-node-ip>`, started at least 10 min before the rest | `sync_uncertainty.csv` |
| AdM node | `OCTO_API_KEY=<key> python3 observer.py` | `t_printing.csv` |
| PdM node | `sudo ./capture_publish.sh <adm-node-ip> wlan0` | `t_send.csv` |
| AdM node | `python3 analyze.py`, after copying `t_send.csv` from the PdM node | summary + OctoPrint transitions |

Then let the arm reach its threshold: the print order of the real application is
what gets measured. The observer needs `paho-mqtt` and `requests`; the PdM node needs
`tshark`.

chrony, a minimal configuration for this setup:

```
# PdM node (time server), /etc/chrony/chrony.conf
local stratum 8
allow <subnet of the AdM node>/24

# AdM node (client), /etc/chrony/chrony.conf
server <pdm-node-ip> iburst prefer
```

Results of the end-to-end trials (Table 6.7; times counted from the start of each trial, noted by hand during the trials):

| Arm | Threshold crossed | Response time | Print completed | Complete failure | Advance notice | Margin |
|---|---|---|---|---|---|---|
| #1 (ADXL345) | 4 h 09 min | 215 ms | 7 h 46 min | 8 h 43 min | 4 h 34 min | 57 min |
| #2 (piezoelectric) | 4 h 16 min | 209 ms | 7 h 51 min | 8 h 48 min | 4 h 32 min | 57 min |
| #3 (MPU9250) | 4 h 07 min | 207 ms | 7 h 42 min | 8 h 55 min | 4 h 48 min | 73 min |

Stages of the response time (Table 6.8): 214.9 / 208.5 / 206.5 ms for Arms #1 / #2 / #3, of which 3.5-11.3 ms
network, 136-164 ms processing on the AdM node, and 39-67 ms for the printer to start.
