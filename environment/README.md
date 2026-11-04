# environment/ - software versions

| File | Where | Used by |
|---|---|---|
| `requirements.txt` | Any computer | Offline pipeline (`prognostics/`), notebooks, online replay. Versions used are in its header (macOS, Python 3.9.6) |
| `requirements-pdm-online.txt` | PdM nodes, own virtual environment | Online estimation scripts (`online/`), launched by `RUL_ESTIMATION` |
| `requirements-pdm-dinasore.txt` | PdM nodes | DINASORE `master` (`d428b74`), Python 3.8 |
| `requirements-adm-dinasore.txt` | AdM node | DINASORE `v0.1`, Python 3.7 |

The `-dinasore` and `-pdm-online` files are `pip freeze` outputs of the testbed, kept
as a record; newer versions will most likely work. The control devices run DINASORE
`v0.1` as well and additionally need `adafruit-circuitpython-servokit` for
`ARM_CONTROL`. Which DINASORE version runs on which node, and why, is explained in
[`../iec61499/README.md`](../iec61499/README.md#dinasore-versions).

Other software on the testbed: Eclipse 4diac IDE 1.11.0 (the version indicated in
the DINASORE documentation), OctoPi 1.1.0 with OctoPrint 1.11.7, and the Mosquitto
broker of the Raspberry Pi OS packages.
