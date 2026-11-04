"""
End-to-end response time - analysis on the AdM node (dissertation, Section 6.5, Table 6.8).

Inputs, in the current directory:
  t_send.csv            publication times of the orders on the PdM node (capture_publish.sh),
                        copied from the PdM node
  t_printing.csv        arrival and printer state times (observer.py)
  sync_uncertainty.csv  clock synchronization uncertainty (log_sync_uncertainty.sh)
and the OctoPrint log (default ~/.octoprint/logs/octoprint.log).

Prints, per run:
  MQTT A->B              publication on the PdM node -> arrival at the AdM node
  Send -> Printing (lo)  lower bound: last poll that did not see Printing
  Send -> Printing (hi)  upper bound: first poll that saw Printing
  the printer state transitions logged by OctoPrint after the order (1 ms resolution)
"""

import argparse
import csv
import datetime
import os
import statistics
from decimal import Decimal

ap = argparse.ArgumentParser(description="End-to-end response time analysis")
ap.add_argument("--octoprint-log", default=os.path.expanduser("~/.octoprint/logs/octoprint.log"))
a = ap.parse_args()

sends = [int(Decimal(line.strip()) * 10**9) for line in open("t_send.csv") if line.strip()]
rows = list(csv.reader(open("t_printing.csv")))
sync_err = max(float(r[1]) for r in csv.reader(open("sync_uncertainty.csv")))
print(f"orders sent: {len(sends)}  arrivals: {len(rows)}")
assert len(sends) == len(rows), "different number of sent and received orders!"

mqtt_ms, lo_ms, hi_ms = [], [], []
for ts, (t_recv, t_lo, t_hi) in zip(sends, rows):
    if not t_lo or t_hi == "TIMEOUT":
        print("measurement discarded")
        continue
    mqtt_ms.append((int(t_recv) - ts) / 1e6)
    lo_ms.append((int(t_lo) - ts) / 1e6)
    hi_ms.append((int(t_hi) - ts) / 1e6)


def summary(v):
    return f"median {statistics.median(v):8.2f}  min {min(v):8.2f}  max {max(v):8.2f} ms"


print(f"valid n = {len(mqtt_ms)}   synchronization uncertainty = +/-{sync_err:.2f} ms")
print("MQTT A->B             ", summary(mqtt_ms))
print("Send -> Printing (lo) ", summary(lo_ms))
print("Send -> Printing (hi) ", summary(hi_ms))

t0 = sends[-1]
print("OctoPrint states after the order:")
for line in open(a.octoprint_log):
    if "Changing monitoring state" not in line:
        continue
    t = int(datetime.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1e9)
    if 0 <= (t - t0) < 600e9:        # only the 10 min after the order
        print(f"  {(t - t0) / 1e6:9.1f} ms  {line.split('state ')[-1].strip()}")
