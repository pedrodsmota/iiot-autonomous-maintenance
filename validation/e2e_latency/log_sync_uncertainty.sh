#!/bin/bash
# End-to-end response time - clock synchronization uncertainty on the AdM node (Section 6.5).
#
# The AdM node is a chrony client of the PdM node of the arm under test. Every 2 s,
# logs  |offset| + delay/2 + dispersion  (ms) from `chronyc tracking` and
# `chronyc ntpdata <server>`; analyze.py reports the worst value of the run.
# Start it at least 10 minutes before the test, so that chrony has converged.
# Usage: ./log_sync_uncertainty.sh <pdm-node-ip>   (Ctrl+C to stop)
# Output: appends "epoch,uncertainty_ms" lines to sync_uncertainty.csv
set -u   # keep logging even if one chronyc call fails
SERVER=${1:?usage: $0 <pdm-node-ip>}

while true; do
  { chronyc -c tracking; chronyc ntpdata "$SERVER"; } |
    awk -F'[,:]' -v t="$(date +%s)" '
      NR==1             { s = $5 < 0 ? -$5 : $5 }   # system time offset (s)
      /Peer delay/      { d = $2 + 0 }
      /Peer dispersion/ { p = $2 + 0 }
      END               { printf "%s,%.3f\n", t, (s + d/2 + p) * 1000 }'
  sleep 2
done >> sync_uncertainty.csv
