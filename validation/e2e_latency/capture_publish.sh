#!/bin/bash
# End-to-end response time - publication timestamps on the PdM node (Section 6.5).
#
# Records the kernel timestamp of every outgoing MQTT PUBLISH on /PRINT sent to the
# broker of the AdM node (first transmissions only), one epoch time per line.
# Usage: sudo ./capture_publish.sh <adm-node-ip> [interface]   (Ctrl+C to stop)
# Output: t_send.csv (copy it to the AdM node for analyze.py)
set -euo pipefail
ADM_IP=${1:?usage: $0 <adm-node-ip> [interface]}
IFACE=${2:-wlan0}

tshark -i "$IFACE" -l \
  -f "tcp dst port 1883 and dst host $ADM_IP" \
  -Y 'mqtt.msgtype == 3 && mqtt.topic == "/PRINT" && mqtt.dupflag == 0 && !tcp.analysis.retransmission' \
  -T fields -e frame.time_epoch > t_send.csv
