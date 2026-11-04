"""
End-to-end response time - observer on the AdM node (dissertation, Section 6.5).

Subscribes to the print order topic, alongside the AdM application, and records:
  t_recv   arrival of the order at the AdM node (local clock, ns);
  t_lo     last poll of GET /api/job that did NOT see the printer printing;
  t_hi     first poll that saw "Printing" (or TIMEOUT).
The transition logged by OctoPrint must fall between t_lo and t_hi (cross-check).

Appends one line per order to t_printing.csv: t_recv,t_lo,t_hi
Start it, and log_sync_uncertainty.sh, before the PdM node sends the order.

Configuration (environment variables):
  OCTO_API_KEY   OctoPrint API key (required)
  OCTO_URL       default http://localhost
  MQTT_BROKER    default localhost
"""

import os
import sys
import threading
import time

import paho.mqtt.client as mqtt
import requests

TOPIC = "/PRINT"
POLL_S = 0.02        # polling period of the printer state
TIMEOUT_S = 120

OCTO_URL = os.environ.get("OCTO_URL", "http://localhost")
BROKER = os.environ.get("MQTT_BROKER", "localhost")
try:
    API_KEY = os.environ["OCTO_API_KEY"]
except KeyError:
    sys.exit("Set the environment variable OCTO_API_KEY.")

session = requests.Session()
session.headers["X-Api-Key"] = API_KEY
log = open("t_printing.csv", "a", buffering=1)


def watch(t_recv):
    """Poll the job state until the printer reports Printing."""
    last_not_printing = ""
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        t_req = time.time_ns()
        try:
            state = session.get(f"{OCTO_URL}/api/job", timeout=2).json().get("state", "")
        except Exception as e:
            print("request error:", e)
            time.sleep(POLL_S)
            continue
        t_resp = time.time_ns()
        if state.startswith("Printing"):
            log.write(f"{t_recv},{last_not_printing},{t_resp}\n")
            print(f"Printing detected ({(t_resp - t_recv) / 1e6:.1f} ms after arrival)")
            return
        last_not_printing = t_req
        time.sleep(POLL_S)
    log.write(f"{t_recv},{last_not_printing},TIMEOUT\n")
    print("TIMEOUT")


def on_message(client, userdata, msg):
    t_recv = time.time_ns()
    print("order received")
    threading.Thread(target=watch, args=(t_recv,), daemon=True).start()


try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho-mqtt 2.x
except AttributeError:
    client = mqtt.Client()                                   # paho-mqtt 1.x

client.on_message = on_message
client.connect(BROKER)
client.subscribe(TOPIC, qos=1)
print("listening on", TOPIC)
client.loop_forever()
