"""
Trigger of the replacement part - round-trip latency test
(article, Section 3.3, Table 5).

Runs on the PdM node of one arm, in place of its prescriptive logic. In each trial,
the script cancels any active print and waits for the printer to become idle,
publishes a print order on the topic to which the AdM node subscribes, and measures
the time until the AdM node publishes the acknowledgment that the start of the print
was requested from OctoPrint. PRINT_QUEUE is bypassed for this test.

Only the standard library and paho-mqtt (1.x or 2.x) are needed, so it runs on any
of the Raspberry Pi of the testbed.

Configuration (environment variables, never hard-coded):
  OCTO_URL       e.g. http://<adm-node-ip>
  OCTO_API_KEY   OctoPrint API key
  MQTT_BROKER    address of the broker on the AdM node

Usage, on the PdM node of each arm:
  export OCTO_URL=http://<adm-node-ip> OCTO_API_KEY=<key> MQTT_BROKER=<adm-node-ip>
  python3 print_validation.py --arm 1            # 30 trials
Writes validation_results_arm<N>.csv with the columns Arm, Trial, Success,
Latency_ms (0.0 when the acknowledgment did not arrive). The files of the three
arms concatenated give results.csv.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from threading import Event

import paho.mqtt.client as mqtt

TRIGGER_TOPIC = "/PRINT"
ACK_TOPIC = "/PRINT_ACK"
MQTT_PAYLOAD = '{"event": "START_PRINT"}'

ack_event = Event()
t1_timestamp = 0.0


def on_connect(client, userdata, flags, rc, properties=None):   # paho 1.x and 2.x
    client.subscribe(ACK_TOPIC)


def on_message(client, userdata, msg):
    global t1_timestamp
    if msg.topic == ACK_TOPIC:
        t1_timestamp = time.perf_counter()
        ack_event.set()


def octoprint(url, api_key, body=None):
    """GET (body None) or POST a JSON body to the OctoPrint REST API."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"X-Api-Key": api_key,
                                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def reset_printer(octo_url, api_key):
    """Cancel any active print and wait for OctoPrint to be operational."""
    job_url = f"{octo_url}/api/job"
    try:
        octoprint(job_url, api_key, {"command": "cancel"})
    except urllib.error.HTTPError:
        pass   # 409 when there is no active job
    while True:
        try:
            if octoprint(job_url, api_key).get("state", "").lower() == "operational":
                return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.5)


def make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho 2.x
    except AttributeError:
        return mqtt.Client()                                   # paho 1.x


def main():
    ap = argparse.ArgumentParser(description="Round-trip latency of the print order (Section 6.4)")
    ap.add_argument("--arm", type=int, required=True, help="Arm whose PdM node runs the test")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for the acknowledgment")
    a = ap.parse_args()

    try:
        octo_url = os.environ["OCTO_URL"].rstrip("/")
        api_key = os.environ["OCTO_API_KEY"]
        broker = os.environ["MQTT_BROKER"]
    except KeyError as missing:
        sys.exit(f"Set the environment variable {missing} (see the docstring).")

    client = make_client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, 1883, 60)
    client.loop_start()

    results = []
    print(f"--- Starting tests for Arm {a.arm} ({TRIGGER_TOPIC}) ---")
    for trial in range(1, a.trials + 1):
        reset_printer(octo_url, api_key)
        time.sleep(1)   # let the system settle before the next trial
        ack_event.clear()
        t0 = time.perf_counter()            # immediately before publishing the order
        client.publish(TRIGGER_TOPIC, payload=MQTT_PAYLOAD)
        success = ack_event.wait(timeout=a.timeout)
        if success:
            latency_ms = (t1_timestamp - t0) * 1000
            print(f"Trial {trial}: Success - {latency_ms:.2f} ms")
        else:
            latency_ms = 0.0
            print(f"Trial {trial}: Failed - Timeout waiting for {ACK_TOPIC}")
        results.append({"Arm": a.arm, "Trial": trial, "Success": success, "Latency_ms": latency_ms})

    client.loop_stop()
    client.disconnect()

    out = f"validation_results_arm{a.arm}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Arm", "Trial", "Success", "Latency_ms"])
        w.writeheader()
        w.writerows(results)

    ok = [r["Latency_ms"] for r in results if r["Success"]]
    print("\n### Validation Summary ###")
    if ok:
        sd = statistics.stdev(ok) if len(ok) > 1 else 0.0
        print(f"Arm {a.arm}: {len(ok)}/{a.trials} Success | Mean: {statistics.mean(ok):.0f} ms "
              f"(σ = {sd:.0f} ms)")
    else:
        print(f"Arm {a.arm}: 0/{a.trials} Success")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
