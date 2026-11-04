"""
One-hour control and acquisition test - production cycles (dissertation, Section 6.2,
Table 6.2).

Subscribes to the topic that starts the first movement of the cycle (by default
/ARM1_12/RUN, on the broker of Arm #1) and takes the interval between consecutive
messages as the duration of one production cycle. Run it from any computer on the
network of the testbed; stop it with Ctrl+C at the end of the test.

Reports the number of cycles, the mean duration (+/- standard deviation), and the
cycle jitter, taken as the range of the durations (maximum - minimum).

Usage:  python cycle_duration.py --broker <arm1-control-ip> [--topic /ARM1_12/RUN]
Requires paho-mqtt 2.x.
"""

import argparse
import time

import numpy as np
import paho.mqtt.client as mqtt

durations = []
last_time = 0.0


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected. Waiting for cycles on '{userdata}'...")
    client.subscribe(userdata)


def on_message(client, userdata, msg):
    global last_time
    t_now = time.time()
    if last_time > 0:
        durations.append(t_now - last_time)
        print(f"Cycle {len(durations)} completed: {durations[-1]:.2f} s")
    else:
        print("First message received; timing cycle 1...")
    last_time = t_now


def main():
    ap = argparse.ArgumentParser(description="Duration of the production cycles")
    ap.add_argument("--broker", required=True)
    ap.add_argument("--topic", default="/ARM1_12/RUN")
    a = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=a.topic)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(a.broker, 1883, 60)
    print("Let the line run (e.g. one hour) and press Ctrl+C at the end.")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass

    print(f"\nCycles observed       : {len(durations)}")
    if durations:
        d = np.array(durations)
        print(f"Mean cycle duration   : {d.mean():.2f} s (+/- {d.std():.2f} s)")
        print(f"Cycle jitter (max-min): {d.max() - d.min():.2f} s")


if __name__ == "__main__":
    main()
