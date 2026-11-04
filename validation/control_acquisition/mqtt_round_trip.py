"""
One-hour control and acquisition test - MQTT round trip to the broker of an arm
(dissertation, Section 6.2, Table 6.2, rows "MQTT jitter").

Publishes a message on a test topic to which it is itself subscribed, waits for it
to come back, and repeats; the round-trip time measures the delay of the network and
of the broker of the arm, from a computer on the network of the testbed. Run once
per arm, during the test.

Usage:  python mqtt_round_trip.py --broker <arm-control-ip> [--n 1000]
Requires paho-mqtt 2.x.
"""

import argparse
import time

import numpy as np
import paho.mqtt.client as mqtt

TOPIC = "test/latency"


def main():
    ap = argparse.ArgumentParser(description="MQTT round-trip time to a broker")
    ap.add_argument("--broker", required=True)
    ap.add_argument("--n", type=int, default=1000, help="Number of messages")
    a = ap.parse_args()

    latencies, t_sent = [], [0.0]

    def send(client):
        t_sent[0] = time.time()
        client.publish(TOPIC, "ping")

    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"Connected. Sending {a.n} messages...")
        client.subscribe(TOPIC)
        send(client)

    def on_message(client, userdata, msg):
        latencies.append((time.time() - t_sent[0]) * 1000)
        if len(latencies) % 100 == 0:
            print(f"  {len(latencies)}/{a.n}")
        if len(latencies) < a.n:
            send(client)
        else:
            client.disconnect()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(a.broker, 1883, 60)
    client.loop_forever()

    lat = np.array(latencies)
    print(f"\nRound trip: mean {lat.mean():.2f} ms (+/- {lat.std():.2f}), max {lat.max():.2f} ms")


if __name__ == "__main__":
    main()
