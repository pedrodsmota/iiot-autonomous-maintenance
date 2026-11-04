"""
One-hour control and acquisition test - continuity of the sensor streams
(dissertation, Section 6.2, Table 6.2, Figures 6.1 and 6.2).

Input: the log of one arm, one line per sample received by its PdM node:
  Arm 1 (ADXL345, MQTT):  local_arrival_time,sensor_time   (sensor_time in seconds)
  Arm 2 (piezo, serial):  local_arrival_time
  Arm 3 (MPU9250, MQTT):  local_arrival_time,sensor_time   (sensor_time = sample index)
local_arrival_time is the Unix time (s) at which the PdM node received the sample.

Metrics:
  sensor jitter (ms)   standard deviation of the intervals between arrivals
  longest gap (s)      longest interval between two arrivals
  samples lost (%)     from the sensor time stamps:
                         Arm 1: intervals longer than 1.5 nominal periods (1/2800 s),
                                each counted as one lost sample
                         Arm 3: missing sample indices
Arms 1 and 3 also get the plot of the intervals between arrivals.

Usage:  python analyze_streams.py --arm 1 validacao_arm1.csv [--plot mqtt_inter_arrival.pdf]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

NOMINAL_FS = 2800          # rate of the ADXL345 stream at the time of this test
COLORS = {1: (80 / 255, 235 / 255, 85 / 255), 3: (187 / 255, 92 / 255, 204 / 255)}


def main():
    ap = argparse.ArgumentParser(description="Continuity of the sensor streams (Section 6.2)")
    ap.add_argument("--arm", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("log", help="CSV log of the arm")
    ap.add_argument("--plot", help="Output file of the inter-arrival plot (Arms 1 and 3)")
    a = ap.parse_args()

    df = pd.read_csv(a.log)
    arrival = df["local_arrival_time"].diff()
    duration = df["local_arrival_time"].iloc[-1] - df["local_arrival_time"].iloc[0]
    print(f"Arm {a.arm}: {len(df)} samples in {duration:.1f} s ({len(df) / duration:.1f} per second)")
    print(f"  sensor jitter (ms) : {arrival.std() * 1000:.2f}")
    print(f"  longest gap (s)    : {arrival.max():.3f}")

    if "sensor_time" in df:
        step = df["sensor_time"].diff()
        if a.arm == 1:
            lost = (step > 1.5 / NOMINAL_FS).sum()
        else:
            jumps = step[step > 1.5]
            lost = (jumps - 1).sum()
        print(f"  samples lost (%)   : {lost / (len(df) + lost) * 100:.4f}")

    if a.plot and a.arm in COLORS:
        elapsed = df["local_arrival_time"] - df["local_arrival_time"].iloc[0]
        gap_ms = arrival.max() * 1000
        plt.figure(figsize=(10, 4))
        plt.plot(elapsed, arrival * 1000, color=COLORS[a.arm], alpha=0.8, linewidth=0.5)
        plt.axhline(y=gap_ms, color="red", linestyle="--", linewidth=1.5, label=f"{gap_ms:.0f} ms")
        plt.xlabel("Time (s)")
        plt.ylabel("Inter-arrival interval (ms)")
        plt.ylim(0, 1000)          # same scale for both arms
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(a.plot, bbox_inches="tight")
        print(f"  plot saved to {a.plot}")


if __name__ == "__main__":
    main()
