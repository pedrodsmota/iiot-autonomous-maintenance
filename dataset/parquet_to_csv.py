"""
Convert the Parquet recordings of the dataset back to CSV, for tools that do not
read Parquet. The folder structure and the file names are kept.

Usage:
  python parquet_to_csv.py --src <dataset folder>/raw --dst <output folder>
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Convert the Parquet recordings to CSV.")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    for p in sorted(a.src.rglob("*.parquet")):
        out = a.dst / p.relative_to(a.src).with_suffix(".csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(p).to_csv(out, index=False)
        print(out)


if __name__ == "__main__":
    main()
