#!/usr/bin/env python3
"""CLI wrapper to preprocess a folder of MIDI files using src.preprocess.

Example:
  python scripts/preprocess_groove.py --input /path/to/groove --out data/processed/groove
"""
from src.preprocess import preprocess_folder
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time_bin", type=float, default=0.05)
    p.add_argument("--max_shift", type=int, default=100)
    p.add_argument("--split", type=float, default=0.9)
    args = p.parse_args()
    preprocess_folder(args.input, args.out, time_bin=args.time_bin, max_shift_bins=args.max_shift, split=args.split)


if __name__ == "__main__":
    main()
