#!/usr/bin/env python3
"""Preprocess a subset (first N) of MIDI files from a folder using src.preprocess.

Usage:
  PYTHONPATH=. python3 scripts/preprocess_subset.py --input /path/to/maestro --out data/processed/maestro_subset --max_files 500
"""
import argparse
import glob
import os
import numpy as np
from src.preprocess import midi_to_tokens, build_vocabulary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_files", type=int, default=500)
    p.add_argument("--time_bin", type=float, default=0.05)
    p.add_argument("--max_shift", type=int, default=100)
    p.add_argument("--split", type=float, default=0.9)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.mid*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("No MIDI files found.")
        return

    files = files[: args.max_files]
    out_dir = args.out
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    vocab, inv = build_vocabulary(args.max_shift)
    # save vocab
    import json

    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    n_train = int(len(files) * args.split)
    train_files = files[:n_train]
    val_files = files[n_train:]

    def process_list(file_list, target_dir):
        for src in file_list:
            try:
                toks = midi_to_tokens(src, time_bin=args.time_bin, max_shift_bins=args.max_shift)
                if not toks:
                    continue
                base = os.path.splitext(os.path.basename(src))[0]
                out_path = os.path.join(target_dir, base + ".npy")
                np.save(out_path, np.array(toks, dtype=np.int32))
            except Exception as e:
                print(f"Skipping {src}: {e}")

    process_list(train_files, train_dir)
    process_list(val_files, val_dir)

    print(f"Wrote {len(train_files)} train and {len(val_files)} val files to {out_dir}")


if __name__ == "__main__":
    main()
