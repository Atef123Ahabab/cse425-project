"""Preprocessing and tokenization helpers (stubs).

Real preprocessing should convert MIDI to an event/token representation and
save processed files under `data/processed/`.
"""
import os
from typing import List, Tuple
import pretty_midi
import numpy as np
import json


def build_vocabulary(max_time_shift_bins: int = 100) -> Tuple[dict, dict]:
    """Build a vocabulary mapping tokens to ids and reverse mapping.

    Tokens:
      - note_on_0..127
      - time_shift_1..max_time_shift_bins (1 means one bin)
      - pad
    Returns (vocab, inv_vocab)
    """
    vocab = {}
    idx = 0
    for p in range(128):
        vocab[f"note_on_{p}"] = idx
        idx += 1
    for b in range(1, max_time_shift_bins + 1):
        vocab[f"time_shift_{b}"] = idx
        idx += 1
    vocab["pad"] = idx
    inv = {v: k for k, v in vocab.items()}
    return vocab, inv


def midi_to_events(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, int]]:
    """Extract (time, pitch) events from a PrettyMIDI object.

    Returns a sorted list of (time, pitch) for note_on events across all instruments.
    """
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append((n.start, n.pitch))
    events.sort(key=lambda x: x[0])
    return events


def midi_to_tokens(midi_path: str, time_bin: float = 0.05, max_shift_bins: int = 100) -> List[int]:
    """Convert a MIDI file to a list of integer tokens.

    - `time_bin` is the quantization bin in seconds for time shifts.
    - `max_shift_bins` is the maximum number of bins represented by a single time_shift token.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = midi_to_events(pm)
    if not events:
        return []
    vocab, _ = build_vocabulary(max_shift_bins)
    tokens: List[int] = []
    last_t = events[0][0]
    # ensure initial time shift if first event is not at time 0
    if last_t > 0:
        bins = int(round(last_t / time_bin))
        while bins > 0:
            take = min(bins, max_shift_bins)
            tokens.append(vocab[f"time_shift_{take}"])
            bins -= take

    for t, pitch in events:
        delta = t - last_t
        if delta < -1e-6:
            # non-monotonic, skip
            last_t = t
            continue
        bins = int(round(delta / time_bin))
        # if bins>0 add time_shift tokens (may be zero for simultaneous events)
        while bins > 0:
            take = min(bins, max_shift_bins)
            tokens.append(vocab[f"time_shift_{take}"])
            bins -= take
        # add note_on token
        if 0 <= pitch < 128:
            tokens.append(vocab[f"note_on_{pitch}"])
        last_t = t

    return tokens


def tokens_to_midi(tokens: List[int], out_path: str, time_bin: float = 0.05, max_time_shift_bins: int = 100):
    """Convert token sequence back to a MIDI file (very simple reconstruction).

    This reconstruction places notes at quantized times with fixed short duration.
    """
    vocab, inv = build_vocabulary(max_time_shift_bins)
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    time = 0.0
    for tok in tokens:
        lbl = inv.get(int(tok), None)
        if lbl is None:
            continue
        if lbl.startswith("time_shift_"):
            b = int(lbl.split("_")[-1])
            time += b * time_bin
        elif lbl.startswith("note_on_"):
            p = int(lbl.split("_")[-1])
            # add a short note of fixed duration
            n = pretty_midi.Note(velocity=100, pitch=p, start=time, end=time + 0.08)
            piano.notes.append(n)
    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pm.write(out_path)


def preprocess_folder(input_dir: str, out_dir: str, time_bin: float = 0.05, max_shift_bins: int = 100, split: float = 0.9):
    """Process all .mid files under `input_dir` and write token .npy files under `out_dir`.

    Creates `out_dir/train/` and `out_dir/val/` with .npy files and saves a `vocab.json`.
    """
    import glob
    files = glob.glob(os.path.join(input_dir, "**", "*.mid"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No .mid files found in {input_dir}")
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # save vocab
    vocab, inv = build_vocabulary(max_shift_bins)
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    # split
    n_train = int(len(files) * split)
    files_sorted = sorted(files)
    train_files = files_sorted[:n_train]
    val_files = files_sorted[n_train:]

    def process_list(file_list, target_dir):
        for src in file_list:
            try:
                toks = midi_to_tokens(src, time_bin=time_bin, max_shift_bins=max_shift_bins)
                if not toks:
                    continue
                base = os.path.splitext(os.path.basename(src))[0]
                out_path = os.path.join(target_dir, base + ".npy")
                np.save(out_path, np.array(toks, dtype=np.int32))
            except Exception as e:
                print(f"Skipping {src}: {e}")

    process_list(train_files, train_dir)
    process_list(val_files, val_dir)

    print(f"Preprocessed {len(train_files)} train and {len(val_files)} val files into {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time_bin", type=float, default=0.05)
    p.add_argument("--max_shift", type=int, default=100)
    p.add_argument("--split", type=float, default=0.9)
    args = p.parse_args()
    preprocess_folder(args.input, args.out, time_bin=args.time_bin, max_shift_bins=args.max_shift, split=args.split)
