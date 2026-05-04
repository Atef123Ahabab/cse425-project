#!/usr/bin/env python3
"""
Preprocess MAESTRO MIDI dataset for Task 1.

Converts MIDI files to token sequences and saves them as .npy files.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pretty_midi

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import midi_to_tokens, build_vocabulary


def preprocess_maestro(input_dir: str, output_dir: str, max_files: int = None):
    """Process MAESTRO MIDI files to token sequences."""
    
    # Find all MIDI files
    midi_files = glob.glob(os.path.join(input_dir, "**", "*.midi"), recursive=True)
    midi_files += glob.glob(os.path.join(input_dir, "**", "*.mid"), recursive=True)
    midi_files = sorted(list(set(midi_files)))  # Remove duplicates and sort
    
    if max_files:
        midi_files = midi_files[:max_files]
    
    print(f"Found {len(midi_files)} MIDI files")
    
    if not midi_files:
        print("No MIDI files found!")
        return
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Build and save vocabulary
    vocab, inv = build_vocabulary(max_time_shift_bins=100)
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split into train/val
    split_idx = int(0.9 * len(midi_files))
    train_files = midi_files[:split_idx]
    val_files = midi_files[split_idx:]
    
    # Process files
    def process_file_list(file_list, target_dir, split_name):
        successful = 0
        failed = 0
        for midi_path in tqdm(file_list, desc=f"Processing {split_name}"):
            try:
                tokens = midi_to_tokens(midi_path, time_bin=0.05, max_shift_bins=100)
                if not tokens or len(tokens) < 10:
                    failed += 1
                    continue
                
                # Save tokens
                base_name = os.path.splitext(os.path.basename(midi_path))[0]
                out_path = os.path.join(target_dir, base_name + ".npy")
                np.save(out_path, np.array(tokens, dtype=np.int32))
                successful += 1
            except Exception as e:
                failed += 1
        
        return successful, failed
    
    train_ok, train_fail = process_file_list(train_files, train_dir, "train")
    val_ok, val_fail = process_file_list(val_files, val_dir, "val")
    
    print(f"\nPreprocessing complete!")
    print(f"Train: {train_ok} successful, {train_fail} failed")
    print(f"Val: {val_ok} successful, {val_fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO dataset")
    parser.add_argument("--input_dir", type=str, default="/Users/atefahabab/Downloads/maestro-v3.0.0",
                        help="Path to MAESTRO dataset root")
    parser.add_argument("--output_dir", type=str, default="data/processed/maestro",
                        help="Output directory for processed data")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    args = parser.parse_args()
    
    preprocess_maestro(args.input_dir, args.output_dir, args.max_files)
