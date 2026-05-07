#!/usr/bin/env python3
"""
Download MAESTRO dataset reliably.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

def download_maestro():
    url = "https://storage.googleapis.com/magenta-datasets/maestro/maestro-v3.0.0-midi.zip"
    filename = "maestro-v3.0.0-midi.zip"
    extract_path = "maestro-data"
    
    # Create directories
    Path(extract_path).mkdir(exist_ok=True)
    
    print(f"Downloading MAESTRO dataset from {url}...")
    print("This may take 5-10 minutes (~1.3 GB)...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = min(block_num * block_size, total_size)
        pct = 100 * downloaded / total_size
        print(f"\rProgress: {pct:.1f}% ({downloaded/1e9:.2f} GB / {total_size/1e9:.2f} GB)", end="")
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print("\n✓ Download complete!")
        
        print("Extracting...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✓ Extraction complete!")
        
        # List extracted structure
        midi_files = list(Path(extract_path).rglob("*.midi")) + list(Path(extract_path).rglob("*.mid"))
        print(f"✓ Found {len(midi_files)} MIDI files")
        print(f"\nNext step: python scripts/preprocess_maestro.py --input_dir {extract_path}/2004 --output_dir data/raw --max_files 100")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_maestro()
