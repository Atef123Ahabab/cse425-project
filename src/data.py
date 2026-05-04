"""Dataset loader and simple collate utilities.

These are minimal stubs so notebooks and training scripts can import them
even before a dataset is provided.
"""
from typing import List, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import json


class MidiEventDataset(Dataset):
    """Dataset that loads tokenized .npy files saved by `src.preprocess`.

    Expects structure:
      data_dir/train/*.npy and data_dir/val/*.npy

    Each .npy file is a 1D integer array of token ids.
    """

    def __init__(self, data_dir: Optional[str] = None, split: str = "train", seq_len: Optional[int] = None):
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        self.files = []
        if data_dir is not None:
            pattern = os.path.join(data_dir, split, "*.npy")
            self.files = sorted(glob.glob(pattern))
        # fallback: empty dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)
        # optionally trim or pad sequence to seq_len
        if self.seq_len is not None:
            if arr.shape[0] >= self.seq_len:
                arr = arr[: self.seq_len]
            else:
                pad_len = self.seq_len - arr.shape[0]
                pad_value = 0
                # try to read pad value from vocab if present
                vocab_path = os.path.join(self.data_dir, "vocab.json")
                if os.path.exists(vocab_path):
                    try:
                        with open(vocab_path, "r") as f:
                            vocab = json.load(f)
                            pad_value = vocab.get("pad", 0)
                    except Exception:
                        pad_value = 0
                arr = np.concatenate([arr, np.full((pad_len,), pad_value, dtype=np.int32)])
        return torch.from_numpy(arr).long()


def collate_fn(batch: List[torch.Tensor]):
    # pad to max length in batch
    lengths = [b.size(0) for b in batch]
    maxlen = max(lengths)
    pad_value = 0
    out = torch.full((len(batch), maxlen), pad_value, dtype=torch.long)
    for i, b in enumerate(batch):
        out[i, : b.size(0)] = b
    return out, torch.tensor(lengths, dtype=torch.long)


def make_dataloader(data_dir: Optional[str], split: str = "train", batch_size: int = 16, shuffle: bool = True, seq_len: Optional[int] = None):
    ds = MidiEventDataset(data_dir, split=split, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
