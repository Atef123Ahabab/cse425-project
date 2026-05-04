"""Training script for Task 1: LSTM Autoencoder (minimal runnable scaffold).

This script runs a short smoke test using randomly generated data from
`src.data.MidiEventDataset` so you can run it before adding a dataset.
"""
import argparse
import torch
import torch.nn as nn
from src.data import make_dataloader
from src.models.lstm_autoencoder import LSTMAutoencoder
import os


def train_epoch(model, loader, optim, device):
    model.train()
    total = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, lengths in loader:
        x = x.to(device)
        optim.zero_grad()
        logits = model(x)
        # shift targets
        target = x
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="outputs/task1")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    import json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load vocab to get correct vocab_size
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
    else:
        vocab_size = 129
    loader = make_dataloader(args.data_dir, batch_size=8, seq_len=200)
    model = LSTMAutoencoder(vocab_size=vocab_size)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optim, device)
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f}")
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1:03d}.pt"))


if __name__ == "__main__":
    main()
