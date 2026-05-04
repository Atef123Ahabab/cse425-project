"""Training script for Task 2: VAE (minimal scaffold).
"""
import argparse
import torch
import torch.nn as nn
from src.data import make_dataloader
from src.models.vae_rnn import RNNVAE
import os


def vae_loss(recon_logits, target, mu, logvar):
    ce = nn.functional.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)), target.view(-1))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return ce + 1e-3 * kld, ce.item(), kld.item()


def train_epoch(model, loader, optim, device):
    model.train()
    total = 0.0
    for x, lengths in loader:
        x = x.to(device)
        optim.zero_grad()
        logits, mu, logvar = model(x)
        loss, ce, kld = vae_loss(logits, x, mu, logvar)
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2.yaml")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="outputs/task2")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_dataloader(args.data_dir, batch_size=8)
    model = RNNVAE()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optim, device)
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f}")
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1:03d}.pt"))


if __name__ == "__main__":
    main()
