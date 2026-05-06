#!/usr/bin/env python3
"""
Task 1: LSTM Autoencoder Training and Generation Script

Trains an LSTM autoencoder on MAESTRO dataset and generates MIDI samples.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data import MidiEventDataset, collate_fn
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.preprocess import tokens_to_midi


def compute_batch_metrics(logits, targets, pad_id):
    """Return token accuracy and token count for a batch."""
    preds = logits.argmax(dim=-1)
    valid_mask = targets.ne(pad_id)
    valid_tokens = valid_mask.sum().item()

    if valid_tokens == 0:
        return 0.0, 0

    correct = ((preds == targets) & valid_mask).sum().item()
    token_accuracy = correct / valid_tokens
    return token_accuracy, valid_tokens


def train_epoch(model, dataloader, optimizer, device, criterion, pad_id):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0
    kl_loss = 0.0  # Task 1 is Autoencoder (not VAE), so KL = 0
    
    for batch_idx, (x, lengths) in enumerate(tqdm(dataloader, desc="Training")):
        x = x.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        
        # Compute reconstruction loss (cross-entropy between logits and input tokens)
        recon_loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        loss = recon_loss  # Task 1: no KL term
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_accuracy, batch_tokens = compute_batch_metrics(logits.detach(), x, pad_id)
        total_accuracy += batch_accuracy * batch_tokens
        total_tokens += batch_tokens
    
    avg_loss = total_loss / max(len(dataloader), 1)
    avg_accuracy = total_accuracy / max(total_tokens, 1)
    return avg_loss, avg_accuracy, 0.0


def eval_epoch(model, dataloader, device, criterion, pad_id):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, lengths in tqdm(dataloader, desc="Validating"):
            x = x.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            total_loss += loss.item()
            batch_accuracy, batch_tokens = compute_batch_metrics(logits, x, pad_id)
            total_accuracy += batch_accuracy * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / max(len(dataloader), 1)
    avg_accuracy = total_accuracy / max(total_tokens, 1)
    return avg_loss, avg_accuracy, 0.0  # Task 1: KL = 0 (Autoencoder, not VAE)


def generate_sample(model, device, vocab_size, seq_len=256, temperature=1.0):
    """Generate a MIDI sample using the trained autoencoder.
    
    Strategy: Sample random latent code and decode it to generate music.
    """
    model.eval()
    with torch.no_grad():
        # Sample random latent code
        z = torch.randn(1, model.latent_fc.out_features, device=device)
        
        # Decode to get token logits
        logits = model.decode(z, seq_len)  # Shape: [1, seq_len, vocab_size]
        
        # Sample from logits using temperature
        logits = logits.squeeze(0)  # [seq_len, vocab_size]
        logits = logits / temperature
        
        probs = torch.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [seq_len]
        
    return tokens.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder on MAESTRO")
    parser.add_argument("--data_dir", type=str, default="data/processed/maestro",
                        help="Path to preprocessed data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/maestro_task1",
                        help="Output directory for checkpoints and samples")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent dimension")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of MIDI samples to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Error: vocab.json not found at {vocab_path}")
        print("Please run: python scripts/preprocess_maestro.py")
        sys.exit(1)
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_id = vocab.get("pad", vocab_size - 1)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MidiEventDataset(args.data_dir, split="train", seq_len=args.seq_len)
    val_dataset = MidiEventDataset(args.data_dir, split="val", seq_len=args.seq_len)
    
    if len(train_dataset) == 0:
        print(f"Error: No training data found in {args.data_dir}/train/")
        print("Please run: python scripts/preprocess_maestro.py")
        sys.exit(1)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    print("Note: Task 1 is LSTM Autoencoder (not VAE), so KL divergence = 0")
    model = LSTMAutoencoder(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=1
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    kl_losses = []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        train_loss, train_accuracy, train_kl = train_epoch(model, train_loader, optimizer, device, criterion, pad_id)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validate
        val_loss, val_accuracy, val_kl = eval_epoch(model, val_loader, device, criterion, pad_id)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        kl_losses.append(val_kl)  # Task 1: KL = 0

        train_perplexity = float(torch.exp(torch.tensor(train_loss)))
        val_perplexity = float(torch.exp(torch.tensor(val_loss)))
        loss_gap = abs(train_loss - val_loss)
        sync_status = "✓ SYNC" if loss_gap < 0.1 else f"GAP: {loss_gap:.4f}"
        
        print(
            f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Acc: {train_accuracy:.4f}/{val_accuracy:.4f} | PPL: {train_perplexity:.2f}/{val_perplexity:.2f} | "
            f"KL: 0.0000 | {sync_status}"
        )
        
        # Learning rate scheduling (Cosine Annealing)
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = os.path.join(args.output_dir, "checkpoint_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, best_checkpoint)
            print(f"  ✓ Best checkpoint saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: validation loss did not improve for {patience} epochs.")
                break

    print(f"\nTraining stopped after {epoch+1} epochs.")
    print(f"Best validation reconstruction loss: {best_val_loss:.4f}")
    print(f"Note: Task 1 is LSTM Autoencoder (not VAE), so KL divergence is always 0.")
    print("\nPlotting training curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Autoencoder: Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"), dpi=150)
    print(f"Saved plot to {args.output_dir}/training_loss.png")
    
    # Save loss data
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "kl_losses": kl_losses,  # Task 1: KL = 0 for all epochs
        "pad_id": pad_id,
    }
    with open(os.path.join(args.output_dir, "losses.json"), "w") as f:
        json.dump(loss_data, f)
    
    # Load best model
    print("\nLoading best model for generation...")
    best_checkpoint = os.path.join(args.output_dir, "checkpoint_best.pt")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Generate MIDI samples
    print(f"\nGenerating {args.num_samples} MIDI samples...")
    samples_dir = os.path.join(args.output_dir, "generated_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    for i in range(args.num_samples):
        # Generate token sequence
        tokens = generate_sample(model, device, vocab_size, seq_len=args.seq_len, temperature=1.0)
        
        # Convert tokens to MIDI
        tokens_list = [int(t) for t in tokens]
        midi_path = os.path.join(samples_dir, f"generated_sample_{i+1:02d}.mid")
        
        try:
            tokens_to_midi(tokens_list, midi_path, time_bin=0.05, max_time_shift_bins=100)
            print(f"Generated: {midi_path}")
        except Exception as e:
            print(f"Error generating MIDI {i+1}: {e}")
    
    print("\nTask 1 training and generation complete!")
    print(f"Checkpoints: {args.output_dir}")
    print(f"Generated samples: {samples_dir}")


if __name__ == "__main__":
    main()
