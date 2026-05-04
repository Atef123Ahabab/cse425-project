#!/usr/bin/env python3
"""
Load and evaluate Task 1 LSTM Autoencoder model.

This script demonstrates how to:
1. Load the trained model from checkpoint
2. Generate new MIDI samples
3. Compute evaluation metrics
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.preprocess import tokens_to_midi, build_vocabulary


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Assuming vocab_size from checkpoint or default
    vocab_size = 229
    model = LSTMAutoencoder(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=512,
        latent_dim=128,
        num_layers=2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_sample(model, device, seq_len=256, temperature=1.0, seed=None):
    """Generate a MIDI sample from the model."""
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    with torch.no_grad():
        # Sample random latent code
        z = torch.randn(1, model.latent_fc.out_features, device=device)
        
        # Decode to get token logits
        logits = model.decode(z, seq_len)  # Shape: [1, seq_len, vocab_size]
        logits = logits.squeeze(0) / temperature
        
        # Sample tokens
        probs = torch.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return tokens.cpu().numpy()


def interpolate_latent(model, device, z1, z2, steps=5):
    """Interpolate between two latent codes to create a smooth transition."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            logits = model.decode(z_interp.to(device), seq_len=256)
            logits = logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            samples.append(tokens.cpu().numpy())
    
    return samples


def evaluate_reconstruction(model, data_loader, device):
    """Evaluate model reconstruction loss on a dataset."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch, lengths in data_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits.view(-1, logits.size(-1)), batch.view(-1))
            total_loss += loss.item()
            count += 1
    
    return total_loss / count if count > 0 else 0.0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and evaluate Task 1 model')
    parser.add_argument('--checkpoint', type=str, default='outputs/maestro_task1/checkpoint_best.pt',
                        help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/maestro_task1/test_samples',
                        help='Output directory for test samples')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    
    # Generate samples
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nGenerating {args.num_samples} samples...")
    for i in range(args.num_samples):
        tokens = generate_sample(model, device, seq_len=256, temperature=1.0, seed=i)
        
        # Convert to MIDI
        midi_path = os.path.join(args.output_dir, f'test_sample_{i+1:02d}.mid')
        try:
            tokens_to_midi(tokens.astype(int).tolist(), midi_path, 
                          time_bin=0.05, max_time_shift_bins=100)
            print(f"  ✓ Generated: {midi_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTest samples saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
