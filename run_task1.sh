#!/bin/bash
# Task 1: LSTM Autoencoder - Quick Start Guide

echo "=== Task 1: LSTM Autoencoder Setup ==="

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

# Step 2: Preprocess dataset
echo "Step 2: Preprocessing MAESTRO dataset..."
python3 scripts/preprocess_maestro.py \
  --input_dir /Users/atefahabab/Downloads/maestro-v3.0.0 \
  --output_dir data/processed/maestro \
  --max_files 100

# Step 3: Train model
echo "Step 3: Training LSTM Autoencoder..."
python3 src/train/train_task1_full.py \
  --data_dir data/processed/maestro \
  --output_dir outputs/maestro_task1 \
  --epochs 30 \
  --batch_size 16 \
  --seq_len 256 \
  --hidden_dim 512 \
  --latent_dim 128 \
  --lr 1e-3 \
  --num_samples 5

echo "=== Training Complete ==="
echo "Outputs saved to: outputs/maestro_task1/"
echo "- Checkpoints: checkpoint_*.pt"
echo "- Generated samples: generated_samples/*.mid"
echo "- Loss plot: training_loss.png"
