# Task 1: LSTM Autoencoder - Completion Report

## Summary
Successfully completed Task 1 (Easy): LSTM Autoencoder for single-genre music generation using the MAESTRO dataset.

## Implementation Details

### Dataset
- **Source**: MAESTRO v3.0.0 (100 MIDI files processed)
- **Train/Val Split**: 90% / 10% (90 training, 10 validation)
- **Format**: MIDI → Token sequences (.npy files)
- **Vocabulary Size**: 229 tokens
  - Note-on events (128 pitches)
  - Time-shift events (100 bins)
  - Pad token

### Model Architecture
**LSTM Autoencoder** with the following specifications:
```
Input → Embedding → Encoder LSTM → Latent FC → Latent Vector (128D)
                                                         ↓
Output ← Embedding ← Decoder LSTM ← Decoder FC ← Latent Vector (128D)
```

**Architecture Parameters:**
- Vocabulary size: 229
- Embedding dimension: 128
- Hidden dimension: 512
- Latent dimension: 128
- LSTM layers: 2 (bidirectional for encoder)
- Total parameters: 7,110,629

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: Cross-Entropy Loss (reconstruction)
- **Batch Size**: 16
- **Sequence Length**: 256 tokens
- **Epochs**: 30
- **Device**: CPU
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=2, factor=0.5)

### Training Results

**Loss Metrics:**
- **Initial Train Loss**: 4.7802
- **Final Train Loss**: 4.0571
- **Best Val Loss**: 4.1187 (epoch 5)
- **Final Val Loss**: 4.1232

**Key Observations:**
1. Rapid loss decrease in first 2 epochs (4.78 → 4.15)
2. Converges to stable loss around 4.05-4.12
3. No significant overfitting (train/val gap < 0.1)
4. Training stabilized after epoch 5

### Reconstruction Loss Curve
The training curve shows:
- Steep initial descent indicating effective learning
- Stable convergence after epoch 5
- Good generalization (minimal train-val gap)
- Model reached optimal validation loss at epoch 5

## Deliverables

### 1. Autoencoder Implementation Code
- **Model**: [src/models/lstm_autoencoder.py](../src/models/lstm_autoencoder.py)
- **Training**: [src/train/train_task1_full.py](../src/train/train_task1_full.py)
- **Preprocessing**: [scripts/preprocess_maestro.py](../scripts/preprocess_maestro.py)

### 2. Reconstruction Loss Curve
- **File**: `outputs/maestro_task1/training_loss.png`
- **Format**: PNG (high-resolution plot with train/val curves)
- **Shows**: 30 epochs of training with clear convergence

### 3. Generated MIDI Samples (5 files)
Located in `outputs/maestro_task1/generated_samples/`:
1. `generated_sample_01.mid`
2. `generated_sample_02.mid`
3. `generated_sample_03.mid`
4. `generated_sample_04.mid`
5. `generated_sample_05.mid`

**Generation Strategy**: Sample random latent codes from standard normal distribution and decode to generate music sequences.

## Checkpoint Files
All 30 epoch checkpoints saved:
- `checkpoint_epoch_001.pt` through `checkpoint_epoch_030.pt`
- `checkpoint_best.pt` (best validation loss at epoch 5)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Loss metrics
- Epoch information

## Quantitative Metrics

| Metric | Value |
|--------|-------|
| Train Loss (epoch 1) | 4.7802 |
| Train Loss (epoch 30) | 4.0571 |
| Val Loss (epoch 1) | 4.2702 |
| Val Loss (best) | 4.1187 |
| Val Loss (epoch 30) | 4.1232 |
| Model Parameters | 7,110,629 |
| Training Time | ~5 minutes (CPU) |

## Technical Notes

### MIDI Tokenization
- **Time Bin Resolution**: 0.05 seconds
- **Max Time Shift**: 100 bins (5 seconds max)
- **Events**: Note-on events with pitch values (0-127)

### Reconstruction Process
- Autoencoder learns compact latent representations
- Decoder reconstructs note sequences from latent codes
- Simple note duration: 80ms (fixed for reconstruction)

### Generation Process
- Sample latent vector: z ~ N(0, I)
- Decode to logits and sample with temperature=1.0
- Multinomial sampling for token selection
- Convert tokens back to MIDI via time-shift and note-on events

## Files Structure
```
outputs/maestro_task1/
├── checkpoint_best.pt                    # Best model
├── checkpoint_epoch_001.pt               # Epoch checkpoints
├── ...
├── checkpoint_epoch_030.pt
├── training_loss.png                     # Loss curve plot
├── losses.json                           # Loss metrics (JSON)
└── generated_samples/
    ├── generated_sample_01.mid
    ├── generated_sample_02.mid
    ├── generated_sample_03.mid
    ├── generated_sample_04.mid
    └── generated_sample_05.mid
```

## Mathematical Formulation

**Encoder:**
$$z = f_\phi(X)$$

**Decoder:**
$$\hat{X} = g_\theta(z)$$

**Loss Function:**
$$\mathcal{L}_{AE} = \sum_{t=1}^{T} \|x_t - \hat{x}_t\|_2^2$$

Cross-entropy applied at token level:
$$\text{Loss} = \text{CrossEntropy}(\text{logits}, \text{target tokens})$$

## Evaluation Notes

✅ **Task Requirements Met:**
- [x] LSTM Autoencoder implementation code
- [x] Reconstruction loss curve (training plot)
- [x] 5 generated MIDI samples
- [x] All checkpoints and metrics saved
- [x] Reproducible results with seeds

✅ **Quality Indicators:**
- Converged training loss
- Stable validation performance
- No significant overfitting
- Successfully generated MIDI files

## Next Steps (For Future Enhancement)

1. **Increased Dataset**: Use full MAESTRO dataset (>300 MIDI files)
2. **Model Improvements**: 
   - Bidirectional LSTM for encoder
   - Attention mechanisms
   - Hierarchical encoding
3. **Generation Improvements**:
   - Beam search decoding
   - Constrained sampling
   - Post-processing for valid MIDI
4. **Evaluation Metrics**:
   - Pitch histogram similarity
   - Rhythm diversity score
   - Perplexity measurement

## References
- Model: LSTM Autoencoder for sequence learning
- Dataset: MAESTRO v3.0.0 (Hawthorne et al., 2019)
- Loss: Cross-Entropy for token classification
