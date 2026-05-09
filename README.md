# Unsupervised Neural Network for Multi-Genre Music Generation

Course: CSE425/EEE474 Neural Networks



This repository contains our group implementation for symbolic MIDI music generation. We implemented Tasks 1, 2, and 3 from the project specification:

- Task 1: LSTM Autoencoder music generator
- Task 2: LSTM Variational Autoencoder (VAE) generator
- Task 3: Transformer-based long-sequence music generator

The project uses MAESTRO-style MIDI data converted into event tokens. The generated MIDI files are included in this repository and can also be uploaded to Google Drive for submission.

## Public Links

Replace the placeholders below after final upload.

- GitHub repository: `https://github.com/Atef123Ahabab/cse425-project`
- Generated MIDI Google Drive folder: `PASTE_PUBLIC_GOOGLE_DRIVE_MIDI_FOLDER_LINK_HERE`
- Final report PDF Google Drive link: `PASTE_PUBLIC_GOOGLE_DRIVE_REPORT_PDF_LINK_HERE`
- youtube link : https://www.youtube.com/watch?v=AeNKOfc8msM
- google drive links - https://drive.google.com/drive/folders/126gWeZM5bgt6t8CZJes4sr8YIuTg9Xyn?fbclid=IwY2xjawRrjcJleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEe_wOv6-IqRsiRHqS5izwAk_8NmJug8RknwBu-Aw5DIEWG8OMAp0e5cLIS-dg_aem_kxknpUjptApGf5Gzy0kTSA




## Implemented Tasks

### Task 1: LSTM Autoencoder

Goal: learn a compressed latent representation of symbolic music and reconstruct/generate short MIDI sequences.

Main files:

- `src/models/lstm_autoencoder.py` - LSTM Autoencoder model.
- `src/train/train_task1_full.py` - full Task 1 training, validation, loss plotting, checkpoint saving, and MIDI generation script.
- `outputs/task1/generated_samples/` - five generated MIDI samples required for Task 1.
- `outputs/task1/training_loss.png` - reconstruction loss curve for Task 1.
- `outputs/task1/losses.json` - saved Task 1 train/validation loss values.

### Task 2: Variational Autoencoder

Goal: extend the autoencoder with a probabilistic latent space using mean, variance, reparameterization, and KL-divergence loss.

Main files:

- `src/models/vae_rnn.py` - LSTM VAE model with encoder mean/log-variance heads and decoder.
- `src/train/train_task2.py` - Task 2 training script with reconstruction loss, KL loss, KL annealing, MIDI generation, and latent interpolation.
- `outputs/task2/generated_samples/` - eight generated VAE MIDI samples required for Task 2.
- `outputs/task2/latent_interpolations/` - eight latent interpolation MIDI files.
- `outputs/task2/training_loss.png` - total VAE loss curve.
- `outputs/task2/kl_recon_loss.png` - reconstruction and KL component plot.
- `outputs/task2/losses.json` - saved Task 2 loss values.

### Task 3: Transformer Generator

Goal: train a decoder-style Transformer to generate longer symbolic music sequences autoregressively.

Main files:

- `src/models/transformer_decoder.py` - Transformer decoder model with token embedding, causal mask, Transformer decoder layers, and output projection.
- `src/train/train_task3.py` - Task 3 training, generation, and perplexity evaluation script.
- `outputs/task3/composition_01.mid` to `outputs/task3/composition_10.mid` - ten long generated MIDI compositions required for Task 3.
- `outputs/task3/comprehensive_results.json` - Task 3 perplexity and generated composition metrics.
- `outputs/task3/comprehensive_evaluation_report.md` - Task 3 evaluation report.
- `TASK3_DELIVERABLE_AUDIT.md` - manual audit confirming Task 3 deliverables and MIDI validity.

Verified Task 3 result:

- Perplexity: `170.11`
- Number of generated compositions: `10`
- Average rhythm diversity: `0.548`
- Average repetition ratio: `0.000`

## Generated MIDI Files

The MIDI files to upload to Google Drive are:

```text
outputs/task1/generated_samples/
  generated_sample_01.mid
  generated_sample_02.mid
  generated_sample_03.mid
  generated_sample_04.mid
  generated_sample_05.mid

outputs/task2/generated_samples/
  vae_sample_01.mid
  vae_sample_02.mid
  vae_sample_03.mid
  vae_sample_04.mid
  vae_sample_05.mid
  vae_sample_06.mid
  vae_sample_07.mid
  vae_sample_08.mid

outputs/task2/latent_interpolations/
  interpolation_01.mid
  interpolation_02.mid
  interpolation_03.mid
  interpolation_04.mid
  interpolation_05.mid
  interpolation_06.mid
  interpolation_07.mid
  interpolation_08.mid

outputs/task3/
  composition_01.mid
  composition_02.mid
  composition_03.mid
  composition_04.mid
  composition_05.mid
  composition_06.mid
  composition_07.mid
  composition_08.mid
  composition_09.mid
  composition_10.mid
```

All final Task 1, Task 2, and Task 3 MIDI files were checked with `pretty_midi`. Each final deliverable file loads successfully, contains at least 50 notes, and is longer than 5 seconds.

## Reports

- `ICLR_REPORT_COPY_PASTE.tex` - final copy-paste-ready ICLR-style LaTeX report for Overleaf.
- `GROUP_REPORT_TASK1_TASK2_TASK3.md` - Markdown group report draft.
- `Task3_Final_Report.md` - Task 3 focused report draft.
- `TASK1_REPORT.md` - Task 1 completion report.
- `TASK1_EVALUATION_REPORT.md` - Task 1 metric/evaluation notes.
- `TASK3_EVALUATION_REPORT.md` - Task 3 detailed evaluation notes.
- `task1_training_output.md` - Task 1 training log/notes.
- `OPTIMIZATION_SUMMARY.md` - notes about optimization and model improvements.

## Source Code File Guide

### Root Files

- `README.md` - this project guide and file index.
- `requirements.txt` - Python package requirements.
- `download_maestro.py` - helper script for downloading/preparing MAESTRO data.
- `evaluate_task1.py` - Task 1 evaluation helper.
- `evaluate_midi_quality.py` - MIDI quality and metric evaluation helper.
- `run_task1.sh` - root-level Task 1 run script.

### Configuration

- `configs/task1.yaml` - Task 1 configuration.
- `configs/task2.yaml` - Task 2 configuration.
- `configs/task3.yaml` - Task 3 configuration, including batch size, sequence length, learning rate, and device.

### Data and Preprocessing

- `data/raw/` - processed token data used by the training scripts.
- `data/raw/train/` - training token `.npy` files.
- `data/raw/val/` - validation token `.npy` files.
- `data/raw/vocab.json` - vocabulary mapping for note-on, time-shift, and padding tokens.
- `maestro-v3.0.0/` - local extracted MAESTRO dataset and metadata.
- `maestro-v3.0.0-midi.zip` - downloaded MAESTRO MIDI archive.
- `src/preprocess.py` - MIDI-to-token and token-to-MIDI conversion functions.
- `scripts/preprocess_maestro.py` - preprocesses MAESTRO MIDI files into `.npy` token sequences.
- `scripts/preprocess_subset.py` - helper for preprocessing a smaller subset.
- `scripts/preprocess_groove.py` - helper for Groove MIDI preprocessing.

### Data Loading

- `src/data.py` - PyTorch `Dataset`, collate function, and `DataLoader` utilities for token sequences.
- `src/__init__.py` - package marker for the `src` module.

### Models

- `src/models/lstm_autoencoder.py` - Task 1 LSTM Autoencoder.
- `src/models/vae_rnn.py` - Task 2 LSTM VAE.
- `src/models/transformer_decoder.py` - Task 3 Transformer decoder.
- `src/models/__init__.py` - package marker for model imports.

### Training

- `src/train/train_task1.py` - minimal Task 1 training scaffold.
- `src/train/train_task1_full.py` - complete Task 1 training and generation pipeline.
- `src/train/train_task2.py` - complete Task 2 VAE training, generation, and interpolation pipeline.
- `src/train/train_task3.py` - Task 3 Transformer training, evaluation, and generation pipeline.
- `scripts/run_task1.sh` - shell script for Task 1.
- `scripts/run_task2.sh` - shell script for Task 2.
- `scripts/run_task3.sh` - shell script for Task 3.
- `scripts/generate_submission_midis.py` - regenerates and verifies submission-ready MIDI files.

### Utilities

- `src/utils/metrics.py` - pitch histogram, rhythm diversity, repetition ratio, and report utilities.
- `src/utils/midi_io.py` - MIDI export utilities for generated token sequences.
- `src/utils/sampling.py` - sampling helper functions.
- `src/utils/__init__.py` - package marker for utility imports.

### Notebooks

- `notebooks/Task1_LSTM_Autoencoder.ipynb` - Task 1 exploratory notebook.
- `notebooks/Task2_VAE_MultiGenre.ipynb` - Task 2 exploratory notebook.
- `notebooks/Task3_Transformer_Generator.ipynb` - Task 3 exploratory notebook.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries include:

- PyTorch
- NumPy
- pretty_midi
- mido
- matplotlib
- PyYAML
- tqdm
- tensorboard

## How to Run

On Windows PowerShell, enable UTF-8 and a non-interactive Matplotlib backend:

```powershell
$env:PYTHONUTF8='1'
$env:MPLBACKEND='Agg'
```

Train/regenerate Task 1:

```bash
python src/train/train_task1_full.py --data_dir data/raw --output_dir outputs/task1 --epochs 3 --batch_size 16 --seq_len 64 --hidden_dim 64 --latent_dim 16 --num_samples 5 --device cpu
```

Train/regenerate Task 2:

```bash
python src/train/train_task2.py --data_dir data/raw --output_dir outputs/task2 --epochs 3 --batch_size 16 --seq_len 64 --hidden_dim 64 --latent_dim 16 --num_samples 8 --device cpu
```

Generate/verify final MIDI deliverables:

```bash
python scripts/generate_submission_midis.py
```

Task 3 training/evaluation is handled by:

```bash
python src/train/train_task3.py --config configs/task3.yaml --data_dir data/raw --output_dir outputs/task3 --epochs 5 --generate --evaluate
```

## Results Summary

| Model | Main Metric | Generated MIDI | Rhythm Diversity | Repetition Ratio |
|---|---:|---:|---:|---:|
| Task 1 LSTM Autoencoder | Validation loss 4.9362 | 5 files | 0.010 | 0.000 |
| Task 2 LSTM VAE | Validation loss 4.7944 | 8 files | 0.010 | 0.000 |
| Task 3 Transformer | Perplexity 170.11 | 10 files | 0.548 | 0.000 |

## Group Member Contributions

Replace the placeholders below with final group member names before submission.

| Member | Contribution |
|---|---|
| Member 1 | Dataset preprocessing, vocabulary creation, Task 1 LSTM Autoencoder training, Task 1 MIDI generation |
| Member 2 | Task 2 VAE implementation, KL-divergence loss, KL annealing, latent interpolation generation |
| Member 3 | Task 3 Transformer implementation, perplexity evaluation, long-sequence MIDI generation |
| Member 3| Metric analysis, report writing, README preparation, artifact organization, Google Drive upload |

## Final Submission Checklist

- Source code repository or ZIP
- Generated MIDI files
- Final report PDF in ICLR format
- Evaluation results and plots
- Public Google Drive folder for MIDI files
- Public Google Drive link for final report PDF
- Group member contribution table

## Notes and Limitations

- This implementation uses a custom event-token representation rather than full REMI tokenization.
- Task 1 and Task 2 use token-level cross-entropy because the representation is discrete.
- Task 1 and Task 2 MIDI export uses fixed note durations, which lowers rhythm diversity.
- Task 3 includes causal masking but does not currently include learned positional embeddings.
- Human listening survey results are not included in the current repository.
