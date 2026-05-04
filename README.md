# Music Generative Models (Unsupervised)

Project scaffold for Tasks 1–3 (LSTM AE, VAE, Transformer). Code is organized
for later training once datasets are added under `data/raw/`.

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place MIDI dataset under `data/raw/<genre>/*.mid` (optional for now).

3. Run a training script (example):

```bash
python src/train/train_task1.py --config configs/task1.yaml --data_dir data/raw --output_dir outputs/task1
```

Repository layout

- `data/` — raw and processed dataset files (not included)
- `notebooks/` — exploration notebooks (one per task)
- `src/` — source code (models, data, training)
- `configs/` — YAML configs per task
- `outputs/` — checkpoints, logs, generated MIDI

For details see the notebooks and `src/` modules.
