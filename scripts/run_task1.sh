#!/usr/bin/env bash
python src/train/train_task1.py --config configs/task1.yaml --data_dir data/raw --output_dir outputs/task1 "$@"
