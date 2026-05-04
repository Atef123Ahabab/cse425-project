#!/usr/bin/env bash
python src/train/train_task2.py --config configs/task2.yaml --data_dir data/raw --output_dir outputs/task2 "$@"
