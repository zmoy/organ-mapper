"""
Main training script (single hold-out split) using PyTorch Lightning.
All settings come from src/training/config.py and can be overridden on the command line.

Usage:
    python scripts/train.py
    python scripts/train.py --loss gdl --max-epochs 30 --logger csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import pytorch_lightning as pl

# Ignore MONAI's FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import get_train_val_dataloaders
from src.training.config import TrainConfig, add_train_args
from src.training.factory import build_module, build_trainer, build_transforms


def main():
    cfg = TrainConfig.from_args(add_train_args(argparse.ArgumentParser(description=__doc__)).parse_args())
    pl.seed_everything(cfg.seed)
    print(f"\nConfig: {cfg}")

    # --- 1. Data Pipeline (pre-processing cached on disk) ---
    print("\nInitializing DataLoaders...")
    train_transforms, val_transforms = build_transforms(cfg)
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=PROJECT_ROOT / "data" / "raw",
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=cfg.batch_size,
        num_workers=0  # Keep at 0 for Windows
    )

    # --- 2. Model, Loss, Optimizer, Trainer ---
    print("\nInitializing Model & Lightning Module...")
    pl_module = build_module(cfg)
    trainer = build_trainer(cfg, run_name=cfg.name, ckpt_dir=PROJECT_ROOT / "checkpoints" / cfg.name)

    # --- 3. Train ---
    print("\nStarting Training...")
    trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
