"""
K-Fold Cross Validation Training Script.
Trains the model K times on different splits to ensure statistical robustness.
Uses the same configuration as train.py (src/training/config.py).

Usage:
    python scripts/train_cv.py
    python scripts/train_cv.py --folds 1 --max-epochs 30 --logger csv     # quick trial on fold 1
Checkpoints go to checkpoints/<run_name>/fold_<k>/ ; evaluate them with
    python scripts/evaluate.py --cv --checkpoints-dir checkpoints/<run_name>
"""

import argparse
import sys
import warnings
from pathlib import Path

import pytorch_lightning as pl

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import get_data_dicts, get_kfold_splits, get_loaders
from src.training.config import TrainConfig, add_train_args
from src.training.factory import build_module, build_trainer, build_transforms

def main():
    parser = add_train_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds (standard 5-fold CV).")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subset of folds to train, 1-based. Default: all.")
    args = parser.parse_args()
    cfg = TrainConfig.from_args(args)
    print(f"\nConfig: {cfg}")

    # --- 1. Gather all data dictionaries and split (same folds as evaluate.py) ---
    data_dicts = get_data_dicts(PROJECT_ROOT / "data" / "raw")
    folds = get_kfold_splits(data_dicts, n_splits=args.n_splits)
    selected = args.folds or list(range(1, args.n_splits + 1))
    print(f"Found {len(data_dicts)} total patients. {args.n_splits}-Fold Cross Validation, folds: {selected}")

    # --- 2. Main Cross-Validation Loop ---
    for fold in selected:
        train_files, val_files = folds[fold - 1]
        print(f"\n{'='*50}\n🚀 STARTING FOLD {fold}/{args.n_splits}\n{'='*50}")
        print(f"Training samples: {len(train_files)} | Validation samples: {len(val_files)}")

        pl.seed_everything(cfg.seed)  # same init for every fold

        # Pre-processing cached on disk, shared by all folds
        train_transforms, val_transforms = build_transforms(cfg)
        train_loader, val_loader = get_loaders(
            train_files, val_files, train_transforms, val_transforms, batch_size=cfg.batch_size
        )

        # Fresh model, loss, optimizer, logger and checkpoint for each fold
        pl_module = build_module(cfg)
        trainer = build_trainer(
            cfg,
            run_name=f"{cfg.name}_fold_{fold}",
            ckpt_dir=PROJECT_ROOT / "checkpoints" / cfg.name / f"fold_{fold}",
            group=cfg.name,
        )
        trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Finish the W&B run so the next fold starts a fresh dashboard
        if cfg.logger == "wandb":
            import wandb
            wandb.finish()

    print("\n✅ CROSS-VALIDATION COMPLETE!")

if __name__ == "__main__":
    main()
