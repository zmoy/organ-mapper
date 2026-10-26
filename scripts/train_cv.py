"""
K-Fold Cross Validation Training Script.
Trains the model K times on different splits to ensure statistical robustness.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from monai.losses import GeneralizedDiceLoss
from monai.data import Dataset, DataLoader
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import get_data_dicts, get_kfold_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.builder import get_model
from src.training.lightning_module import OrganMapperModule

def main():
    # --- 1. Configuration ---
    data_dir = PROJECT_ROOT / "data" / "raw"
    spatial_size = (96, 96, 96)
    batch_size = 2
    max_epochs = 100
    k_splits = 5 # Standard 5-Fold CV

    # --- 2. Gather all data dictionaries ---
    # We fetch all training images to split them dynamically
    data_dicts = get_data_dicts(data_dir)

    print(f"Found {len(data_dicts)} total patients. Starting {k_splits}-Fold Cross Validation.")

    # --- 3. K-Fold split (shared with evaluate.py) ---
    folds = get_kfold_splits(data_dicts, n_splits=k_splits)

    # --- 4. Main Cross-Validation Loop ---
    for fold, (train_files, val_files) in enumerate(folds):
        print(f"\n{'='*50}\n🚀 STARTING FOLD {fold + 1}/{k_splits}\n{'='*50}")

        print(f"Training samples: {len(train_files)} | Validation samples: {len(val_files)}")

        # Create localized DataLoaders
        train_ds = Dataset(data=train_files, transform=get_train_transforms(spatial_size=spatial_size))
        val_ds = Dataset(data=val_files, transform=get_val_transforms(spatial_size=spatial_size))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Initialize Model (Reset weights for each fold!)
        model = get_model(model_name="unet", spatial_size=spatial_size)
        loss_function = GeneralizedDiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        
        pl_module = OrganMapperModule(
            model=model,
            loss_function=loss_function,
            optimizer_class=AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
            val_roi_size=spatial_size
        )

        # Unique W&B Logger and Checkpoint directory for this fold
        wandb_logger = WandbLogger(project="organ-mapper", name=f"unet_fold_{fold + 1}")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/fold_{fold + 1}/",
            filename="unet-best-{epoch:02d}-{val/dice:.2f}",
            save_top_k=1,
            monitor="val/dice",
            mode="max"
        )

        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
            log_every_n_steps=5
        )
        
        # Train this fold
        trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Finish the W&B run so the next fold starts a fresh dashboard
        import wandb
        wandb.finish()

    print("\n✅ CROSS-VALIDATION COMPLETE!")

if __name__ == "__main__":
    main()