"""
Main training script using PyTorch Lightning and Weights & Biases.
"""

import os
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from monai.losses import DiceCELoss

# Ignore MONAI's FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.dataset import get_train_val_dataloaders
from src.models.builder import get_model
from src.training.lightning_module import OrganMapperModule

def main():
    # --- 1. Configuration ---
    data_dir = PROJECT_ROOT / "data" / "raw"
    spatial_size = (96, 96, 96)
    batch_size = 1
    learning_rate = 1e-4
    max_epochs = 100
    
    # --- 2. Data Pipeline ---
    print("\nInitializing DataLoaders...")
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        train_transforms=get_train_transforms(spatial_size=spatial_size),
        val_transforms=get_val_transforms(spatial_size=spatial_size),
        batch_size=batch_size,
        cache_rate=0.0, # No caching for now
        num_workers=0 # Keep at 0 for Windows
    )

    # --- 3. Model, Loss, Optimizer ---
    print("\nInitializing Model & Lightning Module...")
    # model = get_model(model_name="unet", spatial_size=spatial_size)
    model = get_model(model_name="swin_unetr", spatial_size=spatial_size)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    
    pl_module = OrganMapperModule(
        model=model,
        loss_function=loss_function,
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": learning_rate, "weight_decay": 1e-5},
        val_roi_size=spatial_size
    )

    # --- 4. Callbacks & Logger ---
    # W&B Logger initialization
    wandb_logger = WandbLogger(project="organ-mapper", name="swin_unetr_sota")
    
    # Save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="swin-best-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        monitor="val/dice",
        mode="max"
    )

    # --- 5. Trainer ---
    print("\nStarting Training...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed", # AMP: Automatic Mixed Precision
        log_every_n_steps=5,
        accumulate_grad_batches=8 # Gradient accumulation to simulate larger batch size
    )
    
    # Start the training loop
    trainer.fit(
        model=pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

if __name__ == "__main__":
    main()