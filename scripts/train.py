"""
Main training script for the Organ Mapper project.
Orchestrates data loading, model initialization, and the training loop.
"""

import os
import torch
from torch.optim import AdamW
from monai.losses import DiceCELoss

# Import our custom modules
import sys
from pathlib import Path
# Add the project root to the Python path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.dataset import get_train_val_dataloaders
from src.models.builder import get_model

def main():
    # --- 1. Configuration ---
    # We will hardcode these for the MVP, but they should be moved to a YAML config later
    data_dir = Path("data/raw")
    spatial_size = (96, 96, 96)
    batch_size = 2
    learning_rate = 1e-4
    max_epochs = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 2. Data Pipeline ---
    print("\nInitializing DataLoaders...")
    train_transforms = get_train_transforms(spatial_size=spatial_size)
    val_transforms = get_val_transforms()
    
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=batch_size,
        cache_rate=1.0 # Adjust if VRAM/RAM is limited
    )

    # --- 3. Model Architecture (Phase 1: MVP U-Net) ---
    print("\nInitializing Model...")
    model = get_model(model_name="unet", spatial_size=spatial_size).to(device)

    # --- 4. Loss & Optimizer ---
    # DiceCELoss configured exactly as discussed
    loss_function = DiceCELoss(
        to_onehot_y=True, 
        softmax=True, 
        include_background=False
    )
    
    # AdamW ready for future Transformer integration
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print("\nSetup complete. Ready for the training loop!")
    
    # TODO: Implement the standard PyTorch training loop
    # TODO: Implement the Sliding Window Inference validation loop

if __name__ == "__main__":
    main()