"""
Offline clinical evaluation script.
Computes absolute Dice Score and Hausdorff Distance (HD95) on full validation volumes
without crashing system RAM.
"""

import torch
import sys
from pathlib import Path

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.builder import get_model
from src.training.lightning_module import OrganMapperModule
from src.data.dataset import get_train_val_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms

def main():
    # --- 1. Configuration ---
    data_dir = PROJECT_ROOT / "data" / "raw"
    # Best checkpoint from training (unet_baseline)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "unet_baseline_best.ckpt" 
    
    spatial_size = (96, 96, 96)
    num_classes = 14
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running clinical evaluation on device: {device}\n")

    # --- 2. Load Dataloaders ---
    # We need true validation labels for comparison
    # WARNING: We remove patch_crop for validation here!
    val_transforms = get_val_transforms(spatial_size=spatial_size)
    # Manually remove the last transform (RandCrop) to keep the full image
    val_transforms.transforms = tuple(val_transforms.transforms[:-1]) 
    
    _, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        train_transforms=get_train_transforms(spatial_size=spatial_size),
        val_transforms=val_transforms,
        batch_size=1, # Required for full images
        cache_rate=0.0,
        num_workers=0
    )

    # --- 3. Load Model ---
    model = get_model(model_name="unet", spatial_size=spatial_size)
    pl_module = OrganMapperModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        loss_function=torch.nn.Identity(), 
        optimizer_class=torch.optim.SGD, 
        optimizer_kwargs={}
    )
    model = pl_module.model.to(device)
    model.eval()

    # --- 4. Initialization of clinical metrics ---
    # exclude background = False to avoid inflating the score
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # HD95: The king metric in radiotherapy
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    # --- 5. Secure Evaluation Loop (Offline) ---
    print(f"Evaluating on {len(val_loader)} full patient volumes...\n")
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            print(f"Patient {i+1}/{len(val_loader)} - Sliding Window Inference...")
            
            # sw_batch_size = 1 to save VRAM
            outputs = sliding_window_inference(images, spatial_size, 1, model, overlap=0.5)
            
            # Post-processing
            val_outputs = [post_pred(i) for i in outputs]
            val_labels = [post_label(i) for i in labels]
            
            # Compute metrics for this patient
            dice_metric(y_pred=val_outputs, y=val_labels)
            hd95_metric(y_pred=val_outputs, y=val_labels)
            
            # PRO TIP: Actively clear VRAM and RAM
            del images, labels, outputs, val_outputs, val_labels
            torch.cuda.empty_cache()
            
    # --- 6. Aggregation and Display ---
    mean_dice = dice_metric.aggregate().item()
    mean_hd95 = hd95_metric.aggregate().item()
    
    print("\n" + "="*50)
    print("🏆 FINAL CLINICAL RESULTS")
    print("="*50)
    print(f"Mean Dice Score:  {mean_dice:.4f} (1.0 is perfect)")
    print(f"Mean HD95:        {mean_hd95:.2f} mm (0.0 is perfect)")
    print("="*50)

if __name__ == "__main__":
    main()