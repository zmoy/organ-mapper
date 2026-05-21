"""
Module for data loading and batching using MONAI.
Handles data splitting and PyTorch DataLoader instantiation.
"""

from pathlib import Path
from monai.data import CacheDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_data_dicts(data_dir: str | Path) -> list[dict]:
    """
    Scans the directory and pairs training images with their labels.
    
    Args:
        data_dir: Path to the raw data directory.
        
    Returns:
        A list of dictionaries, e.g., [{"image": img_path, "label": lbl_path}, ...]
    """
    data_dir = Path(data_dir)
    images = sorted(list((data_dir / "imagesTr").glob("*.nii.gz")))
    labels = sorted(list((data_dir / "labelsTr").glob("*.nii.gz")))
    
    # Ensure every image has a corresponding label
    assert len(images) == len(labels), "Mismatch between number of images and labels."
    
    data_dicts = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(images, labels)]
    return data_dicts


def get_train_val_dataloaders(
    data_dir: str | Path,
    train_transforms,
    val_transforms=None,
    batch_size: int = 2,
    val_split: float = 0.2,
    cache_rate: float = 1.0,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Creates the training and validation DataLoaders.
    
    Args:
        data_dir: Path to the root data directory.
        train_transforms: MONAI transform pipeline for training.
        val_transforms: MONAI transform pipeline for validation.
        batch_size: Number of volumes per batch.
        val_split: Proportion of data to use for validation.
        cache_rate: Percentage of data to cache in RAM (0.0 to 1.0).
        num_workers: Number of CPU workers (keep at 0 on Windows/Jupyter).
        
    Returns:
        train_loader, val_loader
    """
    data_dicts = get_data_dicts(data_dir)
    
    # Split the dataset deterministically for reproducibility
    train_files, val_files = train_test_split(
        data_dicts, 
        test_size=val_split, 
        random_state=42 # Fixed seed
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # --- Training DataLoader ---
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    
    # Note: If batch_size=2 and RandCrop has num_samples=4, 
    # the effective batch size output by the loader will be 8 patches.
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    # --- Validation DataLoader ---
    # We will instantiate the validation loader once its transforms are defined
    val_loader = None
    if val_transforms is not None:
        val_ds = CacheDataset(
            data=val_files, 
            transform=val_transforms, 
            cache_rate=cache_rate, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, # Always 1 for 3D validation
            shuffle=False, 
            num_workers=num_workers
        )
        
    return train_loader, val_loader