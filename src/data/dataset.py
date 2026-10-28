"""
Module for data loading and batching using MONAI.
Handles data splitting and PyTorch DataLoader instantiation.
"""

from pathlib import Path
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.data.utils import pickle_hashing
from sklearn.model_selection import KFold, train_test_split

# On-disk cache of the deterministic part of the pre-processing (data/ is gitignored)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

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


def split_train_val(
    data_dicts: list[dict], val_split: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Deterministic hold-out split (fixed seed for reproducibility)."""
    return train_test_split(data_dicts, test_size=val_split, random_state=seed)


def get_kfold_splits(
    data_dicts: list[dict], n_splits: int = 5, seed: int = 42
) -> list[tuple[list[dict], list[dict]]]:
    """
    Deterministic K-fold split, shared by train_cv.py and evaluate.py so that
    each fold's checkpoint is always evaluated on the volumes it never saw.

    Returns:
        A list of n_splits (train_files, val_files) tuples; fold k is index k - 1.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        ([data_dicts[i] for i in train_idx], [data_dicts[i] for i in val_idx])
        for train_idx, val_idx in kfold.split(data_dicts)
    ]


def get_dataset(
    files: list[dict],
    transforms,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    cache_name: str = "train",
) -> Dataset:
    """
    Builds a dataset whose deterministic transforms (load, orientation, spacing,
    intensity windowing) are computed once and cached on disk. Random transforms
    (crop, augmentations) still run on the fly at every epoch.

    Args:
        files: List of {"image": ..., "label": ...} dictionaries.
        transforms: MONAI transform pipeline.
        cache_dir: Root of the on-disk cache; None disables caching.
        cache_name: Sub-folder of cache_dir. Pipelines with a different
            deterministic prefix (train vs. val vs. full-volume val) must use
            different names.
    """
    if cache_dir is None:
        return Dataset(data=files, transform=transforms)
    return PersistentDataset(
        data=files,
        transform=transforms,
        cache_dir=Path(cache_dir) / cache_name,
        # New cache entries when the transforms change (json_hashing does NOT detect e.g. a new spacing)
        hash_transform=pickle_hashing,
    )


def get_loaders(
    train_files: list[dict],
    val_files: list[dict],
    train_transforms,
    val_transforms=None,
    batch_size: int = 2,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader | None]:
    """
    Creates the training and validation DataLoaders from explicit file lists
    (shared by the hold-out split and the K-fold cross-validation).

    Returns:
        train_loader, val_loader (None if val_transforms is None)
    """
    train_ds = get_dataset(train_files, train_transforms, cache_dir, "train")

    # Note: If batch_size=2 and RandCrop has num_samples=4,
    # the effective batch size output by the loader will be 8 patches.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = None
    if val_transforms is not None:
        val_ds = get_dataset(val_files, val_transforms, cache_dir, "val")
        val_loader = DataLoader(
            val_ds,
            batch_size=1, # Always 1 for 3D validation
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, val_loader


def get_train_val_dataloaders(
    data_dir: str | Path,
    train_transforms,
    val_transforms=None,
    batch_size: int = 2,
    val_split: float = 0.2,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Creates the training and validation DataLoaders for the hold-out split.

    Args:
        data_dir: Path to the root data directory.
        train_transforms: MONAI transform pipeline for training.
        val_transforms: MONAI transform pipeline for validation.
        batch_size: Number of volumes per batch.
        val_split: Proportion of data to use for validation.
        cache_dir: Root of the on-disk pre-processing cache (None = no cache).
        num_workers: Number of CPU workers (keep at 0 on Windows/Jupyter).

    Returns:
        train_loader, val_loader
    """
    data_dicts = get_data_dicts(data_dir)

    # Split the dataset deterministically for reproducibility
    train_files, val_files = split_train_val(data_dicts, val_split=val_split)

    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    return get_loaders(
        train_files, val_files, train_transforms, val_transforms,
        batch_size=batch_size, cache_dir=cache_dir, num_workers=num_workers
    )
