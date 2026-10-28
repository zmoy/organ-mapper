"""
Module handling data transformations for the Organ Mapper project.
Uses MONAI transform chains for reproducible pre-processing.
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandAffined,
    RandGaussianNoised,
    Rand3DElasticd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandShiftIntensityd,
)

def get_train_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0),
    num_samples: int = 1,
    intensity_shift: bool = True
) -> Compose:
    """
    Builds the MONAI transform pipeline for training.

    Args:
        spatial_size: The patch size for the 3D random crop.
        spacing: The target physical resolution in mm (X, Y, Z).
        num_samples: Patches drawn per volume each time it is loaded (the
            loader then yields batch_size * num_samples patches).
        intensity_shift: Whether to apply RandShiftIntensityd (absent from
            the original baseline).

    Returns:
        Compose: The composed MONAI transforms pipeline.
    """
    
    shift = [RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5)] if intensity_shift else []

    transforms = Compose([
        # --- 1. Load and Standardize ---
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Resample physical spacing
        Spacingd(
            keys=["image", "label"], 
            pixdim=spacing, 
            mode=("bilinear", "nearest")
        ),
        
        # --- 2. Intensity Normalization ---
        # Soft tissue windowing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0, a_max=250.0,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        
        # --- 3. Patch Extraction ---
        # Ensure patches contain organs frequently (ratio pos/neg)
        # We look for pixels where label > 0 (any organ)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1,  # Weight for positive samples (center is an organ)
            neg=1,  # Weight for negative samples (center is background)
            num_samples=num_samples, # How many patches to extract per volume loaded
            image_key="image",
            image_threshold=0,
        ),
        
        # --- 4. Data Augmentation (Optional but recommended) ---
        # Slight rotations and translations to prevent overfitting
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=0.5,
            spatial_size=spatial_size,
            rotate_range=(0.1, 0.1, 0.1), # slight rotation
            scale_range=(0.1, 0.1, 0.1)   # slight scaling
        ),
        # Elastic deformation (simulates soft tissue and breathing motion)
        # Rand3DElasticd(
        #     keys=["image", "label"],
        #     mode=("bilinear", "nearest"),
        #     prob=0.2,
        #     sigma_range=(5, 7),
        #     magnitude_range=(50, 150)
        # ),

        # --- 5. Data Augmentation (Intensity) ---
        # Intensity offset (simulates variations in scanner calibration)
        *shift,
        # Add slight noise to the image only
        RandGaussianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.1)
    ])
    
    return transforms

def get_val_transforms(
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0),
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    full_volume: bool = False
) -> Compose:
    """
    Builds the MONAI transform pipeline for validation.

    Deterministic: a given volume always yields the same output, so the
    validation Dice is comparable from one epoch to the next.

    Args:
        spacing: The target physical resolution in mm (X, Y, Z).
        spatial_size: Patch size used when full_volume is False.
        full_volume: If True, keep the whole resampled volume (for
            sliding-window evaluation). Otherwise return a single fixed patch
            centered on the bounding box of the labeled organs.

    Returns:
        Compose: The composed MONAI transforms pipeline.
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=spacing, 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0, a_max=250.0,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
    ]

    if not full_volume:
        # Deterministic single patch: crop to the organs' bounding box, then
        # center crop/pad to the patch size (no randomness, unlike RandCrop).
        transforms += [
            CropForegroundd(keys=["image", "label"], source_key="label"),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        ]

    return Compose(transforms)