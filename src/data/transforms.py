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
)

def get_train_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0)
) -> Compose:
    """
    Builds the MONAI transform pipeline for training.
    
    Args:
        spatial_size: The patch size for the 3D random crop.
        spacing: The target physical resolution in mm (X, Y, Z).
        
    Returns:
        Compose: The composed MONAI transforms pipeline.
    """
    
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
            num_samples=4, # How many patches to extract per volume loaded
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
        
        # Add slight noise to the image only
        RandGaussianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.1)
    ])
    
    return transforms

def get_val_transforms(
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0)
) -> Compose:
    """
    Builds the MONAI transform pipeline for validation and inference.
    It standardizes the geometry and intensity, but keeps the full volume intact.
    
    Args:
        spacing: The target physical resolution in mm (X, Y, Z).
        
    Returns:
        Compose: The composed MONAI transforms pipeline.
    """
    
    transforms = Compose([
        # --- 1. Load and Standardize ---
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Resample physical spacing (Must match training spacing exactly)
        Spacingd(
            keys=["image", "label"], 
            pixdim=spacing, 
            mode=("bilinear", "nearest")
        ),
        
        # --- 2. Intensity Normalization ---
        # Soft tissue windowing (Must match training windowing exactly)
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0, a_max=250.0,
            b_min=0.0, b_max=1.0,
            clip=True
        )
    ])
    
    return transforms