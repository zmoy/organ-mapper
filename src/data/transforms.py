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
    RandCropByLabelClassesd,
    ClassesToIndicesd,
    DeleteItemsd,
    RandAffined,
    RandGaussianNoised,
    Rand3DElasticd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
)

NUM_CLASSES = 14  # 13 organs + background; keep in sync with get_model / OrganMapperModule
# Label ids the baseline never segments (Dice 0 in results/evaluation.json and in the fold-1 trials).
RARE_CLASSES = (4, 5, 10, 12, 13)


def get_class_ratios(rare_weight: float = 3.0, background_weight: float = 2.0) -> list[float]:
    """
    Sampling weight of each class for the patch centers: background_weight for
    the background (class 0), rare_weight for RARE_CLASSES, 1 for the other organs.
    """
    return [
        background_weight if c == 0 else rare_weight if c in RARE_CLASSES else 1.0
        for c in range(NUM_CLASSES)
    ]


def get_train_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0),
    num_samples: int = 1,
    intensity_shift: bool = True,
    sampling: str = "posneg",
    rare_weight: float = 3.0,
    augmentation: str = "light",
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
        sampling: How patch centers are drawn. "posneg": any organ voxel vs
            background, 1:1 (favours big organs). "classes": pick a class first
            (see get_class_ratios), then a voxel of that class.
        rare_weight: With sampling="classes", sampling weight of the RARE_CLASSES
            relative to the other organs (weight 1).
        augmentation: Random transforms applied to the patches (see get_augmentations):
            "light" (the original pipeline) or "strong" (nnU-Net-like ranges).

    Returns:
        Compose: The composed MONAI transforms pipeline.
    """

    augmentations = get_augmentations(augmentation, spatial_size, intensity_shift)

    if sampling == "classes":
        # The class indices are deterministic, so they land in the on-disk cache
        # (this stage comes before the first random transform).
        crop = [
            ClassesToIndicesd(
                keys="label",
                num_classes=NUM_CLASSES,
                indices_postfix="_cls_indices",
                image_key="image",
                image_threshold=0,
                max_samples_per_class=20000,  # keeps the cache small; plenty to draw centers from
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                ratios=get_class_ratios(rare_weight),
                num_classes=NUM_CLASSES,
                num_samples=num_samples,
                indices_key="label_cls_indices",
                warn=False,  # classes absent from a volume are skipped silently
            ),
            # Variable-length index lists cannot be batched
            DeleteItemsd(keys="label_cls_indices"),
        ]
    elif sampling == "posneg":
        # Patch centers on any organ voxel (pos) or background (neg), 1:1
        crop = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,  # Weight for positive samples (center is an organ)
                neg=1,  # Weight for negative samples (center is background)
                num_samples=num_samples, # How many patches to extract per volume loaded
                image_key="image",
                image_threshold=0,
            )
        ]
    else:
        raise ValueError(f"Unknown sampling '{sampling}'. Choose 'posneg' or 'classes'.")

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
        *crop,

        # --- 4. Data Augmentation (on the patches) ---
        *augmentations,
    ])

    return transforms


def get_augmentations(
    level: str = "light",
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    intensity_shift: bool = True,
) -> list:
    """
    Random transforms applied to the training patches.

    No random flip: abdominal anatomy is not symmetric and some labels are
    side-specific (kidneys 2/3, adrenal glands 12/13), so a left-right flip
    would need to swap them. No elastic deformation either: it is slow on the CPU
    and the data loading runs in the main process (num_workers=0 on Windows).

    Args:
        level: "light": the original pipeline (rotation ~6 deg and scaling +/-10 %
            half of the time, intensity offset, rare Gaussian noise). "strong":
            nnU-Net-like ranges (rotation +/-15 deg, scaling +/-15 %, noise, blur,
            brightness and gamma contrast).
        spatial_size: Patch size (output size of the affine transform).
        intensity_shift: Whether to apply RandShiftIntensityd.

    Returns:
        A list of MONAI dictionary transforms.
    """
    shift = [RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5)] if intensity_shift else []
    if level == "light":
        return [
            # Slight rotations and scaling
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                spatial_size=spatial_size,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
            # Intensity offset (simulates variations in scanner calibration)
            *shift,
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        ]
    if level == "strong":
        return [
            # Patient positioning and body size
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                spatial_size=spatial_size,
                rotate_range=(0.26, 0.26, 0.26),  # +/- 15 degrees
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="zeros",
            ),
            # Acquisition noise and resolution (image only, the label is never blurred)
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                sigma_z=(0.5, 1.0)),
            # Scanner calibration and contrast injection timing
            RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15),
            *shift,
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        ]
    raise ValueError(f"Unknown augmentation level '{level}'. Choose 'light' or 'strong'.")

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