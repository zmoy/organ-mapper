"""
Module handling data transformations for the Organ Mapper project.
Uses MONAI transform chains for reproducible pre-processing.
"""

import torch
from monai.transforms import (
    MapTransform,
    RandomizableTransform,
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
# Names of the labels 1..13, BTCV order (checked on img0007: position and volume of each label)
ORGAN_NAMES = (
    "spleen", "right kidney", "left kidney", "gallbladder", "esophagus", "liver", "stomach",
    "aorta", "inferior vena cava", "portal/splenic vein", "pancreas", "right adrenal", "left adrenal",
)
# Label ids the baseline never segments (Dice 0 in results/evaluation.json and in the fold-1 trials).
RARE_CLASSES = (4, 5, 10, 12, 13)
# Side-specific labels (right, left): kidneys and adrenal glands. In the RAS orientation the
# left-right direction is spatial axis 0 (checked on the predictions: right kidney and liver
# at high x, left kidney and spleen at low x).
LATERAL_PAIRS = ((2, 3), (12, 13))
LATERAL_AXIS = 0

# Deterministic pre-processing, shared by training, validation, evaluation and inference
# (get_preprocessing). VOXEL_SPACING in scripts/evaluate.py and src/postprocess.py must equal SPACING.
ORIENTATION = "RAS"
SPACING = (1.5, 1.5, 2.0)        # mm (X, Y, Z)
HU_WINDOW = (-175.0, 250.0)      # soft tissue window, mapped to [0, 1]


def get_preprocessing(
    keys: tuple[str, ...] = ("image", "label"),
    spacing: tuple[float, float, float] = SPACING,
) -> list:
    """
    Deterministic pre-processing: load, channel first, RAS orientation, resampling and HU windowing.

    The single definition used by the training, validation and inference pipelines, so that the
    model always sees the same input space.

    Args:
        keys: "image" first, then optionally "label" (resampled with nearest-neighbour interpolation).
        spacing: Target voxel size in mm (X, Y, Z).

    Returns:
        A list of MONAI dictionary transforms.
    """
    keys = list(keys)
    if keys[0] != "image" or not set(keys) <= {"image", "label"}:
        raise ValueError(f"keys must be ('image',) or ('image', 'label'), got {tuple(keys)}")
    mode = ("bilinear", "nearest")[:len(keys)]
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=ORIENTATION),
        Spacingd(keys=keys, pixdim=spacing, mode=mode if len(mode) > 1 else mode[0]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_WINDOW[0], a_max=HU_WINDOW[1],
            b_min=0.0, b_max=1.0,
            clip=True
        ),
    ]


def get_inference_transforms(spacing: tuple[float, float, float] = SPACING) -> Compose:
    """
    Pre-processing of an unlabelled image for inference (scripts/predict.py).

    Same steps as the training pipeline (get_preprocessing), image only; Invertd uses this
    pipeline to bring the prediction back to the patient space.

    Args:
        spacing: Target voxel size in mm (X, Y, Z).

    Returns:
        Compose: The composed MONAI transforms pipeline.
    """
    return Compose(get_preprocessing(("image",), spacing))


def swap_lateral_labels(label: torch.Tensor, pairs: tuple[tuple[int, int], ...] = LATERAL_PAIRS) -> torch.Tensor:
    """
    Swaps the right / left label values (2 <-> 3, 12 <-> 13) of a label map.

    Args:
        label: Integer-valued label map (any shape, float or integer dtype).
        pairs: (right, left) label pairs to swap.

    Returns:
        A new tensor of the same shape and type.
    """
    swapped = label.clone()
    for right, left in pairs:
        swapped[label == right] = left
        swapped[label == left] = right
    return swapped


class RandFlipLateralSwapd(RandomizableTransform, MapTransform):
    """
    Random mirroring of the patches, independently along each spatial axis, that keeps
    the side-specific labels right.

    Flipping the left-right axis puts the right kidney on the left side of the image: its
    label is then swapped with the left kidney's (same for the adrenal glands), so that a
    label always means "the kidney on this side of the image". The other axes carry no
    side-specific label and are flipped as is. Applied to the training patches only (the
    affine of the flipped patch is not updated).

    Args:
        keys: Keys to flip (image and label).
        label_key: Key whose values are swapped when LATERAL_AXIS is flipped.
        spatial_axes: Spatial axes (0 = left-right in RAS) that may be flipped.
        prob: Probability of flipping each axis.
    """

    def __init__(self, keys, label_key: str = "label", spatial_axes: tuple[int, ...] = (0, 1, 2),
                 prob: float = 0.5):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.label_key = label_key
        self.spatial_axes = tuple(spatial_axes)
        self.flip_prob = prob
        self._flipped: tuple[int, ...] = ()

    def randomize(self, data=None) -> None:
        self._flipped = tuple(a for a in self.spatial_axes if self.R.rand() < self.flip_prob)

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._flipped:
            return d
        dims = [a + 1 for a in self._flipped]  # channel-first tensors
        for key in self.key_iterator(d):
            d[key] = torch.flip(d[key], dims)
            if key == self.label_key and LATERAL_AXIS in self._flipped:
                d[key] = swap_lateral_labels(d[key])
        return d


def get_flips(flip: str = "none") -> list:
    """
    Random mirroring of the training patches (see RandFlipLateralSwapd).

    Args:
        flip: "none" (no mirroring, the pipeline up to cv_classes_w5_aug), "lr" (left-right
            axis only) or "all" (each of the 3 axes with probability 0.5, as in nnU-Net, which
            matches the 8 mirrors of the test-time augmentation).

    Returns:
        A list of MONAI dictionary transforms (empty for "none").
    """
    if flip == "none":
        return []
    if flip == "lr":
        return [RandFlipLateralSwapd(keys=["image", "label"], spatial_axes=(LATERAL_AXIS,))]
    if flip == "all":
        return [RandFlipLateralSwapd(keys=["image", "label"], spatial_axes=(0, 1, 2))]
    raise ValueError(f"Unknown flip '{flip}'. Choose 'none', 'lr' or 'all'.")


def get_class_ratios(
    rare_weight: float = 3.0,
    background_weight: float = 2.0,
    rare_classes: tuple[int, ...] = RARE_CLASSES,
) -> list[float]:
    """
    Sampling weight of each class for the patch centers: background_weight for
    the background (class 0), rare_weight for rare_classes, 1 for the other organs.
    """
    return [
        background_weight if c == 0 else rare_weight if c in rare_classes else 1.0
        for c in range(NUM_CLASSES)
    ]


def get_train_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    spacing: tuple[float, float, float] = SPACING,
    num_samples: int = 1,
    intensity_shift: bool = True,
    sampling: str = "posneg",
    rare_weight: float = 3.0,
    rare_classes: tuple[int, ...] = RARE_CLASSES,
    augmentation: str = "light",
    flip: str = "none",
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
        rare_weight: With sampling="classes", sampling weight of the rare_classes
            relative to the other organs (weight 1).
        rare_classes: Organ labels favoured by rare_weight (default RARE_CLASSES).
        augmentation: Random transforms applied to the patches (see get_augmentations):
            "light" (the original pipeline) or "strong" (nnU-Net-like ranges).
        flip: Random mirroring of the patches (see get_flips): "none", "lr" or "all".

    Returns:
        Compose: The composed MONAI transforms pipeline.
    """

    augmentations = get_augmentations(augmentation, spatial_size, intensity_shift) + get_flips(flip)

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
                ratios=get_class_ratios(rare_weight, rare_classes=rare_classes),
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
        # --- 1. Load, standardize, resample and window (shared with validation and inference) ---
        *get_preprocessing(("image", "label"), spacing),

        # --- 2. Patch Extraction ---
        *crop,

        # --- 3. Data Augmentation (on the patches) ---
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

    No random flip here: abdominal anatomy is not symmetric and some labels are
    side-specific (kidneys 2/3, adrenal glands 12/13), so mirroring is a separate
    option that swaps them (get_flips). No elastic deformation either: it is slow on the CPU
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
    spacing: tuple[float, float, float] = SPACING,
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
    transforms = get_preprocessing(("image", "label"), spacing)

    if not full_volume:
        # Deterministic single patch: crop to the organs' bounding box, then
        # center crop/pad to the patch size (no randomness, unlike RandCrop).
        transforms += [
            CropForegroundd(keys=["image", "label"], source_key="label"),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        ]

    return Compose(transforms)