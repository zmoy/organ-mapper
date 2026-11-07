"""
Post-processing of predicted segmentations, shared by scripts/evaluate.py and
scripts/predict.py so that the inference applies the same rules as the evaluation.
"""

import numpy as np
from scipy import ndimage

# Voxel size (mm) of the resampled volumes the post-processing runs on (see src/data/transforms.py)
VOXEL_SPACING = (1.5, 1.5, 2.0)


def _component_distances_mm(
    components: np.ndarray, largest: int, max_distance_mm: float, spacing: tuple[float, float, float]
) -> dict[int, float]:
    """
    Distance (mm) from each component to the largest one, for the components closer than
    max_distance_mm (the others are left out of the result).

    The distance transform only runs in the bounding box of the largest component, grown by
    max_distance_mm: a voxel outside this box is farther than max_distance_mm anyway, so the
    distances that matter are unchanged and the computation stays small.

    Args:
        components: Component map of one organ (0 = not the organ, k = component k).
        largest: Index of the largest component.
        max_distance_mm: Distance beyond which a component is not reported.
        spacing: Voxel size (mm) along each axis.

    Returns:
        {component index: distance in mm} for the components within max_distance_mm
        (the largest one excluded).
    """
    coords = np.nonzero(components == largest)
    margin = [int(np.ceil(max_distance_mm / s)) + 1 for s in spacing]
    box = tuple(
        slice(max(int(c.min()) - m, 0), min(int(c.max()) + m + 1, size))
        for c, m, size in zip(coords, margin, components.shape)
    )
    crop = components[box]
    distance = ndimage.distance_transform_edt(crop != largest, sampling=spacing)
    ids = [int(k) for k in np.unique(crop) if k not in (0, largest)]
    if not ids:
        return {}
    minima = ndimage.minimum(distance, labels=crop, index=ids)
    return {k: float(d) for k, d in zip(ids, np.atleast_1d(minima)) if d <= max_distance_mm}


def keep_largest_components(
    label_map: np.ndarray,
    num_classes: int = 14,
    min_ratio: float = 1.0,
    max_distance_mm: float = 0.0,
    spacing: tuple[float, float, float] = VOXEL_SPACING,
) -> np.ndarray:
    """
    For each organ (labels 1..num_classes - 1) keep its largest connected component, plus:
      - if min_ratio < 1: every other component at least min_ratio times as large;
      - if max_distance_mm > 0: every other component within max_distance_mm of the
        largest one (an organ cut into pieces by a thin under-segmented area stays whole,
        while a far-away false detection is still removed).
    The other islands are relabelled as background (0). Far-away false detections inflate
    the Hausdorff distance.

    Args:
        label_map: Integer label map of shape (X, Y, Z). Modified in place.
        num_classes: Number of classes, background included.
        min_ratio: Size, relative to the largest component of the organ, from which
            a component is kept. 1.0 keeps only the largest one.
        max_distance_mm: Distance (mm, between the closest voxels) to the largest
            component under which a component is kept. 0 disables the rule.
        spacing: Voxel size (mm) of label_map, used by max_distance_mm.

    Returns:
        The cleaned label map (same array as the input).
    """
    for organ in range(1, num_classes):
        components, n = ndimage.label(label_map == organ)  # face connectivity (6-neighbourhood)
        if n > 1:
            sizes = np.bincount(components.ravel())[1:]
            largest = int(sizes.argmax()) + 1
            keep = np.zeros(n + 1, dtype=bool)  # keep[k]: component k is kept (k = 0: not the organ)
            keep[largest] = True
            if min_ratio < 1.0:
                keep[1:] |= sizes >= min_ratio * sizes.max()
            if max_distance_mm > 0:
                for k in _component_distances_mm(components, largest, max_distance_mm, spacing):
                    keep[k] = True
            label_map[(components > 0) & ~keep[components]] = 0
    return label_map
