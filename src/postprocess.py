"""
Post-processing of predicted segmentations, shared by scripts/evaluate.py and
scripts/predict.py so that the inference applies the same rules as the evaluation.
"""

import numpy as np
from scipy import ndimage


def keep_largest_components(label_map: np.ndarray, num_classes: int = 14, min_ratio: float = 1.0) -> np.ndarray:
    """
    For each organ (labels 1..num_classes - 1) keep its largest connected component,
    plus (if min_ratio < 1) every other component at least min_ratio times as large;
    the other islands are relabelled as background (0). This removes far-away false
    detections, which inflate the Hausdorff distance, while min_ratio < 1 keeps an
    organ predicted in a few large pieces.

    Args:
        label_map: Integer label map of shape (X, Y, Z). Modified in place.
        num_classes: Number of classes, background included.
        min_ratio: Size, relative to the largest component of the organ, from which
            a component is kept. 1.0 keeps only the largest one.

    Returns:
        The cleaned label map (same array as the input).
    """
    for organ in range(1, num_classes):
        components, n = ndimage.label(label_map == organ)  # face connectivity (6-neighbourhood)
        if n > 1:
            sizes = np.bincount(components.ravel())[1:]
            keep = np.zeros(n + 1, dtype=bool)  # keep[k]: component k is kept (k = 0: not the organ)
            keep[sizes.argmax() + 1] = True
            if min_ratio < 1.0:
                keep[1:] |= sizes >= min_ratio * sizes.max()
            label_map[(components > 0) & ~keep[components]] = 0
    return label_map
