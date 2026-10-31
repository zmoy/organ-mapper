"""
Post-processing of predicted segmentations, shared by scripts/evaluate.py and
scripts/predict.py so that the inference applies the same rules as the evaluation.
"""

import numpy as np
from scipy import ndimage


def keep_largest_components(label_map: np.ndarray, num_classes: int = 14) -> np.ndarray:
    """
    For each organ (labels 1..num_classes - 1) keep only its largest connected
    component; the other islands are relabelled as background (0). This removes
    far-away false detections, which inflate the Hausdorff distance.

    Args:
        label_map: Integer label map of shape (X, Y, Z). Modified in place.
        num_classes: Number of classes, background included.

    Returns:
        The cleaned label map (same array as the input).
    """
    for organ in range(1, num_classes):
        components, n = ndimage.label(label_map == organ)  # face connectivity (6-neighbourhood)
        if n > 1:
            sizes = np.bincount(components.ravel())[1:]
            label_map[(components > 0) & (components != sizes.argmax() + 1)] = 0
    return label_map
