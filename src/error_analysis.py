"""
Where does the HD95 of an organ come from? Per-volume diagnostics written by
scripts/evaluate.py (key "per_volume" of the JSON) and summarized by
scripts/analyze_errors.py.

    - Directed HD95: the symmetric HD95 is max(pred -> label, label -> pred). A large
      pred -> label distance means predicted voxels far from the annotation (false
      positive region, organ segmented too far); a large label -> pred distance means
      parts of the organ left out.
    - Craniocaudal over-extent: how far (mm) the prediction goes above / below the
      annotated slices (negative: stops before). Tests the hypothesis of a tubular
      organ (aorta, vena cava) annotated over a shorter range than the model segments.
"""

import numpy as np
from monai.metrics.utils import get_mask_edges, get_surface_distance

# Spatial axis of the craniocaudal direction in the RAS orientation (higher index = superior)
SUPERIOR_AXIS = 2


def directed_hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float]:
    """
    95th percentile of the surface distances in each direction.

    Args:
        pred: Binary prediction mask (X, Y, Z), not empty.
        gt: Binary ground truth mask of the same shape, not empty.
        spacing: Voxel size in mm.

    Returns:
        (pred -> label, label -> pred) in mm; their maximum is the symmetric HD95.
    """
    edges_pred, edges_gt = get_mask_edges(pred.astype(bool), gt.astype(bool), always_return_as_numpy=True)
    pred_to_gt = get_surface_distance(edges_pred, edges_gt, spacing=spacing)
    gt_to_pred = get_surface_distance(edges_gt, edges_pred, spacing=spacing)
    return float(np.percentile(pred_to_gt, 95)), float(np.percentile(gt_to_pred, 95))


def craniocaudal_overextent(pred: np.ndarray, gt: np.ndarray, slice_thickness: float) -> tuple[float, float]:
    """
    Distance the prediction extends beyond the annotated slices.

    Args:
        pred: Binary prediction mask (X, Y, Z), not empty.
        gt: Binary ground truth mask of the same shape, not empty.
        slice_thickness: Voxel size (mm) along SUPERIOR_AXIS.

    Returns:
        (superior, inferior) in mm: positive when the prediction goes above (below) the
        highest (lowest) annotated slice, negative when it stops before.
    """
    other_axes = tuple(a for a in range(3) if a != SUPERIOR_AXIS)
    pred_slices = np.flatnonzero(pred.any(axis=other_axes))
    gt_slices = np.flatnonzero(gt.any(axis=other_axes))
    superior = (pred_slices[-1] - gt_slices[-1]) * slice_thickness
    inferior = (gt_slices[0] - pred_slices[0]) * slice_thickness
    return float(superior), float(inferior)
