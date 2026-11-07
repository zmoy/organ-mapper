"""
Full-volume inference shared by scripts/evaluate.py and scripts/predict.py, so that
the metrics are always measured with the inference used to produce the masks.

Three settings (the defaults of predict_volume are the original inference; the
command-line defaults of add_inference_args are the best measured settings):
    - mode: weighting of the overlapping patches when they are stitched back
      ("constant": plain average, "gaussian": the centre of each patch counts more
      than its borders, where the network sees less context).
    - overlap: fraction of overlap between neighbouring patches.
    - tta: test-time augmentation by mirroring; the softmax probabilities are averaged
      over the 8 combinations of flips of the 3 spatial axes (as in nnU-Net).
      It only makes sense for a model trained with random flips (train_cv.py --flip):
      abdominal anatomy is not symmetric (liver on the right, spleen on the left) and a
      model that never saw a mirrored CT segments it very badly (Dice 0.01 to 0.19 on
      each mirrored input, 0.56 on the original, for one fold-5 volume).
      Such a model learns that a label means "the kidney on this side of the image": on
      an input mirrored along the left-right axis, its right / left channels are swapped
      back (lateral_pairs, scripts pass src.data.transforms.LATERAL_PAIRS).

predict_volume also takes several models (e.g. the 5 cross-validation folds, see
src.checkpoints.find_fold_checkpoints): their softmax probabilities are averaged.
"""

import argparse
from collections.abc import Iterable
from itertools import combinations

import torch
from monai.inferers import sliding_window_inference

# The 8 subsets of the spatial axes of a (B, C, X, Y, Z) tensor: (), (2,), (3,), ..., (2, 3, 4)
MIRROR_AXES: list[tuple[int, ...]] = [axes for n in range(4) for axes in combinations((2, 3, 4), n)]


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds the --sw-overlap, --sw-mode and --tta options (evaluate.py and predict.py).

    The defaults (gaussian, 75 % overlap) are the best settings measured in 5-fold CV
    (Dice 0.666 with constant / 50 %, 0.692 gaussian / 50 %, 0.700 gaussian / 75 %).
    """
    parser.add_argument("--sw-overlap", type=float, default=0.75,
                        help="Overlap between neighbouring sliding-window patches (0.5 is ~3x faster).")
    parser.add_argument("--sw-mode", default="gaussian", choices=["constant", "gaussian"],
                        help="Weighting of the overlapping patches (gaussian: patch centres count more).")
    parser.add_argument("--tta", action="store_true",
                        help="Average the probabilities over the 8 mirrored inputs (8x slower). Only for a "
                             "model trained with random flips: the current models fail on mirrored "
                             "inputs (CV Dice 0.700 -> 0.085).")


def inference_settings(args: argparse.Namespace) -> dict:
    """Keyword arguments of predict_volume taken from the parsed options."""
    return {"overlap": args.sw_overlap, "mode": args.sw_mode, "tta": args.tta}


def predict_volume(
    model: torch.nn.Module | Iterable[torch.nn.Module],
    image: torch.Tensor,
    roi_size: tuple[int, int, int],
    device: torch.device,
    overlap: float = 0.5,
    mode: str = "constant",
    tta: bool = False,
    lateral_pairs: tuple[tuple[int, int], ...] = (),
    lateral_dim: int = 2,
) -> torch.Tensor:
    """
    Sliding-window prediction of a full volume. The patches run on `device`, the
    stitched output is kept on the CPU (4 GB of VRAM).

    Args:
        model: Network in eval mode, already on `device`, or several networks (ensemble):
            the probabilities are averaged over the networks (and over the mirrors with tta).
        image: Input of shape (B, C, X, Y, Z).
        roi_size: Patch size of the sliding window.
        device: Device running the patches.
        overlap: Overlap between neighbouring patches (0.5 = 50 %).
        mode: "constant" or "gaussian" weighting of the overlapping patches.
        tta: Average the probabilities over the 8 mirrored versions of the input.
        lateral_pairs: With tta, (right, left) channel pairs swapped back on the inputs
            mirrored along lateral_dim (the side-specific organs). () swaps nothing.
        lateral_dim: Tensor dimension of the left-right axis (2 = spatial axis 0 in RAS).

    Returns:
        Softmax probabilities of shape (B, num_classes, X, Y, Z), float32, on the CPU.
    """
    if mode not in ("constant", "gaussian"):
        raise ValueError(f"Unknown sliding window mode '{mode}'. Choose 'constant' or 'gaussian'.")
    models = [model] if isinstance(model, torch.nn.Module) else list(model)
    if not models:
        raise ValueError("predict_volume needs at least one model.")
    image = image.to(device)
    mirrors = MIRROR_AXES if tta else [()]
    probs = None
    with torch.no_grad():
        for net, axes in ((net, axes) for net in models for axes in mirrors):
            flipped = torch.flip(image, axes) if axes else image
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    flipped, roi_size, 1, net, overlap=overlap, mode=mode,
                    sw_device=device, device=torch.device("cpu"),
                )
            p = torch.softmax(logits.float(), dim=1)
            del logits
            if axes:
                p = torch.flip(p, axes)  # back to the original orientation
                if lateral_dim in axes and lateral_pairs:
                    # The mirrored right kidney was predicted as the left one (and vice versa)
                    order = list(range(p.shape[1]))
                    for right, left in lateral_pairs:
                        order[right], order[left] = left, right
                    p = p[:, order]
            if probs is None:
                probs = p
            else:
                probs += p
                del p
    return probs / (len(models) * len(mirrors))
