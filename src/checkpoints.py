"""
Checkpoint lookup and loading for inference, shared by scripts/evaluate.py and
scripts/predict.py.

Only the network weights (the "model.*" keys of the Lightning state_dict) are loaded:
the checkpoints also hold the class weights of the loss.
"""

from pathlib import Path

import torch

from src.models.builder import get_model


def find_fold_checkpoint(checkpoints_dir: Path, fold: int, kind: str = "best") -> Path:
    """
    Checkpoint of one cross-validation fold (checkpoints_dir/fold_k/, written by train_cv.py).

    Args:
        checkpoints_dir: Run directory holding fold_1/, fold_2/, ...
        fold: Fold number, 1-based.
        kind: "best" (best-Dice checkpoint) or "last" (last.ckpt, end of the training).

    Returns:
        Path of the checkpoint.
    """
    if kind == "last":
        last = checkpoints_dir / f"fold_{fold}" / "last.ckpt"
        if not last.exists():
            raise FileNotFoundError(f"No last.ckpt in {last.parent}")
        return last
    # rglob: older runs wrote "name-val/dice=x.ckpt", i.e. inside a subfolder
    found = list((checkpoints_dir / f"fold_{fold}").rglob("*.ckpt"))
    # last.ckpt (resume state) is only a fallback: the best-Dice checkpoint is the reference
    candidates = sorted((p for p in found if p.name != "last.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = found
    if not candidates:
        raise FileNotFoundError(f"No .ckpt found in {checkpoints_dir / f'fold_{fold}'}")
    return candidates[-1]


def find_fold_checkpoints(checkpoints_dir: Path, kind: str = "best") -> list[Path]:
    """
    One checkpoint per fold of a cross-validation run (the fold_k/ subfolders, in order).

    Args:
        checkpoints_dir: Run directory holding fold_1/, fold_2/, ...
        kind: "best" or "last", see find_fold_checkpoint.

    Returns:
        The checkpoints of folds 1, 2, ... (raises FileNotFoundError if there is no fold).
    """
    folds = sorted(
        int(p.name.split("_")[1]) for p in checkpoints_dir.glob("fold_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    )
    if not folds:
        raise FileNotFoundError(f"No fold_k/ folder in {checkpoints_dir}")
    return [find_fold_checkpoint(checkpoints_dir, fold, kind) for fold in folds]


def load_model(
    checkpoint: Path,
    model_name: str,
    device: torch.device,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
) -> torch.nn.Module:
    """
    Network of a Lightning checkpoint, in eval mode on `device`.

    Args:
        checkpoint: .ckpt written by train.py / train_cv.py.
        model_name: Architecture, see src.models.builder.get_model.
        device: Device of the returned network.
        spatial_size: Patch size (needed by the transformer models).

    Returns:
        The network with the "model.*" weights of the checkpoint.
    """
    model = get_model(model_name=model_name, spatial_size=spatial_size)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    return model.to(device).eval()
