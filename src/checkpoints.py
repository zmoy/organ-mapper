"""
Checkpoint lookup and loading for inference, shared by scripts/evaluate.py,
scripts/predict.py and scripts/export_models.py.

Two formats:
    - Lightning checkpoints (.ckpt, written by train.py / train_cv.py): only the network
      weights (the "model.*" keys of the state_dict) are loaded, since the checkpoints also
      hold the class weights of the loss and the optimizer state;
    - exported weights (.pt, written by scripts/export_models.py): the network state_dict
      alone, loaded with torch.load(weights_only=True), which runs no pickled code.
A run directory holds either one fold_k/ subfolder per fold (training) or flat files
named *fold_k.pt (export).
"""

import re
from pathlib import Path

import torch

from src.models.builder import get_model

EXPORT_PATTERN = re.compile(r"fold_(\d+)\.pt$")  # exported weights: <name>_fold_<k>.pt


def extract_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    """
    Network weights of a Lightning checkpoint.

    Args:
        checkpoint: .ckpt written by train.py / train_cv.py (a file of this project: it is
            unpickled with weights_only=False).

    Returns:
        The state_dict of the network ("model." prefix removed).
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}


def find_fold_checkpoint(checkpoints_dir: Path, fold: int, kind: str = "best") -> Path:
    """
    Checkpoint of one cross-validation fold (checkpoints_dir/fold_k/, written by train_cv.py).

    Args:
        checkpoints_dir: Run directory holding fold_1/, fold_2/, ...
        fold: Fold number, 1-based.
        kind: "best" (best-Dice checkpoint) or "last" (last.ckpt, end of the training).

    Returns:
        Path of the checkpoint (or of the exported weights, see the module docstring).
    """
    if not (checkpoints_dir / f"fold_{fold}").is_dir():
        exported = [p for p in checkpoints_dir.glob("*.pt") if _exported_fold(p) == fold]
        if len(exported) == 1:
            return exported[0]
        if len(exported) > 1:
            raise FileNotFoundError(f"Several exported weights for fold {fold} in {checkpoints_dir}")
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
    if not folds:  # exported weights: flat *fold_k.pt files
        folds = sorted({f for p in checkpoints_dir.glob("*.pt") if (f := _exported_fold(p)) is not None})
    if not folds:
        raise FileNotFoundError(f"No fold_k/ folder nor *fold_k.pt file in {checkpoints_dir}")
    return [find_fold_checkpoint(checkpoints_dir, fold, kind) for fold in folds]


def _exported_fold(path: Path) -> int | None:
    """Fold number of an exported weights file (<name>_fold_<k>.pt), None for other files."""
    match = EXPORT_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def load_model(
    checkpoint: Path,
    model_name: str,
    device: torch.device,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
) -> torch.nn.Module:
    """
    Network of a Lightning checkpoint or of exported weights, in eval mode on `device`.

    Args:
        checkpoint: .ckpt written by train.py / train_cv.py, or .pt written by
            scripts/export_models.py.
        model_name: Architecture, see src.models.builder.get_model.
        device: Device of the returned network.
        spatial_size: Patch size (needed by the transformer models).

    Returns:
        The network with the weights of the checkpoint.
    """
    model = get_model(model_name=model_name, spatial_size=spatial_size)
    if Path(checkpoint).suffix == ".pt":
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        weights = extract_weights(checkpoint)
    model.load_state_dict(weights)  # strict: a wrong architecture fails here
    return model.to(device).eval()
