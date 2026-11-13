"""
Export of a cross-validation run for publication (scripts/export_models.py).

For each fold, the network weights alone (no optimizer state, no loss buffers) are saved as
<prefix>_fold_<k>.pt, loadable with torch.load(weights_only=True) and src.checkpoints.load_model.
A manifest.json describes what is needed to use them correctly: architecture, classes,
pre-processing, inference settings, training configuration, cross-validation scores and the
SHA-256 of each file.
"""

import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import torch

from src.checkpoints import extract_weights, find_fold_checkpoint
from src.data.transforms import HU_WINDOW, NUM_CLASSES, ORGAN_NAMES, ORIENTATION, SPACING
from src.models.builder import get_model
from src.training.config import TrainConfig

# TrainConfig fields that describe the run environment, not the model
_RUN_FIELDS = ("logger", "run_name", "fast_dev_run", "resume", "min_free_gb")


def sha256(path: Path) -> str:
    """SHA-256 of a file, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture(model_name: str) -> dict:
    """Description of the network built by get_model(model_name), enough to rebuild it."""
    model = get_model(model_name=model_name)
    info = {
        "model_name": model_name,
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "in_channels": 1,
        "out_channels": NUM_CLASSES,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    for attr in ("channels", "strides", "num_res_units"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            info[attr] = list(value) if isinstance(value, (tuple, list)) else value
    return info


def build_manifest(
    run_name: str,
    evaluation: dict,
    folds: list[dict],
    training: TrainConfig,
) -> dict:
    """
    Args:
        run_name: Name of the training run (checkpoints/<run_name>).
        evaluation: Content of the evaluation JSON of the run (scripts/evaluate.py --cv).
        folds: One entry per exported fold (fold, file, sha256, source_checkpoint, epoch).
        training: Training configuration of the run.

    Returns:
        The manifest, JSON-serializable.
    """
    per_fold = {f["fold"]: f for f in evaluation.get("per_fold", [])}
    validation = {}
    for v in evaluation.get("per_volume", []):
        validation.setdefault(v.get("fold"), []).append(v["name"])
    for entry in folds:
        scores = per_fold.get(entry["fold"], {})
        entry["validation_volumes"] = sorted(validation.get(entry["fold"], []))
        entry["validation"] = {k: scores.get(f"mean_{k}") for k in ("dice", "hd95", "nsd")}

    def organ_scores(key: str) -> dict:
        means = evaluation.get(key, {}).get("mean", [])
        return dict(zip(ORGAN_NAMES, means))

    return {
        "name": f"organ-mapper-{evaluation.get('model', training.model_name)}",
        "run": run_name,
        "exported": date.today().isoformat(),
        "task": "3D segmentation of 13 abdominal organs in contrast-enhanced CT",
        "architecture": architecture(training.model_name),
        "classes": {"0": "background", **{str(i): n for i, n in enumerate(ORGAN_NAMES, start=1)}},
        "preprocessing": {
            "orientation": ORIENTATION,
            "spacing_mm": list(SPACING),
            "hu_window": list(HU_WINDOW),
            "intensity_range": [0.0, 1.0],
            "function": "src.data.transforms.get_inference_transforms",
        },
        "inference": {
            "sliding_window_roi": list(training.spatial_size),
            "mode": evaluation.get("inference", {}).get("mode"),
            "overlap": evaluation.get("inference", {}).get("overlap"),
            "tta": evaluation.get("inference", {}).get("tta"),
            "postprocess": "largest connected component per organ" if evaluation.get("postprocess") else None,
        },
        "training": {k: v for k, v in asdict(training).items() if k not in _RUN_FIELDS},
        "evaluation": {
            "protocol": "5-fold cross-validation (KFold, seed 42) on 30 annotated volumes, whole volumes, "
                        "each fold scored by its own model on its 6 validation volumes",
            "checkpoint_selection": f"{evaluation.get('checkpoint_kind', 'best')} validation Dice",
            "nsd_tolerance_mm": evaluation.get("nsd_tolerance_mm"),
            "mean_dice": evaluation.get("mean_dice"),
            "mean_hd95_mm": evaluation.get("mean_hd95"),
            "mean_nsd": evaluation.get("mean_nsd"),
            "dice_per_organ": organ_scores("dice_per_organ"),
            "hd95_per_organ_mm": organ_scores("hd95_per_organ"),
            "nsd_per_organ": organ_scores("nsd_per_organ"),
        },
        "folds": folds,
    }


def export_run(
    checkpoints_dir: Path,
    evaluation_path: Path,
    output_dir: Path,
    prefix: str = "organ_mapper_unet",
    kind: str = "best",
    training: TrainConfig | None = None,
) -> dict:
    """
    Writes <prefix>_fold_<k>.pt for each fold of the run and manifest.json.

    Args:
        checkpoints_dir: Run directory (fold_1/, fold_2/, ...).
        evaluation_path: Evaluation JSON of the same checkpoints.
        output_dir: Destination (created).
        prefix: File name prefix.
        kind: "best" or "last" checkpoint of each fold; must match the evaluation.
        training: Training configuration of the run (default: TrainConfig(), the reference).

    Returns:
        The manifest.
    """
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    if evaluation.get("checkpoint_kind", "best") != kind:
        raise ValueError(f"The evaluation scored the '{evaluation.get('checkpoint_kind')}' checkpoints, "
                         f"not '{kind}'.")
    evaluated = [Path(p).name for p in evaluation.get("checkpoints", [])]
    training = training or TrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    for fold in evaluation["folds"]:
        source = find_fold_checkpoint(checkpoints_dir, fold, kind)
        if evaluated and source.name not in evaluated:
            raise ValueError(f"{source.name} is not one of the evaluated checkpoints: {evaluated}")
        weights = extract_weights(source)
        get_model(model_name=training.model_name).load_state_dict(weights)  # fails on a wrong architecture
        target = output_dir / f"{prefix}_fold_{fold}.pt"
        torch.save(weights, target)
        epoch = torch.load(source, map_location="cpu", weights_only=False).get("epoch")
        folds.append({"fold": fold, "file": target.name, "sha256": sha256(target),
                      "source_checkpoint": source.name, "epoch": epoch})

    text = json.dumps(build_manifest(checkpoints_dir.name, evaluation, folds, training), indent=2)
    (output_dir / "manifest.json").write_text(text, encoding="utf-8")
    return json.loads(text)  # as written (tuples become lists)
