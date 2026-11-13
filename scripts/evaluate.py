"""
Offline clinical evaluation script.
Computes per-organ Dice Score, Hausdorff Distance (HD95, mm) and Normalized Surface
Distance (NSD) on full validation volumes (sliding window inference, outputs kept on the
CPU to stay within 4 GB of VRAM).

An organ present in the label but never predicted gets the worst possible HD95 (the
diagonal of the volume, in mm) and an NSD of 0, instead of being left out of the means.

Two modes:
    - Hold-out (default): one checkpoint on the validation split used by train.py.
        python scripts/evaluate.py --checkpoint checkpoints/xxx.ckpt
    - Cross-validation (--cv): one checkpoint per fold (checkpoints/fold_k/*.ckpt, written
      by train_cv.py), each evaluated on the volumes it never saw during training.
      Reports mean +/- std over the folds.
        python scripts/evaluate.py --cv
      --checkpoint-kind last evaluates last.ckpt (no checkpoint selection on the
      validation volumes) instead of the best-Dice checkpoint.

Inference options (shared with predict.py, see src/inference.py): --sw-overlap,
--sw-mode {constant,gaussian} and --tta (mirroring test-time augmentation).

The JSON also holds per-volume results ("per_volume": Dice, HD95, NSD and the HD95
diagnostics of src/error_analysis.py for each volume and organ), used by
scripts/compare.py (paired tests) and scripts/analyze_errors.py.

--predictions-cache DIR saves the raw predictions (before post-processing) of each volume,
and reuses them on the next run with the same checkpoint and inference settings: comparing
post-processing settings then takes minutes instead of an hour of sliding window.
    python scripts/evaluate.py --cv --checkpoints-dir checkpoints/<run> --postprocess \
        --predictions-cache predictions/cache/<run>_best --max-component-distance 10
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.checkpoints import find_fold_checkpoint, load_model
from src.data.dataset import get_data_dicts, get_dataset, get_kfold_splits, split_train_val
from src.data.transforms import LATERAL_PAIRS, SPACING, get_val_transforms
from src.error_analysis import craniocaudal_overextent, directed_hd95
from src.inference import add_inference_args, inference_settings, predict_volume
from src.postprocess import keep_largest_components as keep_largest_components_np

SPATIAL_SIZE = (96, 96, 96)
# Voxel size (mm) of the resampled volumes (shared pre-processing), so that HD95 is in mm
VOXEL_SPACING = SPACING
NUM_CLASSES = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-volume Dice / HD95 evaluation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "unet_baseline_best" / "unet_baseline.ckpt",
        help="Checkpoint for the hold-out mode.",
    )
    parser.add_argument("--model", default="unet", choices=["unet", "swin_unetr"])
    parser.add_argument("--cv", action="store_true", help="Evaluate one checkpoint per fold.")
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=PROJECT_ROOT / "checkpoints",
        help="Directory holding fold_1/, fold_2/, ... (--cv only).",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds (--cv only).")
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Subset of folds to evaluate, 1-based (--cv only). Default: all.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-kind", default="best", choices=["best", "last"],
        help="--cv only: best-Dice checkpoint of each fold (chosen on the validation volumes, "
             "slightly optimistic) or last.ckpt (no selection).",
    )
    parser.add_argument(
        "--nsd-tolerance", type=float, default=2.0,
        help="Tolerance (mm) of the Normalized Surface Distance.",
    )
    parser.add_argument(
        "--postprocess", action="store_true",
        help="Keep only the largest connected component of each organ before computing the metrics.",
    )
    parser.add_argument(
        "--min-component-ratio", type=float, default=1.0,
        help="With --postprocess: also keep the components at least this fraction of the organ's "
             "largest one (1.0 = largest only).",
    )
    parser.add_argument(
        "--max-component-distance", type=float, default=0.0,
        help="With --postprocess: also keep the components within this distance (mm) of the "
             "organ's largest one (0 = disabled).",
    )
    parser.add_argument(
        "--predictions-cache", type=Path, default=None,
        help="Folder where the raw predictions are saved, and reused by a later run with the same "
             "checkpoint and inference settings (fold_k/ subfolders in --cv mode).",
    )
    add_inference_args(parser)
    return parser.parse_args()


def nan_stat(values: np.ndarray, func, axis: int = 0) -> np.ndarray:
    """nanmean / nanstd that stay quiet when a slice is all-NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(values, axis=axis)


def to_list(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in values]


def keep_largest_components(
    onehot: torch.Tensor, min_ratio: float = 1.0, max_distance_mm: float = 0.0
) -> torch.Tensor:
    """
    Post-processing: for each organ keep its largest connected component (plus the
    components at least min_ratio times as large, or within max_distance_mm of it) and
    relabel the other islands as background (removes far-away false detections).

    Args:
        onehot: uint8 one-hot prediction of shape (1, NUM_CLASSES, X, Y, Z).
        min_ratio, max_distance_mm: See src.postprocess.keep_largest_components.

    Returns:
        A one-hot uint8 tensor of the same shape.
    """
    label_map = keep_largest_components_np(
        onehot[0].argmax(dim=0).numpy().astype(np.uint8), NUM_CLASSES, min_ratio,
        max_distance_mm, VOXEL_SPACING,
    )
    return to_onehot(label_map)


def to_onehot(label_map: np.ndarray) -> torch.Tensor:
    """(X, Y, Z) label map -> uint8 one-hot tensor of shape (1, NUM_CLASSES, X, Y, Z)."""
    onehot = torch.zeros((1, NUM_CLASSES, *label_map.shape), dtype=torch.uint8)
    for c in range(NUM_CLASSES):
        onehot[0, c] = torch.from_numpy(label_map == c)
    return onehot


class PredictionCache:
    """
    Raw predictions (argmax label maps, before post-processing) saved to disk, one
    compressed .npz per volume, so that post-processing settings can be compared without
    running the sliding window again.

    A meta.json file records the checkpoint and the inference settings: reusing the folder
    with other settings raises an error instead of silently mixing predictions.
    """

    def __init__(self, folder: Path, checkpoint: Path, inference: dict | None):
        self.folder = Path(folder)
        self.meta = {"checkpoint": str(Path(checkpoint).resolve()), "inference": inference or {}}
        meta_file = self.folder / "meta.json"
        if meta_file.exists():
            saved = json.loads(meta_file.read_text())
            if saved != self.meta:
                raise ValueError(
                    f"{self.folder} holds predictions of another checkpoint or other inference "
                    f"settings ({saved}); use another --predictions-cache folder."
                )
        else:
            self.folder.mkdir(parents=True, exist_ok=True)
            meta_file.write_text(json.dumps(self.meta, indent=2))

    def _path(self, name: str) -> Path:
        return self.folder / f"{name}.npz"

    def load(self, name: str) -> np.ndarray | None:
        path = self._path(name)
        return np.load(path)["label_map"] if path.exists() else None

    def save(self, name: str, label_map: np.ndarray) -> None:
        np.savez_compressed(self._path(name), label_map=label_map.astype(np.uint8))


DIAGNOSTICS = ("hd95_pred_to_label", "hd95_label_to_pred", "overextent_superior_mm", "overextent_inferior_mm")


def per_volume_record(
    name: str, dice: np.ndarray, hd95: np.ndarray, nsd: np.ndarray, diagnostics: dict, missed: np.ndarray
) -> dict:
    """
    JSON-ready results of one volume: per-organ lists (13 values, None where undefined).

    Args:
        name: Volume name (see volume_name).
        dice, hd95, nsd: Per-organ metrics.
        diagnostics: Per-organ arrays keyed by DIAGNOSTICS.
        missed: Boolean per organ, present in the label but never predicted.
    """
    record = {"name": name, "dice": to_list(dice), "hd95": to_list(hd95), "nsd": to_list(nsd)}
    record.update({key: to_list(diagnostics[key]) for key in DIAGNOSTICS})
    record["missed"] = [int(m) for m in missed]
    return record


def volume_name(file_dict: dict) -> str:
    """'img0007' for an image file data/raw/imagesTr/img0007.nii.gz."""
    return Path(file_dict["image"]).name.split(".")[0]


def evaluate_checkpoint(
    checkpoint: Path, model_name: str, val_files: list[dict], device: torch.device,
    postprocess: bool = False, nsd_tolerance: float = 2.0,
    inference: dict | None = None, min_component_ratio: float = 1.0,
    max_component_distance: float = 0.0, predictions_cache: Path | None = None,
) -> dict:
    """
    Evaluates one checkpoint on full volumes.

    Args:
        postprocess: Keep only the largest connected component of each organ (see
            keep_largest_components) before computing the metrics.
        min_component_ratio: With postprocess, also keep the components at least this
            fraction of the organ's largest one.
        max_component_distance: With postprocess, also keep the components within this
            distance (mm) of the organ's largest one (0 = disabled).
        predictions_cache: Folder of a PredictionCache (raw predictions reused when present,
            saved otherwise); None runs the inference on every volume.
        nsd_tolerance: Tolerance (mm) of the Normalized Surface Distance.
        inference: Keyword arguments of src.inference.predict_volume (overlap, mode, tta);
            None keeps its defaults.

    Returns:
        dict with per-patient arrays (patients x organs): "dice", "hd95", "nsd" (NaN where the
        organ is absent from the label), "missed" (organs present in the label but never
        predicted, per organ), and "volumes" (one per_volume_record per patient).
    """
    cache = PredictionCache(predictions_cache, checkpoint, inference) if predictions_cache else None
    names = [volume_name(f) for f in val_files]
    model = None
    if cache is None or any(cache.load(n) is None for n in names):
        print(f"Loading checkpoint: {checkpoint}")
        model = load_model(checkpoint, model_name, device, SPATIAL_SIZE)
    else:
        print(f"All {len(names)} predictions found in {predictions_cache}: no inference needed.")

    val_ds = get_dataset(
        val_files, get_val_transforms(spatial_size=SPATIAL_SIZE, full_volume=True), cache_name="val_full"
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # reduction="none": one value per patient and per organ
    # Computed one organ (one channel) at a time to keep the RAM usage low
    dice_metric = DiceMetric(include_background=True, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
    nsd_metric = SurfaceDiceMetric(class_thresholds=[nsd_tolerance], include_background=True, reduction="none")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    dice_rows, hd95_rows, nsd_rows, volumes = [], [], [], []
    missed = np.zeros(NUM_CLASSES - 1, dtype=int)

    print(f"Evaluating on {len(val_loader)} full patient volumes...")
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            labels = batch_data["label"]
            cached = cache.load(names[idx]) if cache else None
            if cached is not None:
                print(f"  Patient {idx + 1}/{len(val_loader)} - cached prediction")
                val_outputs = to_onehot(cached)
            else:
                print(f"  Patient {idx + 1}/{len(val_loader)} - Sliding Window Inference...")
                images = batch_data["image"].to(device)
                # Probabilities stitched on the CPU (see src/inference.py)
                outputs = predict_volume(model, images, SPATIAL_SIZE, device, **(inference or {}),
                                         lateral_pairs=LATERAL_PAIRS)
                # One-hot masks as uint8 (4x smaller than float32)
                val_outputs = torch.stack([post_pred(o) for o in outputs]).to(torch.uint8)
                del outputs, images
                if cache:
                    cache.save(names[idx], val_outputs[0].argmax(dim=0).numpy())
            if postprocess:
                val_outputs = keep_largest_components(val_outputs, min_component_ratio, max_component_distance)
            val_labels = torch.stack([post_label(lbl) for lbl in labels]).to(torch.uint8)

            label_present = val_labels[0, 1:].flatten(1).sum(dim=1).numpy() > 0
            pred_present = val_outputs[0, 1:].flatten(1).sum(dim=1).numpy() > 0

            # Worst possible distance: diagonal of the volume (mm)
            max_distance = float(np.linalg.norm(np.array(val_labels.shape[2:]) * np.array(VOXEL_SPACING)))

            dice = np.full(NUM_CLASSES - 1, np.nan)
            hd95 = np.full(NUM_CLASSES - 1, np.nan)
            nsd = np.full(NUM_CLASSES - 1, np.nan)
            # HD95 diagnostics (src/error_analysis.py), defined when the organ is predicted
            diag = {key: np.full(NUM_CLASSES - 1, np.nan) for key in DIAGNOSTICS}
            for organ in np.flatnonzero(label_present):  # organ absent from the ground truth: undefined
                pred = val_outputs[:, organ + 1:organ + 2].float()
                gt = val_labels[:, organ + 1:organ + 2].float()
                dice[organ] = dice_metric(y_pred=pred, y=gt)[0, 0].item()
                if pred_present[organ]:
                    hd95[organ] = hd95_metric(y_pred=pred, y=gt, spacing=VOXEL_SPACING)[0, 0].item()
                    nsd[organ] = nsd_metric(y_pred=pred, y=gt, spacing=VOXEL_SPACING)[0, 0].item()
                    pred_np, gt_np = pred[0, 0].numpy() > 0, gt[0, 0].numpy() > 0
                    to_gt, to_pred = directed_hd95(pred_np, gt_np, VOXEL_SPACING)
                    superior, inferior = craniocaudal_overextent(pred_np, gt_np, VOXEL_SPACING[2])
                    for key, value in zip(DIAGNOSTICS, (to_gt, to_pred, superior, inferior)):
                        diag[key][organ] = value
                else:
                    # Organ present but never predicted: penalised, not left out of the means
                    hd95[organ] = max_distance
                    nsd[organ] = 0.0
            miss = label_present & ~pred_present
            missed += miss

            dice_rows.append(dice)
            hd95_rows.append(hd95)
            nsd_rows.append(nsd)
            volumes.append(per_volume_record(names[idx], dice, hd95, nsd, diag, miss))

            del labels, val_outputs, val_labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "dice": np.stack(dice_rows), "hd95": np.stack(hd95_rows), "nsd": np.stack(nsd_rows), "missed": missed,
        "volumes": volumes,
    }


def summarize_run(result: dict) -> dict:
    """Collapses per-patient arrays into per-organ means and overall means."""
    dice_per_organ = nan_stat(result["dice"], np.nanmean)
    hd95_per_organ = nan_stat(result["hd95"], np.nanmean)
    nsd_per_organ = nan_stat(result["nsd"], np.nanmean)
    return {
        "num_patients": int(result["dice"].shape[0]),
        "dice_per_organ": dice_per_organ,
        "hd95_per_organ": hd95_per_organ,
        "nsd_per_organ": nsd_per_organ,
        "mean_dice": float(np.nanmean(dice_per_organ)),
        "mean_hd95": float(np.nanmean(hd95_per_organ)),
        "mean_nsd": float(np.nanmean(nsd_per_organ)),
        "missed": result["missed"],
        "volumes": result.get("volumes", []),
    }


def fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None or np.isnan(value) else format(value, spec)


def print_holdout(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("FINAL CLINICAL RESULTS (full volumes)")
    print("=" * 60)
    print(f"{'Organ':<10}{'Dice':>10}{'HD95 (mm)':>14}{'NSD':>10}{'Missed':>10}")
    for k in range(NUM_CLASSES - 1):
        print(f"{k + 1:<10}{fmt(summary['dice_per_organ'][k], '.3f'):>10}"
              f"{fmt(summary['hd95_per_organ'][k], '.1f'):>14}"
              f"{fmt(summary['nsd_per_organ'][k], '.3f'):>10}{summary['missed'][k]:>10}")
    print("-" * 60)
    print(f"Mean Dice: {fmt(summary['mean_dice'], '.4f')}  |  "
          f"Mean HD95: {fmt(summary['mean_hd95'], '.2f')} mm  |  Mean NSD: {fmt(summary['mean_nsd'], '.4f')}")
    print("(missed organs: HD95 = volume diagonal, NSD = 0)")
    print("=" * 60)


def print_cv(folds: list[int], summaries: list[dict], agg: dict) -> None:
    print("\n" + "=" * 64)
    print(f"CROSS-VALIDATION RESULTS ({len(folds)} folds, full volumes)")
    print("=" * 64)
    print(f"{'Fold':<8}{'Dice':>10}{'HD95 (mm)':>14}{'NSD':>10}")
    for fold, s in zip(folds, summaries):
        print(f"{fold:<8}{fmt(s['mean_dice'], '.3f'):>10}{fmt(s['mean_hd95'], '.1f'):>14}"
              f"{fmt(s['mean_nsd'], '.3f'):>10}")
    print("-" * 82)
    print(f"{'Organ':<8}{'Dice (mean +/- std)':>24}{'HD95 mm (mean +/- std)':>26}{'NSD (mean +/- std)':>24}")
    for k in range(NUM_CLASSES - 1):
        dice = f"{fmt(agg['dice_per_organ_mean'][k], '.3f')} +/- {fmt(agg['dice_per_organ_std'][k], '.3f')}"
        hd = f"{fmt(agg['hd95_per_organ_mean'][k], '.1f')} +/- {fmt(agg['hd95_per_organ_std'][k], '.1f')}"
        nsd = f"{fmt(agg['nsd_per_organ_mean'][k], '.3f')} +/- {fmt(agg['nsd_per_organ_std'][k], '.3f')}"
        print(f"{k + 1:<8}{dice:>24}{hd:>26}{nsd:>24}")
    print("-" * 82)
    print(f"Mean Dice: {agg['mean_dice_mean']:.4f} +/- {agg['mean_dice_std']:.4f}")
    print(f"Mean HD95: {fmt(agg['mean_hd95_mean'], '.2f')} +/- {fmt(agg['mean_hd95_std'], '.2f')} mm")
    print(f"Mean NSD:  {fmt(agg['mean_nsd_mean'], '.4f')} +/- {fmt(agg['mean_nsd_std'], '.4f')}")
    print("(std = sample std over folds; missed organs: HD95 = volume diagonal, NSD = 0)")
    print("=" * 82)


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / "data" / "raw"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running clinical evaluation on device: {device}\n")

    data_dicts = get_data_dicts(data_dir)

    # --- Hold-out mode ---
    if not args.cv:
        _, val_files = split_train_val(data_dicts)
        result = evaluate_checkpoint(
            args.checkpoint, args.model, val_files, device, args.postprocess, args.nsd_tolerance,
            inference_settings(args), args.min_component_ratio, args.max_component_distance,
            args.predictions_cache,
        )
        summary = summarize_run(result)
        print_holdout(summary)

        output = args.output or PROJECT_ROOT / "results" / "evaluation.json"
        payload = {
            "checkpoint": str(args.checkpoint),
            "model": args.model,
            "postprocess": args.postprocess,
            "min_component_ratio": args.min_component_ratio,
            "max_component_distance_mm": args.max_component_distance,
            "nsd_tolerance_mm": args.nsd_tolerance,
            "inference": inference_settings(args),
            "num_patients": summary["num_patients"],
            "mean_dice": summary["mean_dice"],
            "mean_hd95": None if np.isnan(summary["mean_hd95"]) else summary["mean_hd95"],
            "mean_nsd": None if np.isnan(summary["mean_nsd"]) else summary["mean_nsd"],
            "dice_per_organ": to_list(summary["dice_per_organ"]),
            "hd95_per_organ": to_list(summary["hd95_per_organ"]),
            "nsd_per_organ": to_list(summary["nsd_per_organ"]),
            "missed_per_organ": summary["missed"].tolist(),
            "per_volume": summary["volumes"],
        }
    # --- Cross-validation mode ---
    else:
        splits = get_kfold_splits(data_dicts, n_splits=args.n_splits)
        folds = args.folds or list(range(1, args.n_splits + 1))
        summaries, checkpoints = [], []
        for fold in folds:
            print(f"\n{'=' * 50}\nFOLD {fold}/{args.n_splits}\n{'=' * 50}")
            ckpt = find_fold_checkpoint(args.checkpoints_dir, fold, args.checkpoint_kind)
            checkpoints.append(str(ckpt))
            _, val_files = splits[fold - 1]
            fold_cache = args.predictions_cache / f"fold_{fold}" if args.predictions_cache else None
            summaries.append(summarize_run(
                evaluate_checkpoint(ckpt, args.model, val_files, device, args.postprocess, args.nsd_tolerance,
                                    inference_settings(args), args.min_component_ratio,
                                    args.max_component_distance, fold_cache)
            ))

        stack = lambda key: np.stack([s[key] for s in summaries])  # noqa: E731
        ddof = 1 if len(folds) > 1 else 0
        std = lambda arr: nan_stat(arr, lambda a, axis: np.nanstd(a, axis=axis, ddof=ddof))  # noqa: E731
        agg = {
            "dice_per_organ_mean": nan_stat(stack("dice_per_organ"), np.nanmean),
            "dice_per_organ_std": std(stack("dice_per_organ")),
            "hd95_per_organ_mean": nan_stat(stack("hd95_per_organ"), np.nanmean),
            "hd95_per_organ_std": std(stack("hd95_per_organ")),
            "nsd_per_organ_mean": nan_stat(stack("nsd_per_organ"), np.nanmean),
            "nsd_per_organ_std": std(stack("nsd_per_organ")),
            "mean_dice_mean": float(np.mean(stack("mean_dice"))),
            "mean_dice_std": float(np.std(stack("mean_dice"), ddof=ddof)),
            "mean_hd95_mean": float(nan_stat(stack("mean_hd95"), np.nanmean)),
            "mean_hd95_std": float(std(stack("mean_hd95"))),
            "mean_nsd_mean": float(nan_stat(stack("mean_nsd"), np.nanmean)),
            "mean_nsd_std": float(std(stack("mean_nsd"))),
        }
        print_cv(folds, summaries, agg)

        output = args.output or PROJECT_ROOT / "results" / "evaluation_cv.json"
        nn = lambda v: None if np.isnan(v) else float(v)  # noqa: E731
        payload = {
            "model": args.model,
            "postprocess": args.postprocess,
            "min_component_ratio": args.min_component_ratio,
            "max_component_distance_mm": args.max_component_distance,
            "checkpoint_kind": args.checkpoint_kind,
            "nsd_tolerance_mm": args.nsd_tolerance,
            "inference": inference_settings(args),
            "folds": folds,
            "checkpoints": checkpoints,
            "mean_dice": {"mean": agg["mean_dice_mean"], "std": agg["mean_dice_std"]},
            "mean_hd95": {"mean": nn(agg["mean_hd95_mean"]), "std": nn(agg["mean_hd95_std"])},
            "mean_nsd": {"mean": nn(agg["mean_nsd_mean"]), "std": nn(agg["mean_nsd_std"])},
            "dice_per_organ": {"mean": to_list(agg["dice_per_organ_mean"]), "std": to_list(agg["dice_per_organ_std"])},
            "hd95_per_organ": {"mean": to_list(agg["hd95_per_organ_mean"]), "std": to_list(agg["hd95_per_organ_std"])},
            "nsd_per_organ": {"mean": to_list(agg["nsd_per_organ_mean"]), "std": to_list(agg["nsd_per_organ_std"])},
            "per_fold": [
                {
                    "fold": fold,
                    "num_patients": s["num_patients"],
                    "mean_dice": s["mean_dice"],
                    "mean_hd95": nn(s["mean_hd95"]),
                    "mean_nsd": nn(s["mean_nsd"]),
                    "dice_per_organ": to_list(s["dice_per_organ"]),
                    "hd95_per_organ": to_list(s["hd95_per_organ"]),
                    "nsd_per_organ": to_list(s["nsd_per_organ"]),
                    "missed_per_organ": s["missed"].tolist(),
                }
                for fold, s in zip(folds, summaries)
            ],
            "per_volume": [{"fold": fold, **v} for fold, s in zip(folds, summaries) for v in s["volumes"]],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
