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
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.builder import get_model
from src.data.dataset import get_data_dicts, get_dataset, get_kfold_splits, split_train_val
from src.data.transforms import get_val_transforms
from src.postprocess import keep_largest_components as keep_largest_components_np

SPATIAL_SIZE = (96, 96, 96)
# Voxel size (mm) of the resampled volumes (default of get_val_transforms), so that HD95 is in mm
VOXEL_SPACING = (1.5, 1.5, 2.0)
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
    return parser.parse_args()


def nan_stat(values: np.ndarray, func, axis: int = 0) -> np.ndarray:
    """nanmean / nanstd that stay quiet when a slice is all-NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(values, axis=axis)


def to_list(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in values]


def find_fold_checkpoint(checkpoints_dir: Path, fold: int, kind: str = "best") -> Path:
    """
    Args:
        kind: "best" (best-Dice checkpoint) or "last" (last.ckpt, end of the training).
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


def keep_largest_components(onehot: torch.Tensor) -> torch.Tensor:
    """
    Post-processing: for each organ keep only its largest connected component and
    relabel the other islands as background (removes far-away false detections).

    Args:
        onehot: uint8 one-hot prediction of shape (1, NUM_CLASSES, X, Y, Z).

    Returns:
        A one-hot uint8 tensor of the same shape.
    """
    label_map = keep_largest_components_np(onehot[0].argmax(dim=0).numpy().astype(np.uint8), NUM_CLASSES)
    cleaned = torch.zeros_like(onehot)
    for c in range(NUM_CLASSES):
        cleaned[0, c] = torch.from_numpy(label_map == c)
    return cleaned


def evaluate_checkpoint(
    checkpoint: Path, model_name: str, val_files: list[dict], device: torch.device,
    postprocess: bool = False, nsd_tolerance: float = 2.0
) -> dict:
    """
    Evaluates one checkpoint on full volumes.

    Args:
        postprocess: Keep only the largest connected component of each organ (see
            keep_largest_components) before computing the metrics.
        nsd_tolerance: Tolerance (mm) of the Normalized Surface Distance.

    Returns:
        dict with per-patient arrays (patients x organs): "dice", "hd95", "nsd" (NaN where the
        organ is absent from the label), and "missed" (organs present in the label but never
        predicted, per organ).
    """
    print(f"Loading checkpoint: {checkpoint}")
    model = get_model(model_name=model_name, spatial_size=SPATIAL_SIZE)
    # Only the network weights: the checkpoint may also hold loss buffers (class weights)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    model = model.to(device)
    model.eval()

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

    dice_rows, hd95_rows, nsd_rows = [], [], []
    missed = np.zeros(NUM_CLASSES - 1, dtype=int)

    print(f"Evaluating on {len(val_loader)} full patient volumes...")
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            print(f"  Patient {idx + 1}/{len(val_loader)} - Sliding Window Inference...")
            images = batch_data["image"].to(device)
            labels = batch_data["label"]

            # sw_device: patches on GPU; device: stitched output on CPU
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = sliding_window_inference(
                    images, SPATIAL_SIZE, 1, model, overlap=0.5,
                    sw_device=device, device=torch.device("cpu")
                )

            # One-hot masks as uint8 (4x smaller than float32)
            val_outputs = torch.stack([post_pred(o) for o in outputs.float()]).to(torch.uint8)
            del outputs
            if postprocess:
                val_outputs = keep_largest_components(val_outputs)
            val_labels = torch.stack([post_label(l) for l in labels]).to(torch.uint8)

            label_present = val_labels[0, 1:].flatten(1).sum(dim=1).numpy() > 0
            pred_present = val_outputs[0, 1:].flatten(1).sum(dim=1).numpy() > 0

            # Worst possible distance: diagonal of the volume (mm)
            max_distance = float(np.linalg.norm(np.array(val_labels.shape[2:]) * np.array(VOXEL_SPACING)))

            dice = np.full(NUM_CLASSES - 1, np.nan)
            hd95 = np.full(NUM_CLASSES - 1, np.nan)
            nsd = np.full(NUM_CLASSES - 1, np.nan)
            for organ in np.flatnonzero(label_present):  # organ absent from the ground truth: undefined
                pred = val_outputs[:, organ + 1:organ + 2].float()
                gt = val_labels[:, organ + 1:organ + 2].float()
                dice[organ] = dice_metric(y_pred=pred, y=gt)[0, 0].item()
                if pred_present[organ]:
                    hd95[organ] = hd95_metric(y_pred=pred, y=gt, spacing=VOXEL_SPACING)[0, 0].item()
                    nsd[organ] = nsd_metric(y_pred=pred, y=gt, spacing=VOXEL_SPACING)[0, 0].item()
                else:
                    # Organ present but never predicted: penalised, not left out of the means
                    hd95[organ] = max_distance
                    nsd[organ] = 0.0
            miss = label_present & ~pred_present
            missed += miss

            dice_rows.append(dice)
            hd95_rows.append(hd95)
            nsd_rows.append(nsd)

            del images, labels, val_outputs, val_labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "dice": np.stack(dice_rows), "hd95": np.stack(hd95_rows), "nsd": np.stack(nsd_rows), "missed": missed
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
            args.checkpoint, args.model, val_files, device, args.postprocess, args.nsd_tolerance
        )
        summary = summarize_run(result)
        print_holdout(summary)

        output = args.output or PROJECT_ROOT / "results" / "evaluation.json"
        payload = {
            "checkpoint": str(args.checkpoint),
            "model": args.model,
            "postprocess": args.postprocess,
            "nsd_tolerance_mm": args.nsd_tolerance,
            "num_patients": summary["num_patients"],
            "mean_dice": summary["mean_dice"],
            "mean_hd95": None if np.isnan(summary["mean_hd95"]) else summary["mean_hd95"],
            "mean_nsd": None if np.isnan(summary["mean_nsd"]) else summary["mean_nsd"],
            "dice_per_organ": to_list(summary["dice_per_organ"]),
            "hd95_per_organ": to_list(summary["hd95_per_organ"]),
            "nsd_per_organ": to_list(summary["nsd_per_organ"]),
            "missed_per_organ": summary["missed"].tolist(),
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
            summaries.append(summarize_run(
                evaluate_checkpoint(ckpt, args.model, val_files, device, args.postprocess, args.nsd_tolerance)
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
            "checkpoint_kind": args.checkpoint_kind,
            "nsd_tolerance_mm": args.nsd_tolerance,
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
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
