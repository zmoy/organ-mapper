"""
Offline clinical evaluation script.
Computes per-organ Dice Score and Hausdorff Distance (HD95) on full validation volumes
(sliding window inference, outputs kept on the CPU to stay within 4 GB of VRAM).

Two modes:
    - Hold-out (default): one checkpoint on the validation split used by train.py.
        python scripts/evaluate.py --checkpoint checkpoints/xxx.ckpt
    - Cross-validation (--cv): one checkpoint per fold (checkpoints/fold_k/*.ckpt, written
      by train_cv.py), each evaluated on the volumes it never saw during training.
      Reports mean +/- std over the folds.
        python scripts/evaluate.py --cv
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.builder import get_model
from src.training.lightning_module import OrganMapperModule
from src.data.dataset import get_data_dicts, get_kfold_splits, split_train_val
from src.data.transforms import get_val_transforms

SPATIAL_SIZE = (96, 96, 96)
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
    return parser.parse_args()


def nan_stat(values: np.ndarray, func, axis: int = 0) -> np.ndarray:
    """nanmean / nanstd that stay quiet when a slice is all-NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(values, axis=axis)


def to_list(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in values]


def find_fold_checkpoint(checkpoints_dir: Path, fold: int) -> Path:
    candidates = sorted(
        (checkpoints_dir / f"fold_{fold}").glob("*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"No .ckpt found in {checkpoints_dir / f'fold_{fold}'}")
    return candidates[-1]


def evaluate_checkpoint(
    checkpoint: Path, model_name: str, val_files: list[dict], device: torch.device
) -> dict:
    """
    Evaluates one checkpoint on full volumes.

    Returns:
        dict with per-patient arrays (patients x organs): "dice", "hd95" (NaN where undefined),
        and "missed" (organs present in the label but never predicted, per organ).
    """
    print(f"Loading checkpoint: {checkpoint}")
    model = get_model(model_name=model_name, spatial_size=SPATIAL_SIZE)
    pl_module = OrganMapperModule.load_from_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        loss_function=torch.nn.Identity(),
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={}
    )
    model = pl_module.model.to(device)
    model.eval()

    val_ds = Dataset(data=val_files, transform=get_val_transforms(spatial_size=SPATIAL_SIZE, full_volume=True))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # reduction="none": one value per patient and per organ
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    dice_rows, hd95_rows = [], []
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

            val_outputs = torch.stack([post_pred(o) for o in outputs.float()])
            val_labels = torch.stack([post_label(l) for l in labels])

            dice = dice_metric(y_pred=val_outputs, y=val_labels)[0].cpu().numpy()
            hd95 = hd95_metric(y_pred=val_outputs, y=val_labels)[0].cpu().numpy()

            label_present = val_labels[0, 1:].flatten(1).sum(dim=1).numpy() > 0
            pred_present = val_outputs[0, 1:].flatten(1).sum(dim=1).numpy() > 0

            # Organ absent from the ground truth: metric undefined for this patient
            dice[~label_present] = np.nan
            hd95[~label_present] = np.nan
            # Organ present but never predicted: HD95 is infinite, count it as a miss instead
            miss = label_present & ~pred_present
            hd95[miss] = np.nan
            missed += miss

            dice_rows.append(dice)
            hd95_rows.append(hd95)

            del images, labels, outputs, val_outputs, val_labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {"dice": np.stack(dice_rows), "hd95": np.stack(hd95_rows), "missed": missed}


def summarize_run(result: dict) -> dict:
    """Collapses per-patient arrays into per-organ means and overall means."""
    dice_per_organ = nan_stat(result["dice"], np.nanmean)
    hd95_per_organ = nan_stat(result["hd95"], np.nanmean)
    return {
        "num_patients": int(result["dice"].shape[0]),
        "dice_per_organ": dice_per_organ,
        "hd95_per_organ": hd95_per_organ,
        "mean_dice": float(np.nanmean(dice_per_organ)),
        "mean_hd95": float(np.nanmean(hd95_per_organ)) if not np.all(np.isnan(hd95_per_organ)) else np.nan,
        "missed": result["missed"],
    }


def fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None or np.isnan(value) else format(value, spec)


def print_holdout(summary: dict) -> None:
    print("\n" + "=" * 52)
    print("FINAL CLINICAL RESULTS (full volumes)")
    print("=" * 52)
    print(f"{'Organ':<10}{'Dice':>10}{'HD95 (mm)':>14}{'Missed':>10}")
    for k in range(NUM_CLASSES - 1):
        print(f"{k + 1:<10}{fmt(summary['dice_per_organ'][k], '.3f'):>10}"
              f"{fmt(summary['hd95_per_organ'][k], '.1f'):>14}{summary['missed'][k]:>10}")
    print("-" * 52)
    print(f"Mean Dice: {fmt(summary['mean_dice'], '.4f')}  |  "
          f"Mean HD95: {fmt(summary['mean_hd95'], '.2f')} mm")
    print("(HD95 only covers organs that were predicted at least once)")
    print("=" * 52)


def print_cv(folds: list[int], summaries: list[dict], agg: dict) -> None:
    print("\n" + "=" * 64)
    print(f"CROSS-VALIDATION RESULTS ({len(folds)} folds, full volumes)")
    print("=" * 64)
    print(f"{'Fold':<8}{'Dice':>10}{'HD95 (mm)':>14}")
    for fold, s in zip(folds, summaries):
        print(f"{fold:<8}{fmt(s['mean_dice'], '.3f'):>10}{fmt(s['mean_hd95'], '.1f'):>14}")
    print("-" * 64)
    print(f"{'Organ':<8}{'Dice (mean +/- std)':>24}{'HD95 mm (mean +/- std)':>26}")
    for k in range(NUM_CLASSES - 1):
        dice = f"{fmt(agg['dice_per_organ_mean'][k], '.3f')} +/- {fmt(agg['dice_per_organ_std'][k], '.3f')}"
        hd = f"{fmt(agg['hd95_per_organ_mean'][k], '.1f')} +/- {fmt(agg['hd95_per_organ_std'][k], '.1f')}"
        print(f"{k + 1:<8}{dice:>24}{hd:>26}")
    print("-" * 64)
    print(f"Mean Dice: {agg['mean_dice_mean']:.4f} +/- {agg['mean_dice_std']:.4f}")
    print(f"Mean HD95: {fmt(agg['mean_hd95_mean'], '.2f')} +/- {fmt(agg['mean_hd95_std'], '.2f')} mm")
    print("(std = sample std over folds; HD95 only covers organs predicted at least once)")
    print("=" * 64)


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / "data" / "raw"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running clinical evaluation on device: {device}\n")

    data_dicts = get_data_dicts(data_dir)

    # --- Hold-out mode ---
    if not args.cv:
        _, val_files = split_train_val(data_dicts)
        result = evaluate_checkpoint(args.checkpoint, args.model, val_files, device)
        summary = summarize_run(result)
        print_holdout(summary)

        output = args.output or PROJECT_ROOT / "results" / "evaluation.json"
        payload = {
            "checkpoint": str(args.checkpoint),
            "model": args.model,
            "num_patients": summary["num_patients"],
            "mean_dice": summary["mean_dice"],
            "mean_hd95": None if np.isnan(summary["mean_hd95"]) else summary["mean_hd95"],
            "dice_per_organ": to_list(summary["dice_per_organ"]),
            "hd95_per_organ": to_list(summary["hd95_per_organ"]),
            "missed_per_organ": summary["missed"].tolist(),
        }
    # --- Cross-validation mode ---
    else:
        splits = get_kfold_splits(data_dicts, n_splits=args.n_splits)
        folds = args.folds or list(range(1, args.n_splits + 1))
        summaries, checkpoints = [], []
        for fold in folds:
            print(f"\n{'=' * 50}\nFOLD {fold}/{args.n_splits}\n{'=' * 50}")
            ckpt = find_fold_checkpoint(args.checkpoints_dir, fold)
            checkpoints.append(str(ckpt))
            _, val_files = splits[fold - 1]
            summaries.append(summarize_run(evaluate_checkpoint(ckpt, args.model, val_files, device)))

        stack = lambda key: np.stack([s[key] for s in summaries])  # noqa: E731
        ddof = 1 if len(folds) > 1 else 0
        std = lambda arr: nan_stat(arr, lambda a, axis: np.nanstd(a, axis=axis, ddof=ddof))  # noqa: E731
        agg = {
            "dice_per_organ_mean": nan_stat(stack("dice_per_organ"), np.nanmean),
            "dice_per_organ_std": std(stack("dice_per_organ")),
            "hd95_per_organ_mean": nan_stat(stack("hd95_per_organ"), np.nanmean),
            "hd95_per_organ_std": std(stack("hd95_per_organ")),
            "mean_dice_mean": float(np.mean(stack("mean_dice"))),
            "mean_dice_std": float(np.std(stack("mean_dice"), ddof=ddof)),
            "mean_hd95_mean": float(nan_stat(stack("mean_hd95"), np.nanmean)),
            "mean_hd95_std": float(std(stack("mean_hd95"))),
        }
        print_cv(folds, summaries, agg)

        output = args.output or PROJECT_ROOT / "results" / "evaluation_cv.json"
        nn = lambda v: None if np.isnan(v) else float(v)  # noqa: E731
        payload = {
            "model": args.model,
            "folds": folds,
            "checkpoints": checkpoints,
            "mean_dice": {"mean": agg["mean_dice_mean"], "std": agg["mean_dice_std"]},
            "mean_hd95": {"mean": nn(agg["mean_hd95_mean"]), "std": nn(agg["mean_hd95_std"])},
            "dice_per_organ": {"mean": to_list(agg["dice_per_organ_mean"]), "std": to_list(agg["dice_per_organ_std"])},
            "hd95_per_organ": {"mean": to_list(agg["hd95_per_organ_mean"]), "std": to_list(agg["hd95_per_organ_std"])},
            "per_fold": [
                {
                    "fold": fold,
                    "num_patients": s["num_patients"],
                    "mean_dice": s["mean_dice"],
                    "mean_hd95": nn(s["mean_hd95"]),
                    "dice_per_organ": to_list(s["dice_per_organ"]),
                    "hd95_per_organ": to_list(s["hd95_per_organ"]),
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
