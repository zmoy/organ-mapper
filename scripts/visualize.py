"""
Side-by-side figure of a CT volume, its ground truth (optional) and a predicted mask,
on two axial slices and one coronal slice, in the radiological convention (patient's
right on the left of the screen). With a ground truth, the legend gives the Dice of
each organ for this volume.

The prediction must be in the patient space of the image, as written by predict.py.

Usage:
    python scripts/visualize.py --image data/raw/imagesTr/img0007.nii.gz \
        --label data/raw/labelsTr/label0007.nii.gz \
        --prediction predictions/img0007_preds.nii.gz
    python scripts/visualize.py --image data/raw/imagesTs/img0061.nii.gz \
        --prediction predictions/img0061_preds.nii.gz      # no ground truth
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.transforms import HU_WINDOW, NUM_CLASSES, ORGAN_NAMES  # same window as the training

# One clearly distinct colour per organ (none close to grey, so that each stays visible on the CT)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
          "#ffd700", "#00e5ff", "#ff00ff", "#7cfc00", "#ff1493", "#00bfa0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CT / ground truth / prediction comparison figure.")
    parser.add_argument("--image", type=Path, required=True, help="CT volume (.nii.gz).")
    parser.add_argument("--prediction", type=Path, required=True, help="Predicted mask written by predict.py.")
    parser.add_argument("--label", type=Path, default=None, help="Ground truth mask (optional).")
    parser.add_argument("--focus-organ", type=int, default=4,
                        help="Organ (1-13) the second axial slice goes through. Default: 4 (gallbladder).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .png. Default: next to the prediction, <name>_figure.png.")
    return parser.parse_args()


def load_ras(path: Path) -> nib.Nifti1Image:
    """Loads a NIfTI file reoriented to RAS, so that every volume shares the same axes."""
    return nib.as_closest_canonical(nib.load(str(path)))


def to_label_map(img: nib.Nifti1Image) -> np.ndarray:
    data = np.asarray(img.dataobj).astype(np.uint8)
    return data.reshape(data.shape[:3])  # drops a trailing channel axis, if any


def dice_per_organ(gt: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    """Dice of each organ present in the ground truth."""
    scores = {}
    for organ in range(1, NUM_CLASSES):
        g, p = gt == organ, pred == organ
        if g.any():
            scores[organ] = float(2 * (g & p).sum() / (g.sum() + p.sum()))
    return scores


def choose_slices(reference: np.ndarray, focus_organ: int) -> tuple[int, int, int]:
    """
    Returns (z1, z2, y): the axial slice with the most organs, the middle axial slice of
    focus_organ (z1 if it is absent), and a coronal slice through the organs' centre.
    """
    organs_per_slice = [len(np.unique(reference[:, :, z])) for z in range(reference.shape[2])]
    z1 = int(np.argmax(organs_per_slice))
    z_focus = np.flatnonzero((reference == focus_organ).any(axis=(0, 1)))
    z2 = int(np.median(z_focus)) if len(z_focus) else z1
    y = int(np.median(np.flatnonzero((reference > 0).any(axis=(0, 2))))) if reference.any() else reference.shape[1] // 2
    return z1, z2, y


def main():
    args = parse_args()
    image = load_ras(args.image)
    ct = image.get_fdata(dtype=np.float32)
    pred = to_label_map(load_ras(args.prediction))
    gt = to_label_map(load_ras(args.label)) if args.label else None
    if pred.shape != ct.shape or (gt is not None and gt.shape != ct.shape):
        raise ValueError(f"Shape mismatch: image {ct.shape}, prediction {pred.shape}"
                         + (f", label {gt.shape}" if gt is not None else "")
                         + ". The prediction must be in the patient space (predict.py output).")

    sx, sy, sz = image.header.get_zooms()[:3]
    reference = gt if gt is not None else pred
    z1, z2, y = choose_slices(reference, args.focus_organ)
    focus_name = ORGAN_NAMES[args.focus_organ - 1]

    # rot90 puts anterior at the top, fliplr puts the patient's right on the left of the screen
    views = [
        (f"Axial z={z1}", lambda v: np.fliplr(np.rot90(v[:, :, z1])), sy / sx),
        (f"Axial z={z2} ({focus_name})", lambda v: np.fliplr(np.rot90(v[:, :, z2])), sy / sx),
        (f"Coronal y={y}", lambda v: np.fliplr(np.rot90(v[:, y, :])), sz / sx),
    ]
    window = lambda a: np.clip((a - HU_WINDOW[0]) / (HU_WINDOW[1] - HU_WINDOW[0]), 0, 1)  # noqa: E731
    cmap = ListedColormap(COLORS)

    def overlay(ax, ct_slice, label_slice, aspect):
        ax.imshow(window(ct_slice), cmap="gray", aspect=aspect)
        masked = np.ma.masked_where(label_slice == 0, label_slice.astype(int) - 1)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 2, alpha=0.6, interpolation="nearest", aspect=aspect)
        ax.axis("off")

    panels = [("CT", None)] + ([("Ground truth", gt)] if gt is not None else []) + [("Prediction", pred)]
    fig, axes = plt.subplots(3, len(panels), figsize=(5 * len(panels), 15), facecolor="white", squeeze=False)
    for row, (title, cut, aspect) in enumerate(views):
        for col, (name, volume) in enumerate(panels):
            ax = axes[row, col]
            if volume is None:
                ax.imshow(window(cut(ct)), cmap="gray", aspect=aspect)
                ax.axis("off")
                ax.set_title(f"{title}: {name}", fontsize=12)
            else:
                overlay(ax, cut(ct), cut(volume), aspect)
                ax.set_title(name, fontsize=12)

    scores = dice_per_organ(gt, pred) if gt is not None else {}
    handles = []
    for organ in range(1, NUM_CLASSES):
        text = f"{organ} {ORGAN_NAMES[organ - 1]}"
        if gt is not None:
            text += f": Dice {scores[organ]:.2f}" if organ in scores else ": absent"
        handles.append(Patch(color=COLORS[organ - 1], label=text))
    fig.legend(handles=handles, loc="lower center", ncol=5 if len(panels) == 3 else 4, fontsize=10, frameon=False)
    suptitle = args.image.name
    if scores:
        suptitle += f": mean Dice {np.mean(list(scores.values())):.3f}"
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    output = args.output or args.prediction.with_name(args.prediction.name.replace(".nii.gz", "") + "_figure.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=90)
    print(f"Figure saved to {output}")
    for organ, score in scores.items():
        print(f"  {organ:>2} {ORGAN_NAMES[organ - 1]:<20} Dice {score:.3f}")


if __name__ == "__main__":
    main()
