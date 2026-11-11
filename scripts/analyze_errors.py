"""
Where does the HD95 come from? Reads the "per_volume" diagnostics of an evaluation JSON
(scripts/evaluate.py, see src/error_analysis.py) and prints, per organ:
    - mean and median HD95 (a mean far above the median: a few volumes carry the error),
      and the two directions of the HD95: prediction -> label (predicted voxels far from the
      annotation) and label -> prediction (parts of the organ left out);
    - mean craniocaudal over-extent above / below the annotated slices;
then the worst (volume, organ) cases, with the commands that draw them
(predict.py with the checkpoint of the fold where the volume was in validation, then
visualize.py centred on that organ).

Usage:
    python scripts/analyze_errors.py results/evaluation_cv_classes_w5_aug_pp_best.json
    python scripts/analyze_errors.py <json> --organs 8 10 --top 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.transforms import ORGAN_NAMES


def values(volumes: list[dict], key: str, organ: int) -> np.ndarray:
    return np.array([np.nan if v[key][organ - 1] is None else v[key][organ - 1] for v in volumes], dtype=float)


def nanmean(a: np.ndarray) -> float:
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-organ origin of the HD95 and worst cases.")
    parser.add_argument("evaluation", type=Path, help="Evaluation JSON with per_volume results.")
    parser.add_argument("--organs", type=int, nargs="+", default=list(range(1, 14)), help="Organ labels.")
    parser.add_argument("--top", type=int, default=8, help="Number of worst cases to list.")
    args = parser.parse_args()

    data = json.loads(args.evaluation.read_text(encoding="utf-8"))
    volumes = data.get("per_volume")
    if not volumes:
        print("No 'per_volume' in this evaluation: run scripts/evaluate.py again.")
        return 1

    print(f"{args.evaluation.name}: {len(volumes)} volumes\n")
    print(f"{'Organ':<24}{'HD95':>7}{'median':>8}{'pred->lab':>11}{'lab->pred':>11}{'pred->lab':>11}"
          f"{'above':>8}{'below':>8}{'missed':>8}")
    print(f"{'':<24}{'(mm)':>7}{'(mm)':>8}{'(mm)':>11}{'(mm)':>11}{'dominates':>11}{'(mm)':>8}{'(mm)':>8}")
    for organ in args.organs:
        hd95 = values(volumes, "hd95", organ)
        median = float(np.nanmedian(hd95)) if np.isfinite(hd95).any() else float("nan")
        to_label = values(volumes, "hd95_pred_to_label", organ)
        to_pred = values(volumes, "hd95_label_to_pred", organ)
        both = np.isfinite(to_label) & np.isfinite(to_pred)
        dominates = f"{int((to_label[both] > to_pred[both]).sum())}/{int(both.sum())}"
        missed = sum(v["missed"][organ - 1] for v in volumes)
        print(f"{f'{organ} {ORGAN_NAMES[organ - 1]}':<24}{nanmean(hd95):>7.1f}{median:>8.1f}{nanmean(to_label):>11.1f}"
              f"{nanmean(to_pred):>11.1f}{dominates:>11}"
              f"{nanmean(values(volumes, 'overextent_superior_mm', organ)):>+8.1f}"
              f"{nanmean(values(volumes, 'overextent_inferior_mm', organ)):>+8.1f}{missed:>8}")
    print("\nmedian: median HD95 over the volumes; far below the mean when a few volumes carry the error."
          "\npred->lab: predicted voxels far from the annotation (false positive region, organ segmented"
          "\ntoo far); lab->pred: parts of the organ not predicted. above / below: how far the prediction"
          "\ngoes beyond the highest / lowest annotated slice (negative: stops before). Missed organs have"
          "\nHD95 = volume diagonal and no direction.")

    cases = [
        (v["hd95"][organ - 1], v, organ)
        for v in volumes for organ in args.organs if v["hd95"][organ - 1] is not None
    ]
    cases.sort(key=lambda c: -c[0])
    print(f"\nWorst {args.top} cases:")
    commands = []
    folds, checkpoints = data.get("folds", []), data.get("checkpoints", [])
    for hd95, v, organ in cases[:args.top]:
        i = organ - 1
        detail = ("missed" if v["missed"][i] else
                  f"pred->lab {v['hd95_pred_to_label'][i]:.1f}, lab->pred {v['hd95_label_to_pred'][i]:.1f}, "
                  f"above {v['overextent_superior_mm'][i]:+.0f}, below {v['overextent_inferior_mm'][i]:+.0f} mm")
        fold = v.get("fold")
        print(f"  {v['name']} (fold {fold}) organ {organ} {ORGAN_NAMES[i]}: HD95 {hd95:.1f} mm, "
              f"Dice {v['dice'][i]:.3f} ({detail})")
        if fold in folds:
            checkpoint = checkpoints[folds.index(fold)]
            out_dir = f"predictions/analysis/fold_{fold}"
            number = v["name"].removeprefix("img")
            predict = (f"python scripts/predict.py --checkpoint {checkpoint} --postprocess "
                       f"--image data/raw/imagesTr/{v['name']}.nii.gz --output-dir {out_dir}")
            if predict not in commands:
                commands.append(predict)
            commands.append(
                f"python scripts/visualize.py --image data/raw/imagesTr/{v['name']}.nii.gz "
                f"--label data/raw/labelsTr/label{number}.nii.gz "
                f"--prediction {out_dir}/{v['name']}_preds.nii.gz --focus-organ {organ} "
                f"--output {out_dir}/{v['name']}_organ{organ}.png"
            )
    if commands:
        print("\nCommands to draw them (one GPU task at a time; predict.py ~2-3 min per volume):")
        print("\n".join(commands))
    return 0


if __name__ == "__main__":
    sys.exit(main())
