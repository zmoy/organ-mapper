"""
Exports the models of a cross-validation run for publication (GitHub release, Hugging Face Hub):
one weights-only file per fold and a manifest.json (see src/export.py and MODEL_CARD.md).

Usage:
    python scripts/export_models.py --checkpoints-dir checkpoints/cv_classes_w5_aug \
        --evaluation results/evaluation_cv_classes_w5_aug_pp_best.json

The exported files work with predict.py and evaluate.py like the checkpoints:
    python scripts/predict.py --checkpoint exports/cv_classes_w5_aug/organ_mapper_unet_fold_2.pt --postprocess
    python scripts/predict.py --ensemble-dir exports/cv_classes_w5_aug --postprocess
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.export import export_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the fold models of a run (weights only + manifest).")
    parser.add_argument("--checkpoints-dir", type=Path, required=True, help="Run directory (fold_1/, ...).")
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Evaluation JSON of the same checkpoints (scripts/evaluate.py --cv).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Destination. Default: exports/<run name>.")
    parser.add_argument("--checkpoint-kind", default="best", choices=["best", "last"])
    parser.add_argument("--prefix", default="organ_mapper_unet", help="File name prefix.")
    args = parser.parse_args()

    output_dir = args.output_dir or PROJECT_ROOT / "exports" / args.checkpoints_dir.name
    manifest = export_run(args.checkpoints_dir, args.evaluation, output_dir, args.prefix, args.checkpoint_kind)
    for fold in manifest["folds"]:
        size = (output_dir / fold["file"]).stat().st_size / 1e6
        print(f"fold {fold['fold']}: {fold['file']} ({size:.1f} MB, epoch {fold['epoch']}, "
              f"validation Dice {fold['validation']['dice']:.3f}) sha256 {fold['sha256'][:12]}...")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Cross-validation Dice {manifest['evaluation']['mean_dice']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
