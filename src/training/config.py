"""
Training configuration shared by scripts/train.py and scripts/train_cv.py.
Every field can be overridden from the command line.

The defaults are the reference configuration of the README (run cv_classes_w5_aug).
Earlier settings can be reproduced with:
    Phase 1 defaults:        --batch-size 1 --max-epochs 100 --sampling posneg
                             --rare-loss-weight 1 --augmentation light
    original train.py:       the Phase 1 defaults + --loss gdl --num-samples 1
                             --accumulate 8 --lr-scheduler none
"""

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "unet"           # unet | swin_unetr
    loss: str = "dicece"               # gdl | dicece
    spatial_size: tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 2                # volumes per step (2 x 4 patches of 96^3 fit in 4 GB of VRAM)
    num_samples: int = 4               # patches drawn per volume (effective batch = batch_size * num_samples)
    accumulate_grad_batches: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"       # none | cosine
    warmup_epochs: int = 5
    max_epochs: int = 250
    precision: str = "16-mixed"
    seed: int = 42
    intensity_shift: bool = True       # RandShiftIntensityd augmentation (the baseline had none)
    sampling: str = "classes"          # posneg | classes (class-balanced patch centers)
    rare_weight: float = 3.0           # sampling weight of the rare organs when sampling="classes"
    rare_loss_weight: float = 5.0      # loss weight of the rare organs (dicece only); 1 = unweighted
    augmentation: str = "strong"       # light | strong (see src/data/transforms.get_augmentations)
    flip: str = "none"                 # none | lr | all: random mirroring with left/right label swap
    min_free_gb: float = 2.0           # stop cleanly below this free disk space (see DiskSpaceGuard)
    logger: str = "wandb"              # wandb | csv
    run_name: str | None = None
    fast_dev_run: bool = False
    resume: bool = False

    @property
    def name(self) -> str:
        return self.run_name or f"{self.model_name}_{self.loss}"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        return cls(
            model_name=args.model,
            loss=args.loss,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            accumulate_grad_batches=args.accumulate,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs,
            seed=args.seed,
            intensity_shift=not args.no_intensity_shift,
            sampling=args.sampling,
            rare_weight=args.rare_weight,
            rare_loss_weight=args.rare_loss_weight,
            augmentation=args.augmentation,
            flip=args.flip,
            min_free_gb=args.min_free_gb,
            logger=args.logger,
            run_name=args.run_name,
            fast_dev_run=args.fast_dev_run,
            resume=args.resume,
        )


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    d = TrainConfig()
    parser.add_argument("--model", default=d.model_name, choices=["unet", "swin_unetr"])
    parser.add_argument("--loss", default=d.loss, choices=["gdl", "dicece"])
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--num-samples", type=int, default=d.num_samples, help="Patches per volume.")
    parser.add_argument("--accumulate", type=int, default=d.accumulate_grad_batches)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--lr-scheduler", default=d.lr_scheduler, choices=["none", "cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    parser.add_argument("--max-epochs", type=int, default=d.max_epochs)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--no-intensity-shift", action="store_true", help="Disable RandShiftIntensityd.")
    parser.add_argument("--sampling", default=d.sampling, choices=["posneg", "classes"],
                        help="Patch center sampling: any organ vs background, or class-balanced.")
    parser.add_argument("--rare-weight", type=float, default=d.rare_weight,
                        help="With --sampling classes: weight of the rare organs (others = 1, background = 2).")
    parser.add_argument("--rare-loss-weight", type=float, default=d.rare_loss_weight,
                        help="Weight of the rare organs in the DiceCE loss (others = 1; 1 = no weighting).")
    parser.add_argument("--augmentation", default=d.augmentation, choices=["light", "strong"],
                        help="Patch augmentations: light (original) or strong (nnU-Net-like ranges, no flip).")
    parser.add_argument("--flip", default=d.flip, choices=["none", "lr", "all"],
                        help="Random mirroring of the patches (p=0.5 per axis): left-right axis only (lr) or "
                             "the 3 axes (all, as nnU-Net); the right/left labels (kidneys, adrenal glands) "
                             "are swapped when the left-right axis is flipped.")
    parser.add_argument("--min-free-gb", type=float, default=d.min_free_gb,
                        help="Free disk space (GB) below which the training refuses to start, or stops "
                             "cleanly at the end of the epoch (resumable with --resume) and exits with an error.")
    parser.add_argument("--logger", default=d.logger, choices=["wandb", "csv"],
                        help="csv writes to results/logs (no W&B account needed).")
    parser.add_argument("--run-name", default=None, help="Default: <model>_<loss>.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted training from last.ckpt (same options and --run-name as before). "
                             "Stop a training cleanly with Ctrl+C, or by creating the file STOP in its checkpoint folder.")
    parser.add_argument("--fast-dev-run", action="store_true", help="1 train + 1 val batch, to check the code runs.")
    return parser
