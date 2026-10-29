"""
Builders turning a TrainConfig into the objects needed for training
(loss, Lightning module, trainer, transforms), so that train.py and
train_cv.py cannot drift apart.
"""

import csv
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss, GeneralizedDiceLoss
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.optim import AdamW

from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.builder import get_model
from src.training.callbacks import GracefulStop
from src.training.config import TrainConfig
from src.training.lightning_module import OrganMapperModule

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_loss(name: str) -> torch.nn.Module:
    if name == "gdl":
        return GeneralizedDiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    if name == "dicece":
        # Dice on the organs + cross-entropy on all classes (CE keeps gradients stable early on)
        return DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    raise ValueError(f"Unknown loss '{name}'. Choose 'gdl' or 'dicece'.")


def build_transforms(cfg: TrainConfig):
    train = get_train_transforms(spatial_size=cfg.spatial_size, num_samples=cfg.num_samples,
                                intensity_shift=cfg.intensity_shift,
                                sampling=cfg.sampling, rare_weight=cfg.rare_weight)
    val = get_val_transforms(spatial_size=cfg.spatial_size)
    return train, val


def build_module(cfg: TrainConfig) -> OrganMapperModule:
    return OrganMapperModule(
        model=get_model(model_name=cfg.model_name, spatial_size=cfg.spatial_size),
        loss_function=build_loss(cfg.loss),
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": cfg.lr, "weight_decay": cfg.weight_decay},
        val_roi_size=cfg.spatial_size,
        lr_scheduler=cfg.lr_scheduler,
        warmup_epochs=cfg.warmup_epochs,
        max_epochs=cfg.max_epochs,
    )


def _latest_csv_version(log_dir: Path) -> int | None:
    """Highest N of the existing log_dir/version_N folders (None if there is none)."""
    versions = [int(p.name.split("_")[1]) for p in log_dir.glob("version_*") if p.name.split("_")[1].isdigit()]
    return max(versions) if versions else None


def _read_csv_history(path: Path) -> list[dict[str, float]]:
    """Rows of a CSVLogger metrics.csv (empty cells dropped), [] if the file does not exist."""
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]


def build_trainer(
    cfg: TrainConfig, run_name: str, ckpt_dir: Path, group: str | None = None, resume: bool = False
) -> pl.Trainer:
    """
    Args:
        run_name: Name of the logged run (e.g. "unet_dicece_fold_1").
        ckpt_dir: Where the best checkpoint and last.ckpt are written (absolute path).
        group: W&B group (used to gather the folds of one CV).
        resume: Continue the existing log of this run instead of starting a new one
            (the weights themselves are restored by passing ckpt_path to trainer.fit).
    """
    if cfg.logger == "wandb":
        # Same id on resume so that the curves continue in the same W&B run
        wandb_kwargs = {"id": run_name, "resume": "allow"} if resume else {}
        logger = WandbLogger(project="organ-mapper", name=run_name, group=group, **wandb_kwargs)
    else:
        save_dir = PROJECT_ROOT / "results" / "logs"
        version = _latest_csv_version(save_dir / run_name) if resume else None
        history = _read_csv_history(save_dir / run_name / f"version_{version}" / "metrics.csv") if resume and version is not None else []
        logger = CSVLogger(save_dir=str(save_dir), name=run_name, version=version)
        # CSVLogger deletes the old metrics.csv when it opens the folder: put the rows back
        # so that the epochs run before the stop stay in the log.
        logger.experiment.metrics = history

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        # "/" in a metric name would create a subfolder; keep the name out of the file name
        filename=f"{run_name}-epoch{{epoch:02d}}-dice{{val/dice:.2f}}",
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor="val/dice",
        mode="max",
        save_last=True,  # last.ckpt = full state (weights, optimizer, scheduler) to resume from
    )

    return pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            GracefulStop(Path(ckpt_dir) / "STOP"),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.precision,
        log_every_n_steps=5,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        fast_dev_run=cfg.fast_dev_run,
    )


def is_finished(ckpt_dir: Path) -> bool:
    """True if the training that writes to ckpt_dir ran all its epochs (see fit_with_resume)."""
    return (Path(ckpt_dir) / "DONE").exists()


def fit_with_resume(
    trainer: pl.Trainer,
    module: OrganMapperModule,
    train_loader,
    val_loader,
    ckpt_dir: Path,
    resume: bool = False,
) -> bool:
    """
    Runs trainer.fit, restarting from ckpt_dir/last.ckpt when resume is set.

    Returns:
        True if all epochs were run (a DONE marker is then written in ckpt_dir),
        False if the training was stopped early (Ctrl+C or STOP file); relaunch
        the same command with --resume to continue it.
    """
    ckpt_dir = Path(ckpt_dir)
    last = ckpt_dir / "last.ckpt"
    if not resume:
        (ckpt_dir / "DONE").unlink(missing_ok=True)  # a fresh run replaces the previous one
    ckpt_path = str(last) if resume and last.exists() else None
    if resume:
        print(f"Resuming from {last}" if ckpt_path else f"No {last.name} in {ckpt_dir}: starting from scratch.")

    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    stopper = next(c for c in trainer.callbacks if isinstance(c, GracefulStop))
    finished = not stopper.requested or trainer.current_epoch >= trainer.max_epochs
    if finished and not trainer.fast_dev_run:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "DONE").write_text("")
    return finished
