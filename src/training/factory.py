"""
Builders turning a TrainConfig into the objects needed for training
(loss, Lightning module, trainer, transforms), so that train.py and
train_cv.py cannot drift apart.
"""

from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss, GeneralizedDiceLoss
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.optim import AdamW

from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.builder import get_model
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
                                intensity_shift=cfg.intensity_shift)
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


def build_trainer(cfg: TrainConfig, run_name: str, ckpt_dir: Path, group: str | None = None) -> pl.Trainer:
    """
    Args:
        run_name: Name of the logged run (e.g. "unet_dicece_fold_1").
        ckpt_dir: Where the best checkpoint is written (absolute path).
        group: W&B group (used to gather the folds of one CV).
    """
    if cfg.logger == "wandb":
        logger = WandbLogger(project="organ-mapper", name=run_name, group=group)
    else:
        logger = CSVLogger(save_dir=str(PROJECT_ROOT / "results" / "logs"), name=run_name)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        # "/" in a metric name would create a subfolder; keep the name out of the file name
        filename=f"{run_name}-epoch{{epoch:02d}}-dice{{val/dice:.2f}}",
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor="val/dice",
        mode="max",
    )

    return pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.precision,
        log_every_n_steps=5,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        fast_dev_run=cfg.fast_dev_run,
    )
