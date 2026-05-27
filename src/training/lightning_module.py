"""
PyTorch Lightning module for 3D medical image segmentation.
Encapsulates the model, loss, optimizer, and training/validation steps.
"""

import torch
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

class OrganMapperModule(pl.LightningModule):
    def __init__(
        self, 
        model: torch.nn.Module, 
        loss_function: torch.nn.Module, 
        optimizer_class: type,
        optimizer_kwargs: dict,
        val_roi_size: tuple[int, int, int] = (96, 96, 96),
        num_classes: int = 14
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.val_roi_size = val_roi_size
        
        # Save hyperparameters to W&B
        self.save_hyperparameters(ignore=['model', 'loss_function'])

        # --- NEW: Dice Score Metric ---
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Post-processing to convert raw predictions into discrete one-hot formats
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch["image"], batch["label"]
        
    #     # 1. GPU Inference
    #     outputs = sliding_window_inference(
    #         inputs=images, 
    #         roi_size=self.val_roi_size, 
    #         sw_batch_size=1, 
    #         predictor=self.model
    #     )
        
    #     # 2. CPU Offloading (FIX OOM 4GB VRAM)
    #     # We move these massive full-volume tensors back to system RAM!
    #     outputs = outputs.cpu()
    #     labels = labels.cpu()
        
    #     # Calculate loss on CPU
    #     loss = self.loss_function(outputs, labels)
    #     self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        
    #     # 3. Calculate Dice Score
    #     # Apply post-processing (list of tensors required by MONAI metrics)
    #     val_outputs = [self.post_pred(i) for i in outputs]
    #     val_labels = [self.post_label(i) for i in labels]
        
    #     self.dice_metric(y_pred=val_outputs, y=val_labels)
        
    #     return loss

    # Validation step without loss calculation to save RAM, only compute Dice score (Swin UNETR is too big for GPU memory with loss)
    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch["image"], batch["label"]
        
    #     # 1. GPU Inference
    #     outputs = sliding_window_inference(
    #         inputs=images, 
    #         roi_size=self.val_roi_size, 
    #         sw_batch_size=1, 
    #         predictor=self.model
    #     )
        
    #     # 2. CPU Offloading
    #     outputs = outputs.cpu()
    #     labels = labels.cpu()
        
    #     # ON NE CALCULE PLUS LA LOSS ICI POUR SAUVER 1.5 GO DE RAM
        
    #     # 3. Calculate Dice Score Directly
    #     val_outputs = [self.post_pred(i) for i in outputs]
    #     val_labels = [self.post_label(i) for i in labels]
        
    #     self.dice_metric(y_pred=val_outputs, y=val_labels)
        
    #     # On ne retourne rien, Lightning s'en accommodera très bien
    #     return None

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Inférence directe sur le patch (pas de sliding window)
        outputs = self.forward(images)
        
        # Offloading vers le CPU (les tenseurs sont maintenant tout petits !)
        outputs = outputs.cpu()
        labels = labels.cpu()
        
        # Calcul direct du Dice Score sur ce patch
        val_outputs = [self.post_pred(i) for i in outputs]
        val_labels = [self.post_label(i) for i in labels]
        
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        
        return None

    def on_validation_epoch_end(self):
        # Calculate mean Dice score over the whole validation set
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        # Log to Weights & Biases
        self.log("val/dice", mean_dice, prog_bar=True, logger=True)