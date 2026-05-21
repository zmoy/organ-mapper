"""
PyTorch Lightning module for 3D medical image segmentation.
Encapsulates the model, loss, optimizer, and training/validation steps.
"""

import torch
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference

class OrganMapperModule(pl.LightningModule):
    def __init__(
        self, 
        model: torch.nn.Module, 
        loss_function: torch.nn.Module, 
        optimizer_class: type,
        optimizer_kwargs: dict,
        val_roi_size: tuple[int, int, int] = (96, 96, 96)
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.val_roi_size = val_roi_size
        
        # Save hyperparameters to W&B automatically
        self.save_hyperparameters(ignore=['model', 'loss_function'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Forward pass
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        
        # Log training loss to W&B
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Sliding window inference for validation (uses full volume)
        sw_batch_size = 4 # Number of patches to process simultaneously
        outputs = sliding_window_inference(
            inputs=images, 
            roi_size=self.val_roi_size, 
            sw_batch_size=sw_batch_size, 
            predictor=self.model
        )
        
        loss = self.loss_function(outputs, labels)
        
        # Log validation loss to W&B
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss