"""
Model factory module for the Organ Mapper project.
Provides a model-agnostic interface to load different 3D architectures.
"""

from monai.networks.nets import UNet, SwinUNETR
import torch.nn as nn

def get_model(
    model_name: str = "unet", 
    in_channels: int = 1, 
    out_channels: int = 14, 
    spatial_size: tuple[int, int, int] = (96, 96, 96)
) -> nn.Module:
    """
    Instantiates the requested 3D segmentation model.
    
    Args:
        model_name: "unet" (baseline) or "swin_unetr" (SOTA).
        in_channels: Number of input modalities (1 for CT scans).
        out_channels: Number of output classes (13 organs + 1 background).
        spatial_size: Required for Transformer-based models like SwinUNETR.
        
    Returns:
        torch.nn.Module: The uninitialized PyTorch model.
    """
    model_name = model_name.lower().strip()
    
    if model_name == "unet":
        # Standard 3D U-Net (Baseline)
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        
    elif model_name == "swin_unetr":
        # Swin UNETR (State-of-the-Art Transformer)
        model = SwinUNETR(
            img_size=spatial_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
        )
        
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose 'unet' or 'swin_unetr'.")
        
    return model