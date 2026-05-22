"""
Inference script to generate 3D segmentation masks on test images using the trained baseline model.
Automatically inverts spatial transforms to match the original patient anatomy.
"""

import os
import torch
from pathlib import Path
import sys

# Importations spécifiques MONAI
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, AsDiscreted, Invertd, SaveImaged
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.builder import get_model
from src.training.lightning_module import OrganMapperModule

def main():
    # --- 1. Configuration ---
    test_images_dir = PROJECT_ROOT / "data" / "raw" / "imagesTs"
    output_dir = PROJECT_ROOT / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "unet_baseline_best" / "unet_baseline.ckpt" 
    
    spatial_size = (96, 96, 96)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # --- 2. Chargement du Modèle ---
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = get_model(model_name="unet", spatial_size=spatial_size)
    
    pl_module = OrganMapperModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        loss_function=torch.nn.Identity(), 
        optimizer_class=torch.optim.SGD, 
        optimizer_kwargs={}
    )
    
    model = pl_module.model.to(device)
    model.eval()

    # --- 3. Pipelines MONAI (Le secret est ici) ---
    
    # A. Pré-traitement (Aller vers l'espace du modèle : 234x234x206)
    inference_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True)
    ])

    # B. Post-traitement (Retourner vers l'espace original du patient : 512x512x138)
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        AsDiscreted(keys="pred", argmax=True),
        # L'inversion spatiale magique !
        Invertd(
            keys="pred",
            transform=inference_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True, # Indispensable pour un masque (pas de valeurs décimales)
            to_tensor=True,
        ),
        # Sauvegarde automatique par MONAI
        SaveImaged(
            keys="pred", 
            meta_keys="pred_meta_dict", 
            output_dir=output_dir, 
            output_postfix="pred", 
            separate_folder=False, 
            resample=False
        )
    ])

    # --- 4. Boucle d'Inférence ---
    test_files = sorted(list(test_images_dir.glob("*.nii.gz")))
    if not test_files:
        print("Erreur : Aucune image trouvée dans le dossier imagesTs.")
        return
        
    image_path = test_files[1]
    print(f"\nProcessing image: {image_path.name}")
    
    # Création d'un mini DataLoader pour utiliser la mécanique MONAI
    test_ds = Dataset(data=[{"image": str(image_path)}], transform=inference_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            
            print("Running sliding window inference...")
            outputs = sliding_window_inference(inputs, spatial_size, 4, model, overlap=0.5)
            
            # On stocke les probabilités prédites dans notre dictionnaire
            batch_data["pred"] = outputs
            
            # Decollate sépare le batch pour appliquer le post-traitement image par image
            for i in decollate_batch(batch_data):
                post_transforms(i)
                
    print(f"\nSuccess! La prédiction a été redimensionnée et sauvegardée dans {output_dir}")

if __name__ == "__main__":
    main()