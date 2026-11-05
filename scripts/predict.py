"""
Inference script to generate 3D segmentation masks on test images using a trained model.
Automatically inverts spatial transforms to match the original patient anatomy.

The inference is the one of scripts/evaluate.py (src/inference.py: sliding window,
AMP, options --sw-overlap, --sw-mode, --tta; optional post-processing).

Usage:
    python scripts/predict.py                                  # baseline U-Net, all test images
    python scripts/predict.py --checkpoint checkpoints/cv_classes/fold_1/<name>.ckpt --postprocess
    python scripts/predict.py --model swin_unetr --checkpoint checkpoints/swin_trained_best/swin_trained.ckpt
    python scripts/predict.py --image <file or name in imagesTs>
"""

import argparse
import sys
from pathlib import Path

import torch

# Importations spécifiques MONAI
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, AsDiscreted, Invertd, SaveImaged, MapTransform
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference import add_inference_args, inference_settings, predict_volume
from src.models.builder import get_model
from src.postprocess import keep_largest_components

SPATIAL_SIZE = (96, 96, 96)
NUM_CLASSES = 14


class KeepLargestComponentsd(MapTransform):
    """Dictionary transform applying src.postprocess.keep_largest_components to a label map (1, X, Y, Z)."""

    def __init__(self, keys, num_classes: int = NUM_CLASSES):
        super().__init__(keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label_map = d[key][0].cpu().numpy().astype("uint8")
            cleaned = keep_largest_components(label_map, self.num_classes)
            d[key][0] = torch.from_numpy(cleaned).to(d[key].dtype)  # in place: keeps the MetaTensor metadata
        return d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment the test CT volumes and save .nii.gz masks.")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "unet_baseline_best" / "unet_baseline.ckpt",
        help="Model checkpoint (default: the versioned U-Net baseline).",
    )
    parser.add_argument("--model", default="unet", choices=["unet", "swin_unetr"],
                        help="Architecture of the checkpoint.")
    parser.add_argument("--image", type=Path, default=None,
                        help="One image (path, or file name inside imagesTs). Default: all of imagesTs.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images.")
    parser.add_argument("--postprocess", action="store_true",
                        help="Keep only the largest connected component of each organ.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "predictions")
    add_inference_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. Configuration ---
    test_images_dir = PROJECT_ROOT / "data" / "raw" / "imagesTs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # --- 2. Chargement du Modèle ---
    print(f"Loading {args.model} checkpoint from: {args.checkpoint}")
    model = get_model(model_name=args.model, spatial_size=SPATIAL_SIZE)
    # Only the network weights: the checkpoint may also hold loss buffers (class weights)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    model = model.to(device)
    model.eval()

    # --- 3. Pipelines MONAI (Le secret est ici) ---

    # A. Pré-traitement (Aller vers l'espace du modèle). Must match src/data/transforms.py exactly.
    inference_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True)
    ])

    # B. Post-traitement (Retourner vers l'espace original du patient)
    post_steps = [
        EnsureTyped(keys="pred"),
        AsDiscreted(keys="pred", argmax=True),
    ]
    if args.postprocess:
        # In the model space, before going back to the patient space (same rule as evaluate.py)
        post_steps.append(KeepLargestComponentsd(keys="pred"))
    post_transforms = Compose(post_steps + [
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
            output_dir=args.output_dir,
            output_postfix="preds",
            separate_folder=False,
            resample=False
        )
    ])

    # --- 4. Boucle d'Inférence ---
    if args.image is not None:
        image_path = args.image if args.image.exists() else test_images_dir / args.image
        test_files = [image_path]
    else:
        test_files = sorted(test_images_dir.glob("*.nii.gz"))
    if args.limit is not None:
        test_files = test_files[:args.limit]
    if not test_files or not all(f.exists() for f in test_files):
        print(f"Erreur : aucune image trouvée ({args.image or test_images_dir}).")
        return

    for n, image_path in enumerate(test_files, start=1):
        print(f"\n[{n}/{len(test_files)}] Processing image: {image_path.name}")

        # Création d'un mini DataLoader pour utiliser la mécanique MONAI
        test_ds = Dataset(data=[{"image": str(image_path)}], transform=inference_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        with torch.no_grad():
            for batch_data in test_loader:
                inputs = batch_data["image"].to(device)

                print("Running sliding window inference...")
                # Patches on GPU, stitched probabilities on CPU (4 GB of VRAM), see src/inference.py
                batch_data["pred"] = predict_volume(model, inputs, SPATIAL_SIZE, device, **inference_settings(args))

                # Decollate sépare le batch pour appliquer le post-traitement image par image
                for i in decollate_batch(batch_data):
                    post_transforms(i)

    print(f"\nSuccess! Les prédictions ont été redimensionnées et sauvegardées dans {args.output_dir}")


if __name__ == "__main__":
    main()
