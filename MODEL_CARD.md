# Model card: Organ Mapper U-Net (`cv_classes_w5_aug`)

3D U-Net that segments **13 abdominal organs** in contrast-enhanced CT scans. Five models are published, one per cross-validation fold. They were trained on a 4 GB GPU.

> **Not for clinical use.** This is a personal research and engineering project. The models have not been validated for diagnosis, treatment planning or any medical decision.

## Files

| File | Content |
| :--- | :--- |
| `organ_mapper_unet_fold_1.pt` … `organ_mapper_unet_fold_5.pt` | Network weights only (PyTorch `state_dict`, 19.3 MB each). Load them with `torch.load(..., weights_only=True)`, which runs no pickled code. |
| `manifest.json` | Architecture, classes, pre-processing, inference settings, training configuration, cross-validation scores, validation volumes and SHA-256 of each file. |

The files are produced by `scripts/export_models.py` from the training checkpoints (`src/export.py`). Each export checks that the checkpoints are the ones the evaluation scored.

## Usage

Put the files in `exports/cv_classes_w5_aug/`, then use one fold model or the ensemble of the five:

```bash
python scripts/predict.py --checkpoint exports/cv_classes_w5_aug/organ_mapper_unet_fold_2.pt --postprocess --image <ct.nii.gz>
python scripts/predict.py --ensemble-dir exports/cv_classes_w5_aug --postprocess
```

The input is a CT volume in NIfTI format (`.nii.gz`), in Hounsfield units. The output is a label map with values 0 to 13, written in the space of the input image.

To use the weights without this repository, build `monai.networks.nets.UNet(spatial_dims=3, in_channels=1, out_channels=14, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)`, load the `state_dict`, and reproduce the pre-processing below exactly.

## Model

- **Architecture:** MONAI `UNet` with residual units (2 per level), channels 16 → 256, 4.82 M parameters, 14 output channels (background and 13 organs).
- **Pre-processing:**
  - orientation RAS;
  - resampling to 1.5 × 1.5 × 2.0 mm (bilinear);
  - HU window −175 to 250, mapped to [0, 1].

  It is defined once in `src/data/transforms.py` (`get_inference_transforms`).
- **Inference:**
  - sliding window of 96³ patches with Gaussian weighting and 75 % overlap, then argmax;
  - the largest connected component of each organ is kept;
  - no test-time augmentation: these models were trained without flips, and flipped predictions are wrong.

| Label | Organ | Label | Organ |
| :--- | :--- | :--- | :--- |
| 1 | spleen | 8 | aorta |
| 2 | right kidney | 9 | inferior vena cava |
| 3 | left kidney | 10 | portal and splenic vein |
| 4 | gallbladder | 11 | pancreas |
| 5 | esophagus | 12 | right adrenal gland |
| 6 | liver | 13 | left adrenal gland |
| 7 | stomach | | |

Left and right follow the patient: label 2 is the patient's right kidney.

## Training data

The data are the 30 annotated abdominal CT volumes of the BTCV dataset (MICCAI 2015 Multi-Atlas Labeling Beyond the Cranial Vault, abdomen). They are split into 5 folds with `KFold(n_splits=5, shuffle=True, random_state=42)`, so each model is trained on 24 volumes and validated on the other 6. The validation volumes of each fold are listed in `manifest.json`.

The dataset is not redistributed here. It is available from its original source, under its own terms of use.

## Training

- **Patches:** 96³, 2 volumes × 4 patches per step. Patch centres are sampled per class, with gallbladder, esophagus, portal vein and adrenal glands drawn 3 times more often.
- **Loss and optimizer:** Dice + cross-entropy loss, weighted ×5 on those small organs. AdamW (learning rate 1e-4, weight decay 1e-5), cosine schedule with 5 warm-up epochs, 250 epochs, mixed precision.
- **Augmentations:** rotations ±15°, scaling ±15 %, noise, blur, brightness and gamma contrast. No flips.
- **Checkpoint:** the epoch with the best validation Dice of each fold.

## Evaluation

The scores come from 5-fold cross-validation. Each fold model is scored on its own 6 validation volumes, on whole volumes, and the results are averaged over the 13 organs. A missed organ counts as HD95 = volume diagonal and NSD = 0.

| Metric | Mean ± std over folds |
| :--- | :--- |
| Dice | 0.719 ± 0.022 |
| HD95 | 19.3 ± 4.6 mm |
| NSD (2 mm) | 0.630 ± 0.021 |

| Organ | Dice | HD95 (mm) | NSD |
| :--- | :--- | :--- | :--- |
| spleen | 0.861 | 13.4 | 0.708 |
| right kidney | 0.844 | 10.7 | 0.740 |
| left kidney | 0.847 | 12.3 | 0.745 |
| gallbladder | 0.562 | 17.2 | 0.452 |
| esophagus | 0.670 | 12.1 | 0.634 |
| liver | 0.942 | 9.1 | 0.732 |
| stomach | 0.730 | 23.1 | 0.443 |
| aorta | 0.836 | 30.6 | 0.783 |
| inferior vena cava | 0.756 | 13.4 | 0.607 |
| portal/splenic vein | 0.628 | 40.8 | 0.630 |
| pancreas | 0.610 | 23.8 | 0.455 |
| right adrenal | 0.530 | 8.1 | 0.644 |
| left adrenal | 0.527 | 35.7 | 0.609 |

The ensemble of the five models cannot be scored this way: every volume was in the training set of four of the five models.

## Limitations

- **Small and homogeneous data:** 30 volumes from a single dataset. There is no external test set, so performance on other scanners, protocols, phases or populations is unknown and likely lower.
- **Optimistic scores:** the settings were tuned on fold 1, and the checkpoints were selected on validation Dice. The last-epoch checkpoints score 0.715 Dice, so this bias is about 0.4 points.
- **Small organs:** gallbladder, adrenal glands, pancreas and portal vein stay between 0.53 and 0.63 Dice.
- **Ascending aorta:** in the scans that extend into the thorax (10 of 30), the ascending aorta is annotated at the top of the volume, but the model never predicts it. This drives the mean aorta HD95 (median 7.8 mm, mean 30.6 mm).
- **Atypical anatomy:** an enlarged spleen, displaced organs or a gas-filled stomach can be badly segmented (for example, volume img0009).
- **Comparison with the state of the art:** published nnU-Net results on comparable abdominal benchmarks are around 0.83 Dice, under different protocols. The gap has not been measured on these folds.

## Environment

PyTorch 2.5.1, MONAI 1.5.2, PyTorch Lightning 2.6. The full list is in `requirements.txt`.
