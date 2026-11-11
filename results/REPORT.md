# Rapport d'évaluation

Généré par `python scripts/report.py` depuis `results/evaluation_*.json` : ne pas modifier à la main.

Moyenne sur les volumes de validation (± écart-type entre folds en validation croisée). † : évaluation antérieure à la convention actuelle (organe manqué compté HD95 = diagonale du volume, NSD = 0) ; son HD95 n'est pas comparable.

## Validation croisée

| Résultat | Volumes | Checkpoint | Post-traitement | Inférence | Dice | HD95 (mm) | NSD | Organes manqués |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `cv_classes_w5_aug_pp_best` | 30 | best | oui | gaussian 75 % | 0.719 ± 0.022 | 19.3 ± 4.6 | 0.630 ± 0.021 | 1 |
| `cv_classes_w5_aug_pp_last` | 30 | last | oui | gaussian 75 % | 0.715 ± 0.025 | 19.2 ± 3.3 | 0.629 ± 0.024 | 0 |
| `cv_classes_w5_pp10_gauss_ov75` | 30 | best | oui | gaussian 75 % | 0.702 ± 0.027 | 33.6 ± 14.4 | 0.610 ± 0.029 | 0 |
| `cv_classes_w5_pp_gauss_ov75` | 30 | best | oui | gaussian 75 % | 0.700 ± 0.030 | 21.8 ± 6.4 | 0.610 ± 0.030 | 0 |
| `cv_classes_w5_pp_gauss` | 30 | best | oui | gaussian 50 % | 0.692 ± 0.025 | 22.4 ± 5.5 | 0.598 ± 0.027 | 0 |
| `cv_classes_w5_pp_best` | 30 | best | oui | — | 0.666 ± 0.027 | 23.7 ± 5.6 | 0.565 ± 0.025 | 0 |
| `cv_classes_pp_best` | 30 | best | oui | — | 0.567 ± 0.018 | 133.2 ± 8.9 | 0.455 ± 0.010 | 56 |
| `cv_classes_pp` | 30 | — | oui | — | 0.567 ± 0.018 | 47.5 ± 32.4 † | — | 56 |
| `cv_classes` | 30 | — | non | — | 0.564 ± 0.016 | 111.3 ± 52.5 † | — | 56 |
| `cv_classes_w5_pp_gauss_ov75_tta` | 30 | best | oui | gaussian 75 % + TTA | 0.085 ± 0.014 | 378.3 ± 18.9 | 0.040 ± 0.008 | 164 |

## Un seul fold ou hold-out

| Résultat | Volumes | Checkpoint | Post-traitement | Inférence | Dice | HD95 (mm) | NSD | Organes manqués |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `cv_classes_w5_aug_panc_fold1_last` | 6 | last | oui | gaussian 75 % | 0.687 | 29.1 | 0.595 | 0 |
| `cv_classes_w5_aug_panc_fold1_best` | 6 | best | oui | gaussian 75 % | 0.686 | 29.2 | 0.593 | 0 |
| `trialJ_lossw5_pp` | 6 | — | oui | — | 0.654 | 23.1 † | — | 0 |
| `trialJ_lossw5` | 6 | — | non | — | 0.641 | 122.2 † | — | 0 |
| `cv_classes_w5_aug_flip_fold1_best_tta` | 6 | best | oui | gaussian 75 % + TTA | 0.636 | 29.1 | 0.538 | 0 |
| `cv_classes_w5_aug_flip_fold1_best` | 6 | best | oui | gaussian 75 % | 0.626 | 31.8 | 0.522 | 0 |
| `trialH_classes_pp` | 6 | — | oui | — | 0.565 | 62.0 † | — | 11 |
| `trialH_classes` | 6 | — | non | — | 0.557 | 149.7 † | — | 11 |
| `trialG_250ep_cosine` | 6 | — | non | — | 0.443 | 45.0 † | — | 29 |
| `evaluation` | 6 | — | non | — | 0.439 | 61.0 † | — | 30 |
| `baseline_nsd` | 6 | — | non | — | 0.439 | 336.5 | 0.315 | 30 |

## Par organe : `cv_classes_w5_aug_pp_best` (meilleur Dice en validation croisée)

| Organe | Dice | HD95 (mm) | NSD |
| :--- | ---: | ---: | ---: |
| 1 spleen | 0.861 | 13.4 | 0.708 |
| 2 right kidney | 0.844 | 10.7 | 0.740 |
| 3 left kidney | 0.847 | 12.3 | 0.745 |
| 4 gallbladder | 0.562 | 17.2 | 0.452 |
| 5 esophagus | 0.670 | 12.1 | 0.634 |
| 6 liver | 0.942 | 9.1 | 0.732 |
| 7 stomach | 0.730 | 23.1 | 0.443 |
| 8 aorta | 0.836 | 30.6 | 0.783 |
| 9 inferior vena cava | 0.756 | 13.4 | 0.607 |
| 10 portal/splenic vein | 0.628 | 40.8 | 0.630 |
| 11 pancreas | 0.610 | 23.8 | 0.455 |
| 12 right adrenal | 0.530 | 8.1 | 0.644 |
| 13 left adrenal | 0.527 | 35.7 | 0.609 |
