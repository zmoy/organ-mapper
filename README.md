# 🫀 3D Organ Mapper : Segmentation Multi-Organes par Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-blue?style=for-the-badge)

## 📌 Contexte Clinique
La segmentation automatique des organes sur des scanners 3D (CT scans) est un enjeu majeur en imagerie médicale. Elle permet de :
* **Planifier la radiothérapie** en protégeant les organes sains (Organ-At-Risk).
* **Préparer les interventions chirurgicales** en offrant une cartographie 3D précise au chirurgien.
* **Accélérer le flux de travail** des radiologues en automatisant le détourage manuel.

Ce projet vise à développer un pipeline MLOps complet capable de segmenter simultanément **14 organes de l'abdomen** à partir de volumes CT bruts, en comparant une approche classique (U-Net) avec l'état de l'art des Vision Transformers (Swin UNETR).

---

## ⚙️ Pipeline de Données (MONAI)
Contrairement aux images 2D classiques (RGB), l'imagerie médicale 3D nécessite un pré-traitement rigoureux respectant la physique de l'acquisition. Ce projet utilise la librairie **MONAI** pour garantir l'intégrité clinique des données :

* **`Orientationd (axcodes="RAS")`** : Standardisation de l'orientation anatomique (Right, Anterior, Superior) pour tous les patients.
* **`Spacingd (1.5, 1.5, 2.0 mm)`** : Rééchantillonnage isométrique pour que le modèle apprenne la "vraie" taille physique des organes.
* **`ScaleIntensityRanged`** : Application d'un *Windowing* basé sur les Unités Hounsfield (HU) de l'abdomen, borné entre -175 (air) et 250 (os).
* **`RandCropByLabelClassesd`** : Extraction dynamique de patchs 3D **équilibrés par classe** (les petits organes sont tirés plus souvent que les gros) pour gérer le déséquilibre anatomique et optimiser la VRAM. L'option `--sampling posneg` revient à l'échantillonnage organe / fond (`RandCropByPosNegLabeld`).
* **Cache disque (`PersistentDataset`)** : le prétraitement déterministe (chargement, orientation, spacing, fenêtrage) est calculé une seule fois (environ 60 fois plus rapide ensuite).

---

## 🛠️ Défis d'Ingénierie & MLOps (Contrainte : 4 Go VRAM)
L'entraînement de modèles 3D massifs est notoirement coûteux en mémoire. Pour faire tourner un Transformer State-of-the-Art de 62 millions de paramètres sur une configuration locale limitée à 4 Go de VRAM et une RAM système restreinte, plusieurs stratégies ont été implémentées :

1. **CPU Offloading & Métriques ciblées :** Déplacement des tenseurs de validation sur la RAM hôte et suppression du calcul de la *Validation Loss* géante au profit du *Dice Score* calculé via des tenseurs `one_hot` optimisés.
2. **Gradient Accumulation :** Réduction du `batch_size` effectif à 1 avec accumulation des gradients sur 8 itérations pour simuler un grand batch size sans exploser la VRAM.
3. **Patch-based Validation :** Abandon du *Sliding Window Inference* global en phase de validation au profit d'une évaluation par patchs pour prévenir les erreurs d'allocation système (`alloc_cpu.cpp OOM`).

4. **Entraînement reprenable :** Ctrl+C arrête proprement à la fin de l'epoch (`last.ckpt`), et `--resume` reprend là où l'entraînement s'est arrêté (les folds terminés sont sautés).
5. **Évaluation économe en RAM :** Dice et HD95 calculés organe par organe sur des masques `uint8`.

Les expériences peuvent être trackées sur **Weights & Biases** (ou en CSV avec `--logger csv`).

---

## 📊 Expériences & Résultats

Les modèles sont entraînés sur un jeu très restreint (24 images d'entraînement, 6 de validation par fold) pour observer leur comportement en régime "Low-Data". Toutes les métriques sont calculées sur **volumes entiers** (sliding window, recouvrement 50 %) ; le **HD95 est en millimètres** (spacing 1.5 × 1.5 × 2.0 mm) et ne couvre que les organes prédits au moins une fois.

### U-Net : de la baseline à la validation croisée

| Modèle | Protocole | Dice | HD95 (mm) |
| :--- | :--- | :--- | :--- |
| U-Net baseline (4.8 M) | hold-out, 6 volumes | 0.439 | 61.0 (8 organes sur 13 prédits) |
| U-Net, DiceCE + patchs par classe + cosinus, 250 epochs | **CV 5 folds** | **0.564 ± 0.016** | 111.3 ± 52.5 |
| idem + post-traitement | CV 5 folds | **0.567 ± 0.018** | **47.5 ± 32.4** |
| idem + loss pondérée sur les petits organes + post-traitement | fold 1 uniquement (CV à venir) | **0.654** | **23.1** |

* **Post-traitement :** conservation de la plus grande composante connectée de chaque organe, ce qui supprime les fausses détections lointaines et divise le HD95 par plus de deux, sans dégrader le Dice.
* **Loss pondérée (`--rare-loss-weight 5`) :** elle permet de détecter les organes 12 et 13, jamais détectés sans (Dice 0 sur les 5 folds), avec des Dice de 0.45 et 0.53 sur le fold 1. Ce gain n'est pas encore confirmé en validation croisée.
* **Ce qui a vraiment fait la différence :** l'échantillonnage de patchs par classe (+16 points de Dice sur le fold 1) et une durée d'entraînement suffisante (250 epochs, la courbe de Dice reste plate pendant ~60 epochs avant de décoller).

### Comparaison avec Swin UNETR

| Modèle | Paramètres | Architecture | Dice Score (Val) |
| :--- | :--- | :--- | :--- |
| **U-Net (baseline)** | 4.8 M | Convolutions (CNN) | **~ 44 %** |
| **Swin UNETR (SOTA)** | 62.2 M | Vision Transformer | ~ 8 % (Plafond à 20%) |

L'expérience prouve une règle fondamentale du Deep Learning : **le biais inductif face au besoin de données**.
Le **U-Net**, grâce à la nature de ses convolutions, comprend nativement la contiguïté spatiale (les pixels voisins forment des bords). Il parvient à extraire des caractéristiques pertinentes même avec 24 images.
À l'inverse, le **Swin UNETR**, dépourvu de ce biais inductif, utilise des mécanismes d'Attention qui nécessitent des milliers d'exemples pour "comprendre" l'image. En l'absence de *Pre-training* massif, ses 62 millions de paramètres ont sévèrement sur-appris (*overfitting*) sur le set d'entraînement. *(Ces essais Swin UNETR datent de la baseline et n'ont pas été refaits avec les améliorations ci-dessus.)*

---

## 👁️ Visualisation des Inférences
Le script d'inférence intègre la transformation `Invertd` de MONAI pour replacer les prédictions 3D exactement dans l'espace physique original du patient (512x512).

### Surcouche Anatomique (Coupe Axiale)
> *À gauche : Le scanner CT brut. À droite : La prédiction 3D du U-Net Baseline superposée.*

![Visualisation Inférence](./assets/matplotlib.png)

### Rendu 3D (ITK-SNAP)
> *Modélisation 3D des organes segmentés par l'IA sur un set de test aveugle.*

![Visualisation 3D](./assets/itk-snap.png)

---

## 🚀 Utilisation & Reproductibilité

**1. Installation des dépendances :**

```bash
pip install -r requirements.txt
```

**2. Lancement de l'entraînement :**
Tous les réglages (modèle, loss, patchs, scheduler...) se passent en ligne de commande (`python scripts/train.py --help`). Configuration de référence, en validation croisée 5 folds :

```bash
python scripts/train_cv.py --run-name cv_classes --logger csv --max-epochs 250   --lr-scheduler cosine --warmup-epochs 5 --loss dicece --num-samples 4 --batch-size 2   --sampling classes --rare-weight 3 --rare-loss-weight 5
```

Ctrl+C arrête proprement à la fin de l'epoch ; la même commande avec `--resume` reprend l'entraînement.

**3. Évaluation en volume entier (Dice + HD95 en mm) :**

```bash
python scripts/evaluate.py --cv --checkpoints-dir checkpoints/cv_classes --postprocess
```

**4. Lancement de l'inférence :**
Génère les volumes .nii.gz prédits (par défaut sur toutes les images de `imagesTs`) à partir du checkpoint de votre choix.

```bash
python scripts/predict.py --checkpoint checkpoints/cv_classes/fold_1/<checkpoint>.ckpt --postprocess
```

## Contraintes Actuelles & Perspectives d'Amélioration

Conscient des limites matérielles de mon environnement local (4 Go de VRAM), j'ai implémenté l'architecture et les pipelines MLOps sur ma machine. Les pistes restantes :

* **Confirmer la loss pondérée en validation croisée** (aujourd'hui testée sur un seul fold).
* **Organes encore difficiles :** l'organe 4 (Dice ~0.3 sur le fold 1) et l'organe 10 (~0.55).
* **Augmentations de données** (déformation élastique, déjà écrite dans `transforms.py`) et patchs plus grands ou spacing plus fin.
* **Ré-entraînement cloud** (ex : AWS EC2 avec GPU A10G) pour des modèles plus gros, et pré-entraînement du Swin UNETR.
