"""
Fast unit tests (no real data, no GPU, a few seconds):
    - post-processing (largest connected component per organ),
    - deterministic K-fold split without leakage,
    - class-balanced patch sampling (on small synthetic NIfTI volumes),
    - loss construction with class weights,
    - checkpoint selection of the evaluation (best vs last),
    - full-volume inference (default settings, mirroring test-time augmentation),
    - last.ckpt written at every epoch during training (not only with the best).

Run with:
    pytest tests/test_core.py
"""

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_kfold_splits
from src.data.transforms import NUM_CLASSES, RARE_CLASSES, get_class_ratios, get_train_transforms
from monai.inferers import sliding_window_inference

from src.inference import MIRROR_AXES, predict_volume
from src.postprocess import keep_largest_components
from src.training.factory import build_checkpoint_callbacks, build_loss


def _load_script(name: str):
    """Imports scripts/<name>.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Post-processing ---

def test_keep_largest_components_removes_islands():
    label_map = np.zeros((20, 20, 20), dtype=np.uint8)
    label_map[2:8, 2:8, 2:8] = 1        # organ 1: main blob (216 voxels)
    label_map[15:17, 15:17, 15:17] = 1  # organ 1: far-away island -> removed
    label_map[10:14, 10:14, 10:14] = 2  # organ 2: single blob -> kept

    cleaned = keep_largest_components(label_map.copy(), NUM_CLASSES)

    assert (cleaned == 1).sum() == 216
    assert (cleaned == 2).sum() == 64
    assert (cleaned[15:17, 15:17, 15:17] == 0).all()


def test_keep_largest_components_keeps_single_components_untouched():
    label_map = np.zeros((10, 10, 10), dtype=np.uint8)
    label_map[1:4, 1:4, 1:4] = 3
    label_map[6:9, 6:9, 6:9] = 7
    assert np.array_equal(keep_largest_components(label_map.copy(), NUM_CLASSES), label_map)


# --- K-fold split ---

def _fake_dicts(n: int = 30) -> list[dict]:
    return [{"image": f"img_{i:02d}.nii.gz", "label": f"lbl_{i:02d}.nii.gz"} for i in range(n)]


def test_kfold_splits_are_deterministic():
    names = lambda folds: [[d["image"] for d in val] for _, val in folds]  # noqa: E731
    assert names(get_kfold_splits(_fake_dicts())) == names(get_kfold_splits(_fake_dicts()))


def test_kfold_splits_have_no_leakage_and_cover_everything():
    data = _fake_dicts()
    folds = get_kfold_splits(data, n_splits=5)
    assert len(folds) == 5

    all_val = []
    for train, val in folds:
        train_names = {d["image"] for d in train}
        val_names = {d["image"] for d in val}
        assert not train_names & val_names            # no volume in both train and validation
        assert len(train) + len(val) == len(data)
        all_val += sorted(val_names)
    assert sorted(all_val) == sorted(d["image"] for d in data)  # each volume validated exactly once


# --- Class-balanced sampling ---

def test_class_ratios():
    ratios = get_class_ratios(rare_weight=3.0, background_weight=2.0)
    assert len(ratios) == NUM_CLASSES
    assert ratios[0] == 2.0
    for c in range(1, NUM_CLASSES):
        assert ratios[c] == (3.0 if c in RARE_CLASSES else 1.0)


@pytest.fixture(scope="module")
def synthetic_volume(tmp_path_factory) -> dict:
    """A 64x64x48 CT-like volume, already at 1.5x1.5x2.0 mm, with a big organ and a tiny rare one."""
    folder = tmp_path_factory.mktemp("synthetic")
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    image = np.random.default_rng(0).uniform(-200, 300, size=(64, 64, 48)).astype(np.float32)
    label = np.zeros((64, 64, 48), dtype=np.uint8)
    label[5:40, 5:40, 5:30] = 6      # big organ
    label[55:58, 55:58, 40:43] = 12  # tiny rare organ, far from the big one
    nib.save(nib.Nifti1Image(image, affine), folder / "img.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), folder / "lbl.nii.gz")
    return {"image": str(folder / "img.nii.gz"), "label": str(folder / "lbl.nii.gz")}


def _rare_patch_fraction(volume: dict, sampling: str, draws: int = 15) -> float:
    transforms = get_train_transforms(spatial_size=(16, 16, 16), num_samples=4, sampling=sampling)
    transforms.set_random_state(seed=0)
    patches = [p for _ in range(draws) for p in transforms(volume)]
    return float(np.mean([(p["label"] == 12).any().item() for p in patches]))


def test_class_sampling_outputs_batchable_patches(synthetic_volume):
    transforms = get_train_transforms(spatial_size=(16, 16, 16), num_samples=4, sampling="classes")
    patches = transforms(synthetic_volume)
    assert len(patches) == 4
    for patch in patches:
        assert tuple(patch["image"].shape) == (1, 16, 16, 16)
        assert tuple(patch["label"].shape) == (1, 16, 16, 16)
        assert "label_cls_indices" not in patch  # variable-length indices would break batching


def test_class_sampling_draws_rare_organs_more_often(synthetic_volume):
    posneg = _rare_patch_fraction(synthetic_volume, "posneg")
    classes = _rare_patch_fraction(synthetic_volume, "classes")
    assert classes > posneg
    assert classes > 0.2  # rare organ: weight 3 out of ~8 for the classes present here


def test_unknown_sampling_raises():
    with pytest.raises(ValueError):
        get_train_transforms(sampling="unknown")


# --- Loss ---

def test_build_loss_without_weights():
    loss = build_loss("dicece")
    assert loss.dice.class_weight is None


def test_build_loss_with_rare_class_weights():
    loss = build_loss("dicece", rare_loss_weight=5.0)
    expected = [5.0 if c in RARE_CLASSES else 1.0 for c in range(NUM_CLASSES)]
    # Cross-entropy: one weight per class, background included
    assert loss.cross_entropy.weight.tolist() == expected
    # Dice: MONAI drops the background weight (include_background=False), index 0 = organ 1
    assert loss.dice.class_weight.tolist() == expected[1:]

    logits = torch.randn(2, NUM_CLASSES, 8, 8, 8)
    target = torch.randint(0, NUM_CLASSES, (2, 1, 8, 8, 8))
    assert torch.isfinite(loss(logits, target))


def test_build_loss_unknown_name_raises():
    with pytest.raises(ValueError):
        build_loss("unknown")


# --- Evaluation: checkpoint selection ---

def test_find_fold_checkpoint_best_and_last(tmp_path):
    evaluate = _load_script("evaluate")
    fold_dir = tmp_path / "fold_1"
    fold_dir.mkdir()
    (fold_dir / "run_fold_1-epoch10-dice0.50.ckpt").write_bytes(b"")
    (fold_dir / "last.ckpt").write_bytes(b"")  # written last: must not be taken as "best"

    assert evaluate.find_fold_checkpoint(tmp_path, 1).name == "run_fold_1-epoch10-dice0.50.ckpt"
    assert evaluate.find_fold_checkpoint(tmp_path, 1, kind="last").name == "last.ckpt"
    with pytest.raises(FileNotFoundError):
        evaluate.find_fold_checkpoint(tmp_path, 2)


# --- Inference: sliding window and test-time augmentation ---

def test_mirror_axes_cover_the_8_flip_combinations():
    assert len(MIRROR_AXES) == 8 and len(set(MIRROR_AXES)) == 8
    assert MIRROR_AXES[0] == () and (2, 3, 4) in MIRROR_AXES


def test_default_inference_matches_plain_sliding_window():
    torch.manual_seed(0)
    model = torch.nn.Conv3d(1, 3, kernel_size=3, padding=1).eval()
    image = torch.randn(1, 1, 20, 18, 16)
    with torch.no_grad():
        expected = torch.softmax(sliding_window_inference(image, (8, 8, 8), 1, model, overlap=0.5), dim=1)
    probs = predict_volume(model, image, (8, 8, 8), torch.device("cpu"))
    assert probs.shape == (1, 3, 20, 18, 16)
    assert torch.allclose(probs, expected, atol=1e-5)


def test_tta_is_neutral_for_a_flip_invariant_model():
    # A 1x1x1 convolution gives the same answer whatever the orientation of the input:
    # averaging over the mirrored inputs must give back the plain prediction.
    torch.manual_seed(0)
    model = torch.nn.Conv3d(1, 3, kernel_size=1).eval()
    image = torch.randn(1, 1, 20, 18, 16)
    plain = predict_volume(model, image, (8, 8, 8), torch.device("cpu"))
    tta = predict_volume(model, image, (8, 8, 8), torch.device("cpu"), tta=True, mode="gaussian")
    assert torch.allclose(plain, tta, atol=1e-5)
    assert torch.allclose(tta.sum(dim=1), torch.ones(1, 20, 18, 16), atol=1e-5)


def test_tta_flips_back_the_predictions():
    # A model that is NOT flip-invariant (its output depends on the position along X):
    # each flipped prediction must be flipped back before averaging.
    class XRamp(torch.nn.Module):
        def forward(self, x):
            ramp = torch.linspace(-3, 3, x.shape[2]).view(1, 1, -1, 1, 1)
            return torch.cat([ramp.expand_as(x), -ramp.expand_as(x)], dim=1)

    image = torch.zeros(1, 1, 16, 8, 8)
    plain = predict_volume(XRamp(), image, (16, 8, 8), torch.device("cpu"))
    tta = predict_volume(XRamp(), image, (16, 8, 8), torch.device("cpu"), tta=True)
    # Half of the mirrors flip X: the average is symmetric, i.e. 0.5 everywhere
    assert torch.allclose(tta, torch.full_like(tta, 0.5), atol=1e-5)
    assert not torch.allclose(plain, tta)


def test_unknown_sliding_window_mode_raises():
    with pytest.raises(ValueError):
        predict_volume(torch.nn.Identity(), torch.zeros(1, 1, 8, 8, 8), (8, 8, 8), torch.device("cpu"), mode="x")


# --- Training: checkpoints written during fit ---

class _DecliningDiceModule(pl.LightningModule):
    """Tiny model whose val/dice is best at epoch 0 and then decreases."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/dice", 1.0 / (self.current_epoch + 1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_last_checkpoint_follows_the_last_epoch_not_the_best(tmp_path):
    loader = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
    trainer = pl.Trainer(
        max_epochs=3, callbacks=build_checkpoint_callbacks("run", tmp_path), logger=False,
        accelerator="cpu", enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(_DecliningDiceModule(), train_dataloaders=loader, val_dataloaders=loader)

    best = [p for p in tmp_path.glob("*.ckpt") if p.name != "last.ckpt"]
    assert [p.name for p in best] == ["run-epoch00-dice1.00.ckpt"]
    assert torch.load(tmp_path / "last.ckpt", weights_only=False)["epoch"] == 2
