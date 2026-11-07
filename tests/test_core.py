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

import argparse
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
from src.data.transforms import (
    NUM_CLASSES, RARE_CLASSES, RandFlipLateralSwapd, get_augmentations, get_class_ratios, get_flips,
    get_train_transforms, swap_lateral_labels,
)
from monai.inferers import sliding_window_inference

from src.checkpoints import find_fold_checkpoints, load_model
from src.inference import MIRROR_AXES, predict_volume
from src.models.builder import get_model
from src.postprocess import VOXEL_SPACING as POSTPROCESS_SPACING
from src.postprocess import keep_largest_components
import src.training.callbacks as callbacks_module
from src.training.callbacks import DiskSpaceGuard, GracefulStop, LowDiskSpaceError, free_space_gb
from src.training.config import TrainConfig, add_train_args
from src.training.factory import build_checkpoint_callbacks, build_loss, fit_with_resume


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


def test_min_ratio_keeps_large_pieces_and_drops_small_islands():
    label_map = np.zeros((30, 30, 30), dtype=np.uint8)
    label_map[1:11, 1:11, 1:11] = 1      # organ 1: largest piece (1000 voxels)
    label_map[15:21, 15:21, 15:21] = 1   # second piece, 216 voxels = 21.6 % -> kept with 0.1
    label_map[25:27, 25:27, 25:27] = 1   # island, 8 voxels = 0.8 % -> removed

    only_largest = keep_largest_components(label_map.copy(), NUM_CLASSES)
    relaxed = keep_largest_components(label_map.copy(), NUM_CLASSES, min_ratio=0.1)

    assert (only_largest == 1).sum() == 1000
    assert (relaxed == 1).sum() == 1000 + 216
    assert (relaxed[25:27, 25:27, 25:27] == 0).all()


def _organ_in_three_pieces() -> np.ndarray:
    """Organ 1: largest piece x 1..10, a close piece 3 voxels away along x, a far island."""
    label_map = np.zeros((40, 20, 20), dtype=np.uint8)
    label_map[1:11, 1:11, 1:11] = 1      # largest piece (1000 voxels), last x index 10
    label_map[13:15, 4:6, 4:6] = 1       # 8 voxels, x 13: 3 voxels x 1.5 mm = 4.5 mm away
    label_map[30:32, 4:6, 4:6] = 1       # 8 voxels, x 30: 20 voxels x 1.5 mm = 30 mm away
    return label_map


def test_max_distance_keeps_close_pieces_and_drops_far_islands():
    cleaned = keep_largest_components(_organ_in_three_pieces(), NUM_CLASSES, max_distance_mm=5.0,
                                      spacing=(1.5, 1.5, 2.0))
    assert (cleaned == 1).sum() == 1000 + 8
    assert (cleaned[13:15, 4:6, 4:6] == 1).all()
    assert (cleaned[30:32] == 0).all()


def test_max_distance_below_the_gap_keeps_only_the_largest():
    cleaned = keep_largest_components(_organ_in_three_pieces(), NUM_CLASSES, max_distance_mm=4.0,
                                      spacing=(1.5, 1.5, 2.0))
    assert (cleaned == 1).sum() == 1000


def test_max_distance_uses_the_voxel_spacing():
    # Same gap of 3 voxels, but along z (2 mm per voxel): 6 mm
    label_map = np.zeros((20, 20, 40), dtype=np.uint8)
    label_map[1:11, 1:11, 1:11] = 1
    label_map[4:6, 4:6, 13:15] = 1
    kept = lambda d: (keep_largest_components(label_map.copy(), NUM_CLASSES, max_distance_mm=d,  # noqa: E731
                                              spacing=(1.5, 1.5, 2.0)) == 1).sum()
    assert kept(5.0) == 1000
    assert kept(7.0) == 1008


def test_postprocess_and_evaluation_share_the_voxel_spacing():
    assert POSTPROCESS_SPACING == _load_script("evaluate").VOXEL_SPACING


def test_prediction_cache_round_trip_and_refuses_other_settings(tmp_path):
    evaluate = _load_script("evaluate")
    label_map = np.random.default_rng(0).integers(0, NUM_CLASSES, (8, 8, 8)).astype(np.uint8)
    cache = evaluate.PredictionCache(tmp_path, Path("fold_1/best.ckpt"), {"mode": "gaussian", "overlap": 0.75})
    assert cache.load("img0001") is None
    cache.save("img0001", label_map)

    reopened = evaluate.PredictionCache(tmp_path, Path("fold_1/best.ckpt"), {"mode": "gaussian", "overlap": 0.75})
    assert np.array_equal(reopened.load("img0001"), label_map)
    with pytest.raises(ValueError):
        evaluate.PredictionCache(tmp_path, Path("fold_1/best.ckpt"), {"mode": "gaussian", "overlap": 0.5})
    with pytest.raises(ValueError):
        evaluate.PredictionCache(tmp_path, Path("fold_1/last.ckpt"), {"mode": "gaussian", "overlap": 0.75})


def test_to_onehot_inverts_argmax():
    evaluate = _load_script("evaluate")
    label_map = np.random.default_rng(0).integers(0, NUM_CLASSES, (6, 6, 6)).astype(np.uint8)
    onehot = evaluate.to_onehot(label_map)
    assert onehot.shape == (1, NUM_CLASSES, 6, 6, 6) and onehot.dtype == torch.uint8
    assert np.array_equal(onehot[0].argmax(dim=0).numpy(), label_map)


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


def test_strong_augmentation_keeps_integer_labels_and_patch_size(synthetic_volume):
    transforms = get_train_transforms(spatial_size=(16, 16, 16), num_samples=4, sampling="classes",
                                      augmentation="strong")
    transforms.set_random_state(seed=0)
    for _ in range(5):
        for patch in transforms(synthetic_volume):
            assert tuple(patch["image"].shape) == (1, 16, 16, 16)
            labels = patch["label"].unique()
            # Nearest-neighbour resampling: only the original label values, never interpolated ones
            assert set(labels.tolist()) <= {0.0, 6.0, 12.0}


def test_light_augmentation_is_the_original_pipeline():
    names = [type(t).__name__ for t in get_augmentations("light")]
    assert names == ["RandAffined", "RandShiftIntensityd", "RandGaussianNoised"]
    assert "RandShiftIntensityd" not in [type(t).__name__ for t in get_augmentations("light", intensity_shift=False)]


def test_no_augmentation_flips_the_patches():
    # Mirroring is a separate option (get_flips), which swaps the side-specific labels
    for level in ("light", "strong"):
        assert not any("Flip" in type(t).__name__ for t in get_augmentations(level))
    assert get_flips("none") == []


def test_swap_lateral_labels():
    label = torch.tensor([0, 1, 2, 3, 6, 12, 13])
    assert swap_lateral_labels(label).tolist() == [0, 1, 3, 2, 6, 13, 12]


def _lateral_patch() -> dict:
    """A 1x8x6x4 patch: right kidney (2) at high x, left kidney (3) at low x, liver (6) at high y."""
    label = torch.zeros(1, 8, 6, 4)
    label[0, 6:, :, :] = 2
    label[0, :2, :, :] = 3
    label[0, 3:5, 5, :] = 6
    image = torch.arange(8 * 6 * 4, dtype=torch.float32).view(1, 8, 6, 4)
    return {"image": image, "label": label}


def test_left_right_flip_swaps_the_side_specific_labels():
    patch = _lateral_patch()
    flip = RandFlipLateralSwapd(keys=["image", "label"], spatial_axes=(0,), prob=1.0)
    out = flip(patch)
    assert torch.equal(out["image"], torch.flip(patch["image"], [1]))
    # The kidney now at high x is the original left one, relabelled as the right kidney
    assert (out["label"][0, 6:] == 2).all() and (out["label"][0, :2] == 3).all()
    assert torch.equal(out["label"] == 6, torch.flip(patch["label"] == 6, [1]))


def test_other_axis_flips_keep_the_labels():
    patch = _lateral_patch()
    out = RandFlipLateralSwapd(keys=["image", "label"], spatial_axes=(1, 2), prob=1.0)(patch)
    assert torch.equal(out["label"], torch.flip(patch["label"], [2, 3]))
    assert torch.equal(out["image"], torch.flip(patch["image"], [2, 3]))


def test_flips_are_random_and_reproducible():
    flip = RandFlipLateralSwapd(keys=["image", "label"], prob=0.5)
    flip.set_random_state(seed=0)
    first = [flip(_lateral_patch())["image"] for _ in range(8)]
    flip.set_random_state(seed=0)
    again = [flip(_lateral_patch())["image"] for _ in range(8)]
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert len({tuple(x.flatten()[:3].tolist()) for x in first}) > 1  # not always the same flips


def test_flip_option_in_the_training_pipeline(synthetic_volume):
    transforms = get_train_transforms(spatial_size=(16, 16, 16), num_samples=4, sampling="classes",
                                      augmentation="strong", flip="all")
    assert "RandFlipLateralSwapd" in [type(t).__name__ for t in transforms.transforms]
    transforms.set_random_state(seed=0)
    for patch in transforms(synthetic_volume):
        assert tuple(patch["label"].shape) == (1, 16, 16, 16)
        assert set(patch["label"].unique().tolist()) <= {0.0, 6.0, 12.0, 13.0}  # 12 may become 13
    with pytest.raises(ValueError):
        get_flips("unknown")


def test_unknown_augmentation_raises():
    with pytest.raises(ValueError):
        get_augmentations("unknown")


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


def test_find_fold_checkpoints_one_per_fold_in_order(tmp_path):
    for fold in (2, 10, 1):  # numeric order, not alphabetical (fold_10 after fold_2)
        (tmp_path / f"fold_{fold}").mkdir()
        (tmp_path / f"fold_{fold}" / f"run_fold_{fold}-epoch5-dice0.60.ckpt").write_bytes(b"")
        (tmp_path / f"fold_{fold}" / "last.ckpt").write_bytes(b"")
    (tmp_path / "fold_notes").mkdir()  # ignored

    best = find_fold_checkpoints(tmp_path)
    assert [p.parent.name for p in best] == ["fold_1", "fold_2", "fold_10"]
    assert all(p.name != "last.ckpt" for p in best)
    assert all(p.name == "last.ckpt" for p in find_fold_checkpoints(tmp_path, kind="last"))
    with pytest.raises(FileNotFoundError):
        find_fold_checkpoints(tmp_path / "fold_1")


def test_load_model_keeps_only_the_network_weights(tmp_path):
    torch.manual_seed(0)
    net = get_model("unet")
    state = {f"model.{k}": v for k, v in net.state_dict().items()}
    state["loss_function.class_weight"] = torch.ones(14)  # loss buffer, must be skipped
    torch.save({"state_dict": state}, tmp_path / "x.ckpt")

    loaded = load_model(tmp_path / "x.ckpt", "unet", torch.device("cpu"))
    assert not loaded.training
    for key, value in net.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


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


class _SideAwareModel(torch.nn.Module):
    """
    Ideal model trained with left-right flips and label swap: channel 1 ("right kidney")
    wherever the input is positive on the high-x half of the image, channel 2 ("left kidney")
    on the low-x half, channel 0 (background) elsewhere.
    """

    def forward(self, x):
        high_x = (torch.arange(x.shape[2]) >= x.shape[2] // 2).view(1, 1, -1, 1, 1).float()
        organ = (x > 0).float()
        logits = torch.stack([1 - organ, organ * high_x, organ * (1 - high_x)], dim=1)[:, :, 0]
        return 10 * logits


def test_tta_swaps_back_the_side_specific_channels():
    image = torch.zeros(1, 1, 16, 8, 8)
    image[0, 0, 12:15, 2:6, 2:6] = 1  # an organ on the high-x (right) side
    plain = predict_volume(_SideAwareModel(), image, (16, 8, 8), torch.device("cpu"))
    swapped = predict_volume(_SideAwareModel(), image, (16, 8, 8), torch.device("cpu"), tta=True,
                             lateral_pairs=((1, 2),))
    no_swap = predict_volume(_SideAwareModel(), image, (16, 8, 8), torch.device("cpu"), tta=True)
    # With the swap the mirrored inputs agree with the plain prediction; without it, half of
    # them call the organ "left": the average is split between the two channels.
    assert torch.equal(swapped.argmax(dim=1), plain.argmax(dim=1))
    assert torch.allclose(swapped, plain, atol=1e-4)
    assert torch.allclose(no_swap[0, 1, 13, 3, 3], no_swap[0, 2, 13, 3, 3], atol=1e-4)


def test_ensemble_averages_the_models_probabilities():
    torch.manual_seed(0)
    models = [torch.nn.Conv3d(1, 3, kernel_size=3, padding=1).eval() for _ in range(3)]
    image = torch.randn(1, 1, 20, 18, 16)
    device = torch.device("cpu")
    singles = [predict_volume(m, image, (8, 8, 8), device, mode="gaussian") for m in models]
    ensemble = predict_volume(models, image, (8, 8, 8), device, mode="gaussian")
    assert torch.allclose(ensemble, torch.stack(singles).mean(dim=0), atol=1e-5)
    assert torch.allclose(predict_volume(models[:1], image, (8, 8, 8), device),
                          predict_volume(models[0], image, (8, 8, 8), device))
    # With tta: average over models x mirrors, the lateral swap still applies to each model
    side = predict_volume([_SideAwareModel(), _SideAwareModel()], torch.zeros(1, 1, 16, 8, 8),
                          (16, 8, 8), device, tta=True, lateral_pairs=((1, 2),))
    assert torch.allclose(side.sum(dim=1), torch.ones(1, 16, 8, 8), atol=1e-5)
    with pytest.raises(ValueError):
        predict_volume([], image, (8, 8, 8), device)


def test_unknown_sliding_window_mode_raises():
    with pytest.raises(ValueError):
        predict_volume(torch.nn.Identity(), torch.zeros(1, 1, 8, 8, 8), (8, 8, 8), torch.device("cpu"), mode="x")


# --- Visualization ---

def test_visualize_dice_per_organ():
    visualize = _load_script("visualize")
    gt = np.zeros((10, 10, 10), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[0:4, 0:4, 0:4] = 1
    pred[0:4, 0:4, 0:2] = 1   # half of organ 1 -> Dice 2*32/(64+32)
    gt[6:8, 6:8, 6:8] = 12    # organ 12 missed
    scores = visualize.dice_per_organ(gt, pred)
    assert set(scores) == {1, 12}
    assert scores[1] == pytest.approx(64 / 96)
    assert scores[12] == 0.0


def test_visualize_choose_slices():
    visualize = _load_script("visualize")
    label = np.zeros((10, 12, 20), dtype=np.uint8)
    label[2:5, 3:6, 4] = 1
    label[6:8, 3:6, 4] = 2           # z=4: two organs -> most organs
    label[2:4, 8:10, 10:15] = 4      # focus organ 4 spans z=10..14 -> middle 12
    z1, z2, y = visualize.choose_slices(label, focus_organ=4)
    assert (z1, z2) == (4, 12)
    assert 3 <= y <= 9
    assert visualize.choose_slices(label, focus_organ=7)[1] == z1  # absent organ: falls back to z1


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


# --- Training: disk space guard and default configuration ---

def _guarded_trainer(ckpt_dir, max_epochs=3):
    callbacks = [*build_checkpoint_callbacks("run", ckpt_dir), GracefulStop(ckpt_dir / "STOP"),
                 DiskSpaceGuard(ckpt_dir, min_free_gb=2.0)]
    return pl.Trainer(max_epochs=max_epochs, callbacks=callbacks, logger=False, accelerator="cpu",
                      enable_progress_bar=False, enable_model_summary=False)


def _fake_free_space(monkeypatch, values):
    """free_space_gb returns the given values in turn (then the last one)."""
    values = list(values)
    monkeypatch.setattr(callbacks_module, "free_space_gb",
                        lambda path: values.pop(0) if len(values) > 1 else values[0])


def test_free_space_gb_accepts_a_folder_not_created_yet(tmp_path):
    assert free_space_gb(tmp_path / "not" / "created" / "yet") > 0


def test_disk_guard_refuses_to_start_on_a_full_disk(tmp_path, monkeypatch):
    _fake_free_space(monkeypatch, [0.5])
    loader = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
    with pytest.raises(LowDiskSpaceError):
        fit_with_resume(_guarded_trainer(tmp_path), _DecliningDiceModule(), loader, loader, tmp_path)
    assert not (tmp_path / "last.ckpt").exists()


def test_disk_guard_stops_cleanly_with_an_error_and_a_valid_last_checkpoint(tmp_path, monkeypatch):
    # Enough space at the start and after epoch 0, then the disk fills up during epoch 1
    _fake_free_space(monkeypatch, [10.0, 10.0, 1.0])
    loader = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
    with pytest.raises(LowDiskSpaceError, match="--resume"):
        fit_with_resume(_guarded_trainer(tmp_path, max_epochs=5), _DecliningDiceModule(),
                        loader, loader, tmp_path)
    assert torch.load(tmp_path / "last.ckpt", weights_only=False)["epoch"] == 1  # resumable
    assert not (tmp_path / "DONE").exists()  # the fold is not marked as finished


def test_disk_guard_lets_a_normal_training_finish(tmp_path, monkeypatch):
    _fake_free_space(monkeypatch, [10.0])
    loader = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
    assert fit_with_resume(_guarded_trainer(tmp_path), _DecliningDiceModule(), loader, loader, tmp_path)
    assert (tmp_path / "DONE").exists()


def test_default_config_is_the_reference_configuration():
    """The defaults reproduce the README reference command (run cv_classes_w5_aug)."""
    reference = ("--max-epochs 250 --lr-scheduler cosine --warmup-epochs 5 --loss dicece "
                 "--num-samples 4 --batch-size 2 --accumulate 1 --sampling classes "
                 "--rare-weight 3 --rare-loss-weight 5 --augmentation strong").split()
    parser = add_train_args(argparse.ArgumentParser())
    assert TrainConfig.from_args(parser.parse_args([])) == TrainConfig.from_args(parser.parse_args(reference))
