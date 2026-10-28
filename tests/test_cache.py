"""
Checks that the on-disk pre-processing cache (src/data/dataset.py) works:
    1. it makes data loading drastically faster once filled,
    2. it returns exactly the same data as the uncached pipeline,
    3. it is invalidated when the transforms change,
    4. a short training with the cache runs correctly and later epochs are much faster.

Uses the real volumes in data/raw (a few of them) and a temporary cache directory.

Run with either:
    python tests/test_cache.py
    pytest tests/test_cache.py -s
"""

import sys
import tempfile
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.losses import GeneralizedDiceLoss
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_data_dicts, get_dataset, get_loaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.builder import get_model
from src.training.lightning_module import OrganMapperModule

DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPEEDUP_MIN = 10  # a cached pass must be at least this many times faster than the first one


def _files(n: int) -> list[dict]:
    files = get_data_dicts(DATA_DIR)[:n]
    assert len(files) == n, f"Need {n} volumes in {DATA_DIR}"
    return files


def _load_all(dataset) -> float:
    """Loads every item once and returns the elapsed time in seconds."""
    start = time.perf_counter()
    for i in range(len(dataset)):
        dataset[i]
    return time.perf_counter() - start


def _first(sample):
    """RandCropByPosNegLabeld returns a list of samples; take the first one."""
    return sample[0] if isinstance(sample, list) else sample


def test_cache_speedup():
    files = _files(2)
    with tempfile.TemporaryDirectory() as tmp:
        ds = get_dataset(files, get_val_transforms(), cache_dir=tmp, cache_name="val")
        first = _load_all(ds)   # builds the cache
        second = _load_all(ds)  # reads it
        print(f"\n[speedup] first pass {first:.1f}s | cached pass {second:.2f}s | x{first / second:.0f}")
        assert second * SPEEDUP_MIN < first, "Cached pass is not drastically faster"


def test_cache_returns_same_data_as_no_cache():
    files = _files(1)
    transforms = get_val_transforms()  # deterministic, so outputs must match exactly
    plain = get_dataset(files, transforms, cache_dir=None)[0]
    with tempfile.TemporaryDirectory() as tmp:
        cached = get_dataset(files, transforms, cache_dir=tmp, cache_name="val")
        cached[0]                  # fills the cache
        from_disk = cached[0]      # read back from disk
    for key in ("image", "label"):
        assert torch.equal(plain[key], from_disk[key]), f"'{key}' differs with the cache"


def test_random_augmentations_still_vary():
    files = _files(1)
    with tempfile.TemporaryDirectory() as tmp:
        ds = get_dataset(files, get_train_transforms(), cache_dir=tmp, cache_name="train")
        a, b = _first(ds[0]), _first(ds[0])
    assert not torch.equal(a["image"], b["image"]), "Random crop/augmentations are frozen by the cache"


def test_cache_invalidated_when_transforms_change():
    files = _files(1)
    with tempfile.TemporaryDirectory() as tmp:
        get_dataset(files, get_train_transforms(), cache_dir=tmp, cache_name="train")[0]
        n_before = len(list(Path(tmp).rglob("*.pt")))
        get_dataset(files, get_train_transforms(), cache_dir=tmp, cache_name="train")[0]
        n_same = len(list(Path(tmp).rglob("*.pt")))
        get_dataset(files, get_train_transforms(spacing=(2.0, 2.0, 2.0)), cache_dir=tmp, cache_name="train")[0]
        n_changed = len(list(Path(tmp).rglob("*.pt")))
    assert n_same == n_before, "Identical transforms should reuse the cache"
    assert n_changed > n_before, "A new spacing must create new cache entries (stale cache!)"


class _EpochTimer(pl.Callback):
    def __init__(self):
        self.durations: list[float] = []
        self.losses: list[float] = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        self.durations.append(time.perf_counter() - self._start)
        self.losses.append(float(trainer.callback_metrics["train/loss_epoch"]))


def test_training_with_cache_is_correct_and_gets_faster():
    files = _files(3)
    train_files, val_files = files[:2], files[2:]
    spatial_size = (96, 96, 96)
    pl.seed_everything(42)

    with tempfile.TemporaryDirectory() as tmp:
        train_loader, val_loader = get_loaders(
            train_files, val_files,
            train_transforms=get_train_transforms(spatial_size=spatial_size),
            val_transforms=get_val_transforms(spatial_size=spatial_size),
            batch_size=1,
            cache_dir=tmp,
        )

        module = OrganMapperModule(
            model=get_model("unet", spatial_size=spatial_size),
            loss_function=GeneralizedDiceLoss(to_onehot_y=True, softmax=True, include_background=False),
            optimizer_class=AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
            val_roi_size=spatial_size,
        )
        weights_before = [p.detach().clone() for p in module.parameters()]

        timer = _EpochTimer()
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[timer],
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    d, losses = timer.durations, timer.losses
    print(f"\n[training] epoch durations: {[round(x, 1) for x in d]}s | losses: {[round(x, 3) for x in losses]}")
    dice = float(trainer.callback_metrics["val/dice"])

    # Training works: finite loss, weights updated, validation metric computed
    assert all(torch.isfinite(torch.tensor(losses))), "Non-finite training loss"
    assert any(not torch.equal(b, a) for b, a in zip(weights_before, module.parameters())), "Weights did not change"
    assert 0.0 <= dice <= 1.0, f"Invalid validation Dice: {dice}"

    # Epoch 1 fills the cache (loading + resampling); the next ones only read it
    assert d[1] * 3 < d[0], f"Epoch 2 ({d[1]:.1f}s) is not much faster than epoch 1 ({d[0]:.1f}s)"
    assert d[2] * 3 < d[0], f"Epoch 3 ({d[2]:.1f}s) is not much faster than epoch 1 ({d[0]:.1f}s)"


if __name__ == "__main__":
    # Minimal runner so the checks work without pytest installed
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
