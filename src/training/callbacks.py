"""
Callbacks shared by the training scripts.
"""

import shutil
import signal
from pathlib import Path

import pytorch_lightning as pl


class LowDiskSpaceError(RuntimeError):
    """Raised when the disk holding the checkpoints is (nearly) full."""


def free_space_gb(path: str | Path) -> float:
    """
    Free space of the disk holding path, in GB.

    Args:
        path: A file or folder; it does not need to exist yet (its closest
            existing parent is used).

    Returns:
        The free space in GB (1e9 bytes).
    """
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


class DiskSpaceGuard(pl.Callback):
    """
    Stops the training cleanly before the disk is full.

    A full disk makes the checkpoint write fail in the middle of an epoch. This
    callback checks the free space of the checkpoint disk:
      - when the training starts: below min_free_gb, it raises LowDiskSpaceError
        before any work is done;
      - at the end of each epoch (after last.ckpt has been written): below
        min_free_gb, it stops the training, which can then be resumed with
        --resume once space has been freed. fit_with_resume turns this stop
        into a LowDiskSpaceError, so that the script exits with an error code.

    Attributes:
        triggered: True once the training has been stopped for lack of space.
    """

    def __init__(self, path: str | Path, min_free_gb: float):
        self.path = Path(path)
        self.min_free_gb = min_free_gb
        self.triggered = False

    def _message(self, free: float) -> str:
        return (f"Only {free:.1f} GB free on the disk of {self.path} "
                f"(minimum {self.min_free_gb:g} GB, option --min-free-gb).")

    def on_fit_start(self, trainer, pl_module):
        free = free_space_gb(self.path)
        if free < self.min_free_gb:
            raise LowDiskSpaceError(self._message(free) + " Free some space before training.")

    def on_train_epoch_end(self, trainer, pl_module):
        # Runs after the validation of this epoch, once the checkpoints are written
        free = free_space_gb(self.path)
        if free < self.min_free_gb:
            self.triggered = True
            trainer.should_stop = True
            print(f"\n{self._message(free)} Stopping at the end of this epoch; "
                  "free some space, then relaunch the same command + --resume.")


class GracefulStop(pl.Callback):
    """
    Lets the user stop a long training cleanly, to resume it later with --resume.

    The training stops at the end of the current epoch (after the validation and
    after last.ckpt has been written) when either:
      - Ctrl+C is pressed once (press it a second time to abort immediately), or
      - the file ``stop_file`` exists (touch it from another terminal; useful
        when the training runs in the background).

    Attributes:
        requested: True once a stop has been asked for.
    """

    def __init__(self, stop_file: Path):
        self.stop_file = Path(stop_file)
        self.requested = False
        self._previous_handler = None

    def _on_sigint(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt  # second Ctrl+C: abort now
        self.requested = True
        print("\nCtrl+C received: stopping at the end of the current epoch "
              "(press Ctrl+C again to abort immediately).")

    def setup(self, trainer, pl_module, stage):
        self.stop_file.unlink(missing_ok=True)  # ignore a stale file from an earlier run
        try:
            self._previous_handler = signal.signal(signal.SIGINT, self._on_sigint)
        except ValueError:  # not in the main thread: Ctrl+C keeps its default behaviour
            self._previous_handler = None

    def teardown(self, trainer, pl_module, stage):
        if self._previous_handler is not None:
            signal.signal(signal.SIGINT, self._previous_handler)
            self._previous_handler = None

    def on_train_epoch_end(self, trainer, pl_module):
        # Runs after the validation of this epoch, once the checkpoints are written
        if self.stop_file.exists():
            self.requested = True
        if self.requested:
            trainer.should_stop = True
