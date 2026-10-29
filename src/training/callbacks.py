"""
Callbacks shared by the training scripts.
"""

import signal
from pathlib import Path

import pytorch_lightning as pl


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
