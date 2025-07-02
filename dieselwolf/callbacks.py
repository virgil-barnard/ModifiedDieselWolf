"""Training callbacks for DieselWolf."""

from __future__ import annotations

import pytorch_lightning as pl


class SNRCurriculumCallback(pl.Callback):
    """Decrease dataset SNR after validation loss plateaus."""

    def __init__(
        self,
        dataset,
        start_snr: int = 20,
        step: int = 5,
        min_snr: int = 0,
        patience: int = 2,
    ) -> None:
        self.dataset = dataset
        self.current_snr = start_snr
        self.step = step
        self.min_snr = min_snr
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def _set_snr(self) -> None:
        t = getattr(self.dataset, "transform", None)
        if t is None:
            return
        if hasattr(t, "SNRdB"):
            t.SNRdB = self.current_snr
        if hasattr(t, "low") and hasattr(t, "hi"):
            t.low = t.hi = self.current_snr

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._set_snr()

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        loss = val_loss.item() if hasattr(val_loss, "item") else float(val_loss)
        if loss + 1e-6 < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if (
                self.wait >= self.patience
                and self.current_snr - self.step >= self.min_snr
            ):
                self.current_snr -= self.step
                self._set_snr()
                self.wait = 0
