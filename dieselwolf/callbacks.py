"""Training callbacks for DieselWolf."""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


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


class ConfusionMatrixCallback(pl.Callback):
    """Generate a confusion matrix on a separate dataloader after each epoch."""

    def __init__(
        self,
        dataloader,
        output_dir: str | None = None,
        log_tag: str = "confusion_matrix",
    ) -> None:
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.log_tag = log_tag
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _evaluate(self, pl_module: pl.LightningModule) -> torch.Tensor:
        pl_module.eval()
        preds: List[int] = []
        targets: List[int] = []
        device = pl_module.device
        with torch.no_grad():
            for batch in self.dataloader:
                x = batch["data"].to(device)
                y = batch["label"].to(device)
                logits = pl_module(x)
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(y.cpu().tolist())
        return torch.tensor(
            sk_confusion_matrix(
                targets, preds, labels=list(range(len(self.dataloader.dataset.classes)))
            )
        )

    def _plot(self, cm: torch.Tensor) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        classes = self.dataloader.dataset.classes
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm[i, j] / max(1, cm[i].sum())
                color = "white" if val > 0.7 else "black"
                ax.text(
                    j,
                    i,
                    f"{val*100:2.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        cm = self._evaluate(pl_module)
        fig = self._plot(cm)
        if self.output_dir:
            path = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch}.png")
            fig.savefig(path)
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_figure(
                self.log_tag, fig, global_step=trainer.current_epoch
            )
        plt.close(fig)
