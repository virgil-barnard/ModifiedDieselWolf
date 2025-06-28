"""PyTorch Lightning module for AMR classification."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import Accuracy

from dieselwolf.optim import Lookahead


class AMRClassifier(pl.LightningModule):
    """Minimal LightningModule wrapping an arbitrary classifier network."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        y = batch["label"]
        logits = self.forward(batch["data"])
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        y = batch["label"]
        logits = self.forward(batch["data"])
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        base_opt = AdamW(self.parameters(), lr=self.hparams.lr)
        optimizer = Lookahead(base_opt)
        t_max = getattr(self.trainer, "max_epochs", 1)

        if self.hparams.warmup_steps > 0:
            sched_warmup = LinearLR(
                optimizer, start_factor=0.1, total_iters=self.hparams.warmup_steps
            )
            sched_cosine = CosineAnnealingLR(
                optimizer, T_max=max(1, t_max - self.hparams.warmup_steps)
            )
            scheduler = SequentialLR(
                optimizer, [sched_warmup, sched_cosine], [self.hparams.warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, t_max))

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
