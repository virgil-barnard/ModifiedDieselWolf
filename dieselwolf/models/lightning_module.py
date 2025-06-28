"""PyTorch Lightning module for AMR classification."""

from __future__ import annotations

import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl


class AMRClassifier(pl.LightningModule):
    """Minimal LightningModule wrapping an arbitrary classifier network."""

    def __init__(self, backbone: nn.Module, num_classes: int, lr: float = 1e-3) -> None:
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

