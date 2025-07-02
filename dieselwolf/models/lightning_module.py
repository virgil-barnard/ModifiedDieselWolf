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
        predict_snr: bool = False,
        predict_channel: bool = False,
        snr_weight: float = 1.0,
        channel_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        if predict_snr:
            self.snr_head = nn.Linear(num_classes, 1)
        else:
            self.snr_head = None
        if predict_channel:
            self.channel_head = nn.Linear(num_classes, 2)
        else:
            self.channel_head = None

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        outputs = [logits]
        if self.snr_head is not None:
            outputs.append(self.snr_head(logits).squeeze(-1))
        if self.channel_head is not None:
            outputs.append(self.channel_head(logits))
        if len(outputs) == 1:
            return outputs[0]
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        return outputs[0], outputs[1], outputs[2]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        y = batch["label"]
        out = self.forward(batch["data"])
        logits = out[0] if isinstance(out, tuple) else out
        snr_pred = None
        channel_pred = None
        if self.snr_head is not None and self.channel_head is not None:
            _, snr_pred, channel_pred = out
        elif self.snr_head is not None:
            _, snr_pred = out
        elif self.channel_head is not None:
            _, channel_pred = out

        loss = self.criterion(logits, y)
        if (
            snr_pred is not None
            and "metadata" in batch
            and "SNRdB" in batch["metadata"]
        ):
            snr_target = batch["metadata"]["SNRdB"].float()
            snr_loss = self.regression_loss(snr_pred, snr_target)
            loss = loss + self.hparams.snr_weight * snr_loss
            self.log("train_snr_loss", snr_loss)
        if (
            channel_pred is not None
            and "metadata" in batch
            and "CarrierFrequencyOffset" in batch["metadata"]
            and "CarrierPhaseOffset" in batch["metadata"]
        ):
            target = torch.stack(
                [
                    batch["metadata"]["CarrierFrequencyOffset"].float(),
                    batch["metadata"]["CarrierPhaseOffset"].float(),
                ],
                dim=1,
            )
            chan_loss = self.regression_loss(channel_pred, target)
            loss = loss + self.hparams.channel_weight * chan_loss
            self.log("train_channel_loss", chan_loss)
        self.train_acc(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        y = batch["label"]
        out = self.forward(batch["data"])
        logits = out[0] if isinstance(out, tuple) else out
        snr_pred = None
        channel_pred = None
        if self.snr_head is not None and self.channel_head is not None:
            _, snr_pred, channel_pred = out
        elif self.snr_head is not None:
            _, snr_pred = out
        elif self.channel_head is not None:
            _, channel_pred = out

        loss = self.criterion(logits, y)
        if (
            snr_pred is not None
            and "metadata" in batch
            and "SNRdB" in batch["metadata"]
        ):
            snr_target = batch["metadata"]["SNRdB"].float()
            snr_loss = self.regression_loss(snr_pred, snr_target)
            loss = loss + self.hparams.snr_weight * snr_loss
            self.log("val_snr_loss", snr_loss, prog_bar=True)
        if (
            channel_pred is not None
            and "metadata" in batch
            and "CarrierFrequencyOffset" in batch["metadata"]
            and "CarrierPhaseOffset" in batch["metadata"]
        ):
            target = torch.stack(
                [
                    batch["metadata"]["CarrierFrequencyOffset"].float(),
                    batch["metadata"]["CarrierPhaseOffset"].float(),
                ],
                dim=1,
            )
            chan_loss = self.regression_loss(channel_pred, target)
            loss = loss + self.hparams.channel_weight * chan_loss
            self.log("val_channel_loss", chan_loss, prog_bar=True)
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
