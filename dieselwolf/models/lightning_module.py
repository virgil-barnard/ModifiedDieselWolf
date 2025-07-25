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
        adv_eps: float = 0.0,
        adv_weight: float = 0.5,
        adv_norm: float = float("inf"),
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
        x = batch["data"]
        out = self.forward(x)
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
        if self.hparams.adv_eps > 0:
            x_adv = x.detach().clone().requires_grad_(True)
            adv_logits = self.backbone(x_adv)
            adv_loss = self.criterion(adv_logits, y)
            grad = torch.autograd.grad(adv_loss, x_adv)[0]
            if self.hparams.adv_norm == float("inf"):
                delta = self.hparams.adv_eps * grad.sign()
            else:
                g_norm = grad.view(grad.size(0), -1).norm(
                    p=self.hparams.adv_norm, dim=1
                )
                g_norm = g_norm.view(-1, 1, 1)
                delta = self.hparams.adv_eps * grad / (g_norm + 1e-8)
            adv_data = x + delta.detach()
            adv_logits = self.backbone(adv_data)
            adv_loss = self.criterion(adv_logits, y)
            loss = (
                1 - self.hparams.adv_weight
            ) * loss + self.hparams.adv_weight * adv_loss

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

    def load_moco_weights(self, checkpoint_path: str) -> None:
        """Load encoder weights from a MoCo-v3 checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder_q."):
                backbone_state[k[len("encoder_q.") :]] = v
        self.backbone.load_state_dict(backbone_state, strict=False)
