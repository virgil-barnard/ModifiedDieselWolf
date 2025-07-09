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
        if hasattr(t, "low"):
            t.low = self.current_snr
        if hasattr(t, "hi") and not hasattr(t, "SNRdB"):
            t.hi = self.current_snr
        print(f"[SNR callback] Set SNR to {self.current_snr} dB")

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

    def _evaluate(
        self, pl_module: pl.LightningModule
    ) -> tuple[torch.Tensor, float, float, float]:
        pl_module.eval()
        preds: List[int] = []
        targets: List[int] = []
        losses: List[torch.Tensor] = []
        snrs: List[torch.Tensor] = []
        device = pl_module.device
        with torch.no_grad():
            for batch in self.dataloader:
                x = batch["data"].to(device)
                y = batch["label"].to(device)
                logits = pl_module(x)
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(y.cpu().tolist())
                losses.append(pl_module.criterion(logits, y).cpu())
                if "metadata" in batch and "SNRdB" in batch["metadata"]:
                    snrs.append(batch["metadata"]["SNRdB"].float().mean().cpu())
        cm = torch.tensor(
            sk_confusion_matrix(
                targets, preds, labels=list(range(len(self.dataloader.dataset.classes)))
            )
        )
        acc = (cm.diagonal().sum() / cm.sum()).item() if cm.sum() > 0 else 0.0
        loss = torch.stack(losses).mean().item() if losses else 0.0
        avg_snr = torch.stack(snrs).mean().item() if snrs else float("nan")
        return cm, loss, acc, avg_snr

    def _plot(
        self, cm: torch.Tensor, loss: float, acc: float, avg_snr: float
    ) -> plt.Figure:
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
        title = f"Loss: {loss:.4f}  Acc: {acc*100:.2f}%"
        if not torch.isnan(torch.tensor(avg_snr)):
            title += f"  Avg SNR: {avg_snr:.1f} dB"
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        cm, loss, acc, avg_snr = self._evaluate(pl_module)
        fig = self._plot(cm, loss, acc, avg_snr)
        if self.output_dir:
            path = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch}.png")
            fig.savefig(path)
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_figure(
                self.log_tag, fig, global_step=trainer.current_epoch
            )
        plt.close(fig)


# Callback to store latent space projections


class LatentSpaceCallback(pl.Callback):
    """Save latent space projections when validation loss improves."""

    def __init__(
        self, dataloader, output_dir: str | None = None, log_tag: str = "latent_space"
    ) -> None:
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.log_tag = log_tag
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.class_names = getattr(dataloader.dataset, "classes", None)
        self.best_model_path: str | None = None

    def _extract_features(
        self, pl_module: pl.LightningModule, x: torch.Tensor
    ) -> torch.Tensor:
        backbone = pl_module.backbone
        if hasattr(backbone, "conv") and hasattr(backbone, "classifier"):
            feats = backbone.conv(x)
            feats = feats.flatten(1)
        elif hasattr(backbone, "frontend") and hasattr(backbone, "transformer"):
            feats = backbone.frontend(x)
            feats = backbone.transformer(feats)
            feats = feats.mean(dim=2)
        else:
            feats = backbone(x)
            if isinstance(feats, torch.Tensor) and feats.ndim > 2:
                feats = feats.flatten(1)
        return feats

    def _project(self, pl_module: pl.LightningModule) -> list[dict]:
        pl_module.eval()
        device = pl_module.device
        records = []
        with torch.no_grad():
            for batch in self.dataloader:
                x = batch["data"].to(device)
                feats = self._extract_features(pl_module, x)
                out = pl_module(x)
                logits = out[0] if isinstance(out, tuple) else out
                preds = logits.argmax(dim=1)
                for i in range(x.size(0)):
                    rec = {
                        "embedding": feats[i].cpu(),
                        "pred": preds[i].cpu(),
                        "label": batch["label"][i].cpu(),
                        "metadata": {},
                    }
                    if self.class_names is not None:
                        label_idx = int(rec["label"].item())
                        if 0 <= label_idx < len(self.class_names):
                            rec["class_name"] = str(self.class_names[label_idx])
                    meta = batch.get("metadata")
                    if meta:
                        rec["metadata"] = {
                            k: (v[i].cpu() if isinstance(v, torch.Tensor) else v)
                            for k, v in meta.items()
                        }
                    for k, v in batch.items():
                        if k in {"data", "label", "metadata"}:
                            continue
                        rec[k] = v[i].cpu() if isinstance(v, torch.Tensor) else v
                    records.append(rec)
        return records

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        loss = val_loss.item() if hasattr(val_loss, "item") else float(val_loss)
        if loss + 1e-6 < self.best_loss:
            self.best_loss = loss
            self.best_epoch = trainer.current_epoch
        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_cb is not None:
            self.best_model_path = checkpoint_cb.best_model_path

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate latent space projection from the best model after training."""
        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        ckpt_path = self.best_model_path
        if checkpoint_cb is not None:
            ckpt_path = checkpoint_cb.best_model_path or ckpt_path
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=pl_module.device, weights_only=False)
            if "state_dict" in state:
                pl_module.load_state_dict(state["state_dict"])
            else:
                pl_module.load_state_dict(state)
        records = self._project(pl_module)
        if self.output_dir:
            path = os.path.join(self.output_dir, f"epoch_{self.best_epoch}.pt")
            torch.save({"epoch": self.best_epoch, "records": records}, path)
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            embeddings = torch.stack([r["embedding"] for r in records])
            rows = []
            for r in records:
                label_name = r.get("class_name")
                if label_name is None:
                    idx = int(r["label"].item())
                    if self.class_names is not None and 0 <= idx < len(
                        self.class_names
                    ):
                        label_name = str(self.class_names[idx])
                    else:
                        label_name = str(idx)
                snr = None
                if isinstance(r.get("metadata"), dict):
                    snr = r["metadata"].get("SNRdB")
                    if isinstance(snr, torch.Tensor):
                        snr = snr.item()
                rows.append([label_name, snr])
            trainer.logger.experiment.add_embedding(
                embeddings,
                metadata=rows,
                metadata_header=["class", "SNRdB"],
                global_step=self.best_epoch,
                tag=self.log_tag,
            )
