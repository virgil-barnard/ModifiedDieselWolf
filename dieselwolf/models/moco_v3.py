"""Minimal MoCo-v3 implementation for self-supervised learning."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl


class MoCoQueue(nn.Module):
    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        self.register_buffer("queue", torch.randn(size, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.size = size

    @torch.no_grad()
    def dequeue_and_enqueue(self, feats: torch.Tensor) -> None:
        feats = feats.detach()
        batch_size = feats.shape[0]
        ptr = int(self.ptr)
        if ptr + batch_size <= self.size:
            self.queue[ptr : ptr + batch_size] = feats
        else:
            first = self.size - ptr
            self.queue[ptr:] = feats[:first]
            self.queue[: batch_size - first] = feats[first:]
        self.ptr[0] = (ptr + batch_size) % self.size


class MoCoV3(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 128,
        queue_size: int = 1024,
        momentum: float = 0.99,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.queue = MoCoQueue(feature_dim, queue_size)
        self.m = momentum
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder_q(x), dim=1)

    @torch.no_grad()
    def momentum_update(self) -> None:
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.mul_(self.m).add_(q.data, alpha=1.0 - self.m)

    def training_step(self, batch, batch_idx: int):
        x1, x2 = batch
        q = self.forward(x1)
        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.encoder_k(x2), dim=1)
        queue = self.queue.queue.clone().detach()
        logits = q @ torch.cat([k, queue], dim=0).t()
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.criterion(logits / 0.07, labels)
        self.queue.dequeue_and_enqueue(k)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.encoder_q.parameters(), lr=self.lr)
