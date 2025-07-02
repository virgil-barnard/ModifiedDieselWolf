from __future__ import annotations

from itertools import product
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    device = next(model.parameters()).device
    for batch in dataloader:
        data = batch["data"].to(device)
        label = batch["label"].to(device)
        metadata = batch.get("metadata", {})
        out = model(data)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits, label)
        snr_pred = None
        channel_pred = None
        if model.snr_head is not None and model.channel_head is not None:
            _, snr_pred, channel_pred = out
        elif model.snr_head is not None:
            _, snr_pred = out
        elif model.channel_head is not None:
            _, channel_pred = out
        if snr_pred is not None and "SNRdB" in metadata:
            snr_t = metadata["SNRdB"].to(device).float()
            loss = loss + model.hparams.snr_weight * mse(snr_pred, snr_t)
        if (
            channel_pred is not None
            and "CarrierFrequencyOffset" in metadata
            and "CarrierPhaseOffset" in metadata
        ):
            tgt = torch.stack(
                [
                    metadata["CarrierFrequencyOffset"].to(device).float(),
                    metadata["CarrierPhaseOffset"].to(device).float(),
                ],
                dim=1,
            )
            loss = loss + model.hparams.channel_weight * mse(channel_pred, tgt)
        total_loss += loss.item() * data.size(0)
        total += data.size(0)
    return total_loss / max(1, total)


def grid_search_loss_weights(
    model: nn.Module,
    val_loader: DataLoader,
    snr_weights: Iterable[float],
    channel_weights: Iterable[float],
) -> Tuple[float, float]:
    """Return weights yielding the lowest validation loss."""

    best = (model.hparams.snr_weight, model.hparams.channel_weight)
    best_loss = float("inf")
    for sw, cw in product(snr_weights, channel_weights):
        model.hparams.update({"snr_weight": sw, "channel_weight": cw})
        loss = _evaluate(model, val_loader)
        if loss < best_loss:
            best_loss = loss
            best = (sw, cw)
    # restore best weights
    model.hparams.update({"snr_weight": best[0], "channel_weight": best[1]})
    return best
