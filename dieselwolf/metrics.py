"""Evaluation utilities for DieselWolf models."""

from __future__ import annotations

import time
from typing import Iterable

import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def accuracy_per_snr(
    preds: Iterable[int], targets: Iterable[int], snrs: Iterable[float]
):
    """Return accuracy grouped by SNR."""
    preds = torch.tensor(list(preds))
    targets = torch.tensor(list(targets))
    snrs = torch.tensor(list(snrs))
    acc = {}
    for snr in torch.unique(snrs):
        mask = snrs == snr
        correct = (preds[mask] == targets[mask]).float().mean().item()
        acc[float(snr)] = correct
    return acc


def confusion_at_0db(
    preds: Iterable[int], targets: Iterable[int], snrs: Iterable[float]
):
    """Confusion matrix computed on samples with 0 dB SNR."""
    preds = torch.tensor(list(preds))
    targets = torch.tensor(list(targets))
    snrs = torch.tensor(list(snrs))
    mask = snrs == 0
    return sk_confusion_matrix(targets[mask].numpy(), preds[mask].numpy())


def measure_latency(
    model: torch.nn.Module, inputs: torch.Tensor, device: torch.device
) -> float:
    """Measure average inference latency in milliseconds."""
    model = model.to(device)
    inputs = inputs.to(device)
    with torch.no_grad():
        for _ in range(5):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000.0
