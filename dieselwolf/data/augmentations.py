"""Data augmentation utilities for self-supervised training."""

from __future__ import annotations

import torch

from .TransformsRF import RandomCarrierFrequency


class RandomCrop:
    """Randomly crop an IQ sequence along the time dimension."""

    def __init__(self, crop_size: int) -> None:
        self.crop_size = crop_size

    def __call__(self, item: dict) -> dict:
        data = item["data"]
        n = data.shape[-1]
        if n <= self.crop_size:
            return item
        start = torch.randint(0, n - self.crop_size + 1, (1,)).item()
        item["data"] = data[..., start : start + self.crop_size]
        return item


class IQSwap:
    """Swap in-phase and quadrature channels with 50% probability."""

    def __call__(self, item: dict) -> dict:
        if torch.rand(1).item() < 0.5:
            item["data"] = item["data"].flip(0)
        return item


class RFAugment:
    """Compose random CFO, cropping and IQ swap augmentations."""

    def __init__(self, max_cfo: float, crop_size: int) -> None:
        self.random_cfo = RandomCarrierFrequency(delta_f_range=max_cfo)
        self.random_crop = RandomCrop(crop_size)
        self.iq_swap = IQSwap()

    def __call__(self, item: dict) -> dict:
        item = self.random_cfo(item)
        item = self.random_crop(item)
        item = self.iq_swap(item)
        return item
