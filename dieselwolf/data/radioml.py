"""Loaders for the RadioML datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class RadioMLDataset(Dataset):
    """Generic loader for RadioML datasets stored as ``.npz`` files."""

    def __init__(self, path: str | Path, transform: Optional[callable] = None) -> None:
        path = Path(path)
        data = np.load(path)
        self.iq = torch.from_numpy(data["iq"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()
        self.snr = torch.from_numpy(data.get("snr", np.zeros(len(self.iq)))).float()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {
            "data": self.iq[idx],
            "label": self.labels[idx],
            "metadata": {"SNRdB": self.snr[idx]},
        }
        if self.transform is not None:
            item = self.transform(item)
        return item


class RadioML2016Dataset(RadioMLDataset):
    """RadioML 2016.10a dataset."""


class RadioML2018Dataset(RadioMLDataset):
    """RadioML 2018.01 dataset."""


class RML22Dataset(RadioMLDataset):
    """DeepSig RML22 dataset."""
