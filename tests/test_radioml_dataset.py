import tempfile
from pathlib import Path

import numpy as np
import torch

from dieselwolf.data import RadioML2016Dataset, RFAugment


def test_radioml_dataset_loading():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "radio.npz"
        np.savez(
            path,
            iq=np.random.randn(2, 2, 8).astype(np.float32),
            labels=np.array([0, 1]),
            snr=np.array([0, 0]),
        )
        ds = RadioML2016Dataset(path)
        item = ds[0]
        assert item["data"].shape == (2, 8)
        assert item["label"].item() == 0


def test_rfaugment():
    aug = RFAugment(max_cfo=0.01, crop_size=4)
    item = {"data": torch.randn(2, 8), "metadata": {"dt": 1.0}}
    out = aug(item)
    assert out["data"].shape[-1] == 4
