import torch

from dieselwolf.models.mobile_rat import MobileRaT


def test_mobile_rat_forward():
    model = MobileRaT(seq_len=32, num_classes=3)
    x = torch.randn(2, 2, 32)
    out = model(x)
    assert out.shape == (2, 3)
