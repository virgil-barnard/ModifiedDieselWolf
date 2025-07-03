import torch

from dieselwolf.models import ConfigurableMobileRaT


def test_configurable_mobile_rat_forward():
    model = ConfigurableMobileRaT(
        seq_len=32,
        num_classes=3,
        conv_channels=[8, 16],
        kernel_sizes=[3, 3],
        nhead=2,
        num_layers=1,
    )
    x = torch.randn(2, 2, 32)
    out = model(x)
    assert out.shape == (2, 3)
