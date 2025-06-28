import torch

from dieselwolf.models.nmformer import NMformer


def test_nmformer_forward():
    model = NMformer(seq_len=32, num_classes=3, num_noise_tokens=4)
    x = torch.randn(2, 2, 32)
    out = model(x)
    assert out.shape == (2, 3)
