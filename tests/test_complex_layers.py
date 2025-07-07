import torch
from dieselwolf.complex_layers import (
    ComplexConv1d,
    ComplexBatchNorm1d,
    ComplexLinear,
    ComplexLayerNorm,
)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = ComplexConv1d(2, 4, kernel_size=3, padding=1)
        self.bn = ComplexBatchNorm1d(4)
        self.fc = ComplexLinear(4 * 32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = x.flatten(1)
        return self.fc(x)


def test_checkpoint_io(tmp_path):
    model = DummyModel()
    inp = torch.randn(1, 4, 32)
    out1 = model(inp)

    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = DummyModel()
    model2.load_state_dict(torch.load(ckpt))
    out2 = model2(inp)

    assert torch.allclose(out1, out2)


def test_weight_initialisation():
    layer = ComplexLinear(8, 4)
    for param in layer.parameters():
        assert param.abs().sum() > 0


def test_layernorm_shape():
    norm = ComplexLayerNorm(4)
    x = torch.randn(2, 8, 16)
    out = norm(x)
    assert out.shape == x.shape
