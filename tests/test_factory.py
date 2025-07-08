import yaml
import torch

from dieselwolf.models import build_backbone, ConfigurableMobileRaT, ConfigurableCNN


def test_build_mobile_rat(tmp_path):
    cfg = {
        "backbone": "ConfigurableMobileRaT",
        "seq_len": 32,
        "num_classes": 3,
        "params": {"nhead": 2, "num_layers": 1},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    model = build_backbone(path)
    x = torch.randn(1, 2, 32)
    out = model(x)
    assert isinstance(model, ConfigurableMobileRaT)
    assert out.shape == (1, 3)


def test_build_cnn(tmp_path):
    cfg = {
        "backbone": "ConfigurableCNN",
        "seq_len": 32,
        "num_classes": 4
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    model = build_backbone(path)
    x = torch.randn(1, 2, 32)
    out = model(x)
    assert isinstance(model, ConfigurableCNN)
    assert out.shape == (1, 4)
