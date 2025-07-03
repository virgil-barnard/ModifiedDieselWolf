import yaml
import torch

from dieselwolf.models import build_backbone, MobileRaT, NMformer


def test_build_mobile_rat(tmp_path):
    cfg = {
        "backbone": "MobileRaT",
        "seq_len": 32,
        "num_classes": 3,
        "params": {"d_model": 16, "nhead": 2, "num_layers": 1},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    model = build_backbone(path)
    x = torch.randn(1, 2, 32)
    out = model(x)
    assert isinstance(model, MobileRaT)
    assert out.shape == (1, 3)


def test_build_nmformer(tmp_path):
    cfg = {
        "backbone": "NMformer",
        "seq_len": 32,
        "num_classes": 4,
        "params": {"d_model": 16, "nhead": 2, "num_layers": 1, "num_noise_tokens": 2},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    model = build_backbone(path)
    x = torch.randn(1, 2, 32)
    out = model(x)
    assert isinstance(model, NMformer)
    assert out.shape == (1, 4)
