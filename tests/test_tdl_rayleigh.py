import importlib.util
import pathlib
import torch

spec = importlib.util.spec_from_file_location(
    "TransformsRF",
    pathlib.Path(__file__).resolve().parents[1]
    / "dieselwolf"
    / "data"
    / "TransformsRF.py",
)
transforms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transforms)
TDLRayleigh = transforms.TDLRayleigh


def test_tdl_rayleigh_impulse():
    torch.manual_seed(0)
    item = {"data": torch.zeros(2, 16), "metadata": {}}
    item["data"][0, 0] = 1.0
    transform = TDLRayleigh(delays=[0.0, 1.0], avg_gains_dB=[0.0, 0.0], sample_rate=1.0)
    out = transform({"data": item["data"].clone(), "metadata": {}})
    assert out["data"].shape == item["data"].shape
    assert out["data"].abs().sum() > 0
