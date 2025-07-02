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
TDLNakagami = transforms.TDLNakagami


def test_tdl_nakagami_impulse():
    torch.manual_seed(0)
    item = {"data": torch.zeros(2, 16), "metadata": {}}
    item["data"][0, 0] = 1.0
    transform = TDLNakagami(
        delays=[0.0, 1.0], avg_gains_dB=[0.0, 0.0], sample_rate=1.0, m=2.0
    )
    out = transform({"data": item["data"].clone(), "metadata": {}})
    assert out["data"].shape == item["data"].shape
    assert out["data"].abs().sum() > 0
