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
DopplerShift = transforms.DopplerShift


def test_doppler_shift_delay():
    sr = 1000
    shift = 100
    item = {
        "data": torch.zeros(2, 32),
        "metadata": {},
    }
    item["data"][0, 8] = 1.0
    transform = DopplerShift(shift_hz=shift, sample_rate=sr)
    out = transform({"data": item["data"].clone(), "metadata": {}})
    idx = out["data"][0].argmax().item()
    expected = int(round(8 * (sr + shift) / sr))
    assert abs(idx - expected) <= 1
