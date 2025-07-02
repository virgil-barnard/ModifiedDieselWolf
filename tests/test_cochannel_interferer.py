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
CoChannelQPSKInterferer = transforms.CoChannelQPSKInterferer


def test_cochannel_qpsk_interferer_power():
    torch.manual_seed(0)
    item = {"data": torch.ones(2, 64), "metadata": {}}
    transform = CoChannelQPSKInterferer(sir_dB=10, seed=0)
    out = transform({"data": item["data"].clone(), "metadata": {}})
    interference = out["data"] - item["data"]
    p_signal = item["data"].pow(2).mean()
    p_interf = interference.pow(2).mean()
    ratio_db = 10 * torch.log10(p_signal / p_interf)
    assert abs(ratio_db.item() - 10) < 0.5
    assert out["metadata"]["JammerSNRdB"] == 10
