import torch
import torch.nn as nn

from dieselwolf.models import AMRClassifier


class DummyNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def test_amrclassifier_snr_forward():
    model = AMRClassifier(DummyNet(num_classes=3), num_classes=3, predict_snr=True)
    x = torch.randn(2, 2, 2)
    logits, snr = model(x)
    assert logits.shape == (2, 3)
    assert snr.shape == (2,)


def test_amrclassifier_channel_forward():
    model = AMRClassifier(DummyNet(num_classes=3), num_classes=3, predict_channel=True)
    x = torch.randn(2, 2, 2)
    logits, channel = model(x)
    assert logits.shape == (2, 3)
    assert channel.shape == (2, 2)


def test_amrclassifier_training_step_snr():
    model = AMRClassifier(DummyNet(num_classes=3), num_classes=3, predict_snr=True)
    batch = {
        "data": torch.randn(2, 2, 2),
        "label": torch.tensor([0, 1]),
        "metadata": {"SNRdB": torch.tensor([10.0, 5.0])},
    }
    loss = model.training_step(batch, 0)
    assert loss.item() > 0


def test_amrclassifier_training_step_channel():
    model = AMRClassifier(DummyNet(num_classes=3), num_classes=3, predict_channel=True)
    batch = {
        "data": torch.randn(2, 2, 2),
        "label": torch.tensor([0, 1]),
        "metadata": {
            "CarrierFrequencyOffset": torch.tensor([0.1, 0.2]),
            "CarrierPhaseOffset": torch.tensor([10.0, 5.0]),
        },
    }
    loss = model.training_step(batch, 0)
    assert loss.item() > 0
