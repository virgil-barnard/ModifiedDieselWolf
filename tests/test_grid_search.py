import torch
from torch.utils.data import DataLoader

from dieselwolf.data.DigitalModulations import DigitalModulationDataset
from dieselwolf.models import AMRClassifier
from dieselwolf.tuning import grid_search_loss_weights


class DummyNet(torch.nn.Module):
    def __init__(self, num_samples: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * num_samples, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_grid_search_returns_weights():
    ds = DigitalModulationDataset(num_examples=2, num_samples=16, return_message=False)
    loader = DataLoader(ds, batch_size=2)
    model = AMRClassifier(
        DummyNet(16, len(ds.classes)),
        num_classes=len(ds.classes),
        predict_snr=True,
        predict_channel=True,
    )
    weights = grid_search_loss_weights(model, loader, [0.5, 1.0], [0.5, 1.0])
    assert weights[0] in [0.5, 1.0]
    assert weights[1] in [0.5, 1.0]
