import tempfile
import torch
from dieselwolf.models import AMRClassifier, MoCoV3


class DummyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=-1))


def test_load_moco_weights():
    encoder = DummyEncoder()
    model = MoCoV3(encoder, feature_dim=4, queue_size=4)
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save({"state_dict": model.state_dict()}, f.name)
        clf = AMRClassifier(DummyEncoder(), num_classes=4)
        clf.load_moco_weights(f.name)
        for p1, p2 in zip(clf.backbone.fc.parameters(), encoder.fc.parameters()):
            assert torch.allclose(p1, p2)
