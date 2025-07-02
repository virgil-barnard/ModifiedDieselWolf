import subprocess
import torch
from dieselwolf.models import AMRClassifier


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_samples: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * num_samples, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def test_finetune_script(tmp_path):
    ckpt = tmp_path / "pruned.ckpt"
    model = AMRClassifier(SimpleCNN(16, 4), num_classes=4)
    torch.save({"state_dict": model.state_dict()}, ckpt)
    out_ckpt = tmp_path / "finetuned.ckpt"
    subprocess.check_call(
        [
            "python",
            "scripts/finetune_pruned.py",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out_ckpt),
            "--epochs",
            "1",
            "--num-examples",
            "4",
            "--num-samples",
            "16",
            "--batch-size",
            "2",
        ]
    )
    assert out_ckpt.exists()
