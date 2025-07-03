import subprocess
import torch
from dieselwolf.models import AMRClassifier, ConfigurableCNN


def test_finetune_script(tmp_path):
    ckpt = tmp_path / "pruned.ckpt"
    model = AMRClassifier(ConfigurableCNN(16, 4), num_classes=4)
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
            "--num-classes",
            "4",
        ]
    )
    assert out_ckpt.exists()
