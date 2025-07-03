import pytest
import subprocess
import torch
from dieselwolf.models import AMRClassifier, ConfigurableCNN

pytest.importorskip("onnx")


def test_benchmark_onnx(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    model = AMRClassifier(ConfigurableCNN(16, 4), num_classes=4)
    torch.save({"state_dict": model.state_dict()}, ckpt)
    onnx_path = tmp_path / "model.onnx"
    quant_path = tmp_path / "model_int8.onnx"
    subprocess.check_call(
        [
            "python",
            "scripts/export_onnx.py",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(onnx_path),
            "--num-samples",
            "16",
            "--num-classes",
            "4",
        ]
    )
    subprocess.check_call(
        [
            "python",
            "scripts/quantize_onnx.py",
            "--input",
            str(onnx_path),
            "--output",
            str(quant_path),
        ]
    )
    subprocess.check_call(
        [
            "python",
            "scripts/benchmark_onnx.py",
            "--model",
            str(quant_path),
            "--input-size",
            "16",
            "--num-iters",
            "1",
        ]
    )
