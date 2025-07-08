import argparse

import torch
import importlib

from dieselwolf.models import AMRClassifier, ConfigurableCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export model to ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("onnx") is None:
        print("onnx package not installed, skipping export.")
        return
    model = AMRClassifier(
        ConfigurableCNN(args.num_samples, args.num_classes), args.num_classes
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    dummy = torch.randn(1, 2, args.num_samples)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )


if __name__ == "__main__":
    main()
