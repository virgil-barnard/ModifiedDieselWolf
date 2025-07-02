import argparse

import torch

from dieselwolf.models import AMRClassifier


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_samples: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * num_samples, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export model to ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = AMRClassifier(
        SimpleCNN(args.num_samples, args.num_classes), args.num_classes
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
