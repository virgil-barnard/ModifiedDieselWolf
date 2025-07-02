import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.metrics import accuracy_per_snr, confusion_at_0db, measure_latency
from dieselwolf.models import AMRClassifier


class SimpleCNN(nn.Module):
    """Very small CNN used for quick benchmarks."""

    def __init__(self, num_samples: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * num_samples, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="benchmarks")
    return parser.parse_args()


def evaluate(
    model: nn.Module, loader: DataLoader
) -> tuple[List[int], List[int], List[float]]:
    model.eval()
    preds: List[int] = []
    targets: List[int] = []
    snrs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["data"])
            preds.extend(logits.argmax(dim=1).tolist())
            targets.extend(batch["label"].tolist())
            snrs.extend(
                batch.get("metadata", {})
                .get("SNRdB", torch.zeros(len(logits)))
                .tolist()
            )
    return preds, targets, snrs


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ds = DigitalModulationDataset(
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        return_message=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = AMRClassifier(
        SimpleCNN(args.num_samples, len(ds.classes)), num_classes=len(ds.classes)
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    preds, targets, snrs = evaluate(model, loader)

    acc = accuracy_per_snr(preds, targets, snrs)
    conf = confusion_at_0db(preds, targets, snrs)

    # accuracy vs snr plot
    plt.figure()
    xs = sorted(acc.keys())
    ys = [acc[x] for x in xs]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR")
    plt.savefig(os.path.join(args.output_dir, "accuracy_vs_snr.png"))
    plt.close()

    # confusion matrix
    plt.figure()
    plt.imshow(conf, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix @ 0 dB")
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close()

    latency = measure_latency(
        model, torch.randn(1, 2, args.num_samples), device=torch.device("cpu")
    )
    with open(os.path.join(args.output_dir, "latency_ms.txt"), "w") as f:
        f.write(f"{latency:.3f}")


if __name__ == "__main__":
    main()
