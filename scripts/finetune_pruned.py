import argparse

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.models import AMRClassifier


class SimpleCNN(nn.Module):
    def __init__(self, num_samples: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * num_samples, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune pruned model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-examples", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = DigitalModulationDataset(
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        return_message=False,
        transform=None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = DigitalModulationDataset(
        num_examples=max(1, args.num_examples // 4),
        num_samples=args.num_samples,
        return_message=False,
        transform=None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AMRClassifier(
        SimpleCNN(args.num_samples, len(train_ds.classes)),
        num_classes=len(train_ds.classes),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="cpu", devices=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(args.output)


if __name__ == "__main__":
    main()
