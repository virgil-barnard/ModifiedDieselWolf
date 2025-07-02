import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EMA,
)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AMR classifier")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16-mixed", "32"],
        default="32",
        help="Trainer precision mode",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="LR warm-up steps")
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Exponential moving average decay. Set 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_ds = DigitalModulationDataset(
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        return_message=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = DigitalModulationDataset(
        num_examples=max(1, args.num_examples // 4),
        num_samples=args.num_samples,
        return_message=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    backbone = SimpleCNN(args.num_samples, len(train_ds.classes))
    model = AMRClassifier(
        backbone,
        num_classes=len(train_ds.classes),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
    )

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="val_loss"),
        LearningRateMonitor(),
    ]
    if args.ema_decay > 0:
        callbacks.append(EMA(decay=args.ema_decay, use_ema_weights=True))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
