import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.models import AMRClassifier, ConfigurableCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune pruned model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-examples", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_classes = [
        "OOK",
        "4ASK",
        "8ASK",
        "BPSK",
        "QPSK",
        "Pi4QPSK",
        "8PSK",
        "16PSK",
        "16QAM",
        "32QAM",
        "64QAM",
        "16APSK",
        "32APSK",
    ]
    class_names = all_classes[: args.num_classes] if args.num_classes else "all"
    train_ds = DigitalModulationDataset(
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        classes=class_names,
        return_message=False,
        transform=None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = DigitalModulationDataset(
        num_examples=max(1, args.num_examples // 4),
        num_samples=args.num_samples,
        classes=class_names,
        return_message=False,
        transform=None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    num_classes = args.num_classes or len(train_ds.classes)
    model = AMRClassifier(
        ConfigurableCNN(args.num_samples, num_classes),
        num_classes=num_classes,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="cpu", devices=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(args.output)


if __name__ == "__main__":
    main()
