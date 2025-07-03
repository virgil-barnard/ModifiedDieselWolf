import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from dieselwolf.data import (
    DigitalModulationDataset,
    RadioML2016Dataset,
    RadioML2018Dataset,
    RFAugment,
)
from dieselwolf.models import MoCoV3, build_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised pre-training")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument(
        "--radio2016", type=str, default=None, help="Path to RadioML2016 dataset"
    )
    parser.add_argument(
        "--radio2018", type=str, default=None, help="Path to RadioML2018 dataset"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="YAML file specifying the backbone for the encoder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    augment = RFAugment(max_cfo=0.01, crop_size=args.num_samples)
    synth = DigitalModulationDataset(
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        return_message=False,
        transform=augment,
    )
    datasets = [synth]
    if args.radio2016:
        datasets.append(RadioML2016Dataset(args.radio2016, transform=augment))
    if args.radio2018:
        datasets.append(RadioML2018Dataset(args.radio2018, transform=augment))
    train_ds = ConcatDataset(datasets)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if args.model_config:
        backbone = build_backbone(args.model_config)
    else:
        backbone = build_backbone("configs/mobile_rat.yaml")
    model = MoCoV3(
        encoder=backbone,
        feature_dim=args.feature_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        lr=args.lr,
    )

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices=1)
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
