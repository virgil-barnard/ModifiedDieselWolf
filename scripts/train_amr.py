import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EMA,
)

from dieselwolf.callbacks import SNRCurriculumCallback
from dieselwolf.data.TransformsRF import AWGN
from torch.utils.data import DataLoader

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.models import AMRClassifier, ConfigurableCNN, build_backbone


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
    parser.add_argument(
        "--snr-start", type=int, default=20, help="Starting SNR for curriculum"
    )
    parser.add_argument(
        "--snr-patience", type=int, default=2, help="Epochs to wait before lowering SNR"
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=str,
        default=None,
        help="Path to MoCo checkpoint for fine-tuning",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to YAML file specifying the backbone configuration",
    )
    parser.add_argument(
        "--cnn-channels",
        type=str,
        default="32",
        help="Comma-separated channels for configurable CNN",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=str,
        default="3",
        help="Comma-separated kernel sizes",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Enable batch normalization",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability between conv layers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_ds = DigitalModulationDataset(
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        return_message=False,
        transform=AWGN(args.snr_start),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = DigitalModulationDataset(
        num_examples=max(1, args.num_examples // 4),
        num_samples=args.num_samples,
        return_message=False,
        transform=AWGN(args.snr_start),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    if args.model_config:
        backbone = build_backbone(args.model_config)
    else:
        channels = [int(c) for c in args.cnn_channels.split(",")]
        kernels = [int(k) for k in args.kernel_sizes.split(",")]
        backbone = ConfigurableCNN(
            seq_len=args.num_samples,
            num_classes=len(train_ds.classes),
            conv_channels=channels,
            kernel_sizes=kernels,
            batch_norm=args.batch_norm,
            activation=args.activation,
            dropout=args.dropout,
        )
    model = AMRClassifier(
        backbone,
        num_classes=len(train_ds.classes),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
    )
    if args.ssl_checkpoint:
        model.load_moco_weights(args.ssl_checkpoint)

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="val_loss"),
        LearningRateMonitor(),
    ]
    if args.ema_decay > 0:
        callbacks.append(EMA(decay=args.ema_decay, use_ema_weights=True))
    callbacks.append(
        SNRCurriculumCallback(
            train_ds, start_snr=args.snr_start, patience=args.snr_patience
        )
    )
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
