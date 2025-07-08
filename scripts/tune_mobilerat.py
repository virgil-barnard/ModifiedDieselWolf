import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import DataLoader
import torch

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.data.TransformsRF import AWGN
from dieselwolf.models import AMRClassifier, ConfigurableMobileRaT


def train_mobile_rat(config: dict) -> None:
    train_ds = DigitalModulationDataset(
        num_examples=config["num_examples"],
        num_samples=config["num_samples"],
        return_message=False,
        transform=AWGN(20),
    )
    val_ds = DigitalModulationDataset(
        num_examples=max(1, config["num_examples"] // 4),
        num_samples=config["num_samples"],
        return_message=False,
        transform=AWGN(20),
    )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    backbone = ConfigurableMobileRaT(
        seq_len=config["num_samples"],
        num_classes=len(train_ds.classes),
        conv_channels=[config["channels1"], config["channels2"]],
        kernel_sizes=[config["kernel1"], config["kernel2"]],
        dropout=config["dropout"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )
    model = AMRClassifier(
        backbone,
        num_classes=len(train_ds.classes),
        lr=config["lr"],
        adv_eps=config["adv_eps"],
        adv_weight=config["adv_weight"],
        adv_norm=config["adv_norm"],
    )

    logger = TensorBoardLogger(save_dir=config["log_dir"], name="")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                {"val_loss": "val_loss", "val_acc": "val_acc"}, on="validation_end"
            )
        ],
        accelerator="auto",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)


def _float_or_inf(value: str) -> float:
    """Parse a float value that may be 'inf'."""

    if value == "inf":
        return float("inf")
    return float(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune ConfigurableMobileRaT with Ray Tune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-examples", type=int, default=2**12)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-trials", type=int, default=20)
    p.add_argument(
        "--adv-eps",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Adversarial eps values",
    )
    p.add_argument(
        "--adv-weight",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Adversarial loss weight values",
    )
    p.add_argument(
        "--adv-norm",
        type=_float_or_inf,
        nargs="+",
        default=[float("inf"), float(1), float(2)],
        help="Gradient norm types to try",
    )
    p.add_argument("--log-dir", type=str, default="/app/ray_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "epochs": args.epochs,
        "num_examples": args.num_examples,
        "num_samples": args.num_samples,
        "batch_size": tune.choice([16, 32]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "channels1": tune.choice([16, 32, 64]),
        "channels2": tune.choice([32, 64, 128]),
        "kernel1": tune.choice([3, 5]),
        "kernel2": tune.choice([3, 5]),
        "dropout": tune.uniform(0.0, 0.5),
        "nhead": tune.choice([2, 4, 8]),
        "num_layers": tune.choice([1, 2, 3]),
        "adv_eps": tune.choice(args.adv_eps),
        "adv_weight": tune.choice(args.adv_weight),
        "adv_norm": tune.choice(args.adv_norm),
        "log_dir": args.log_dir,
    }

    tune.run(
        train_mobile_rat,
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=args.max_trials,
        storage_path=args.log_dir,
        name="mobilerat_tuning",
    )


if __name__ == "__main__":
    main()
