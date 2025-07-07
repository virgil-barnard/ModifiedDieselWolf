import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import DataLoader

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.data.TransformsRF import AWGN
from dieselwolf.models import AMRClassifier, ConfigurableCNN


def train_cnn(config: dict) -> None:
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

    backbone = ConfigurableCNN(
        seq_len=config["num_samples"],
        num_classes=len(train_ds.classes),
        conv_channels=[config["channels1"], config["channels2"]],
        kernel_sizes=[config["kernel1"], config["kernel2"]],
        dropout=config["dropout"],
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune ConfigurableCNN with Ray Tune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-examples", type=int, default=2**12)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-trials", type=int, default=20)
    p.add_argument("--adv-eps", type=float, default=0.0)
    p.add_argument("--adv-weight", type=float, default=0.5)
    p.add_argument("--adv-norm", type=float, default=float("inf"))
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
        "adv_eps": args.adv_eps,
        "adv_weight": args.adv_weight,
        "adv_norm": args.adv_norm,
        "log_dir": args.log_dir,
    }

    tune.run(
        train_cnn,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=args.max_trials,
        storage_path=args.log_dir,
        name="cnn_tuning",
    )


if __name__ == "__main__":
    main()
