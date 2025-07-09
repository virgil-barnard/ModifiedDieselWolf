import argparse
import os
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.bayesopt import BayesOptSearch
from dieselwolf.callbacks import ConfusionMatrixCallback, LatentSpaceCallback
from torch.utils.data import DataLoader
import torch

from dieselwolf.data import DigitalModulationDataset
from dieselwolf.data.TransformsRF import AWGN, RandomAWGN
from dieselwolf.models import AMRClassifier, ConfigurableCNN


def train_cnn(config: dict) -> None:
    train_ds = DigitalModulationDataset(
        num_examples=config["num_examples"],
        num_samples=config["num_samples"],
        return_message=False,
        transform=RandomAWGN(0, 30),
    )
    val_ds = DigitalModulationDataset(
        num_examples=max(1, config["num_examples"] // 4),
        num_samples=config["num_samples"],
        return_message=False,
        transform=RandomAWGN(0, 30),
    )
    latent_ds = DigitalModulationDataset(
        num_examples=max(1, config["num_examples"] // 16),
        num_samples=config["num_samples"],
        return_message=False,
        transform=RandomAWGN(0, 30),
    )    
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), num_workers=16)
    latent_loader = DataLoader(latent_ds, batch_size=int(config["batch_size"]), num_workers=16)

    cm_callback = ConfusionMatrixCallback(val_loader, log_tag="val_confusion_matrix")

    act_options = ["relu", "leakyrelu", "tanh"]
    pool_options = ["max", "avg", "lp", "adaptive"]
    backbone = ConfigurableCNN(
        seq_len=config["num_samples"],
        num_classes=len(train_ds.classes),
        conv_channels=[int(config["channels1"]), int(config["channels2"])],
        kernel_sizes=[int(config["kernel1"]), int(config["kernel2"])],
        dropout=config["dropout"],
        activation=act_options[int(config["activation_idx"])],
        pooling=pool_options[int(config["pool_idx"])],
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
    latent_cb = LatentSpaceCallback(
        latent_loader,
        output_dir=os.path.join(logger.log_dir, "latent_space"),
        log_tag="val_latent",
    )
    ckpt_cb = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val_loss")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                {"val_loss": "val_loss", "val_acc": "val_acc"}, on="validation_end"
            ),
            cm_callback,
            latent_cb,
            ckpt_cb,
            EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001),
        ],
        accelerator="auto",
        devices=1,
    )
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_time = time.time() - start_time
    metrics = {
        "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
        "val_acc": float(trainer.callback_metrics.get("val_acc", 0.0)),
        "train_loss": float(trainer.callback_metrics.get("train_loss", 0.0)),
        "train_acc": float(trainer.callback_metrics.get("train_acc", 0.0)),
        "train_time": train_time,
    }
    logger.log_hyperparams(config, metrics)


def _float_or_inf(value: str) -> float:
    """Parse a float value that may be 'inf'."""

    if value == "inf":
        return float("inf")
    return float(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune ConfigurableCNN with Ray Tune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-examples", type=int, default=2**12)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-trials", type=int, default=200)
    p.add_argument(
        "--adv-eps",
        type=float,
        nargs="+",
        default=[0.0, 50.0],
        help="Adversarial eps values",
    )
    p.add_argument(
        "--adv-weight",
        type=float,
        nargs="+",
        default=[0.0, 0.5],
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
    act_options = ["relu", "leakyrelu", "tanh"]
    pool_options = ["max", "avg", "lp", "adaptive"]

    config = {
        "epochs": args.epochs,
        "num_examples": args.num_examples,
        "num_samples": args.num_samples,
        "batch_size": tune.uniform(16, 32),
        "lr": tune.loguniform(1e-4, 1e-2),
        "channels1": tune.uniform(16, 64),
        "channels2": tune.uniform(32, 128),
        "kernel1": tune.uniform(3, 15),
        "kernel2": tune.uniform(3, 15),
        "dropout": tune.uniform(0.0, 0.5),
        "activation_idx": tune.uniform(-0.5, len(act_options) - 0.5),
        "pool_idx": tune.uniform(-0.5, len(pool_options) - 0.5),
        "adv_eps": tune.uniform(min(args.adv_eps), max(args.adv_eps)),
        "adv_weight": tune.uniform(min(args.adv_weight), max(args.adv_weight)),
        "adv_norm": float("inf"),
        "log_dir": args.log_dir,
    }

    try:
        search_alg = BayesOptSearch(
            metric="val_loss", mode="min", random_search_steps=10
        )
    except Exception:
        search_alg = None

    tune.run(
        train_cnn,
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        search_alg=search_alg,
        num_samples=args.max_trials,
        storage_path=args.log_dir,
        name="cnn_tuning",
    )


if __name__ == "__main__":
    main()
