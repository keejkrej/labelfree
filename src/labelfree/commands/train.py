"""
Train command for labelfree CLI.

Usage:
    labelfree train -c config.yaml
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from labelfree.data import PhaseToFluorescenceDataModule
from labelfree.lightning_module import FluorescencePredictionModule


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_train(config_path: str, resume_from: str | None = None) -> int:
    """Run training from config file.

    Args:
        config_path: Path to YAML configuration file
        resume_from: Optional path to checkpoint to resume from

    Returns:
        Exit code (0 for success)
    """
    # Load config
    config = load_config(config_path)

    # Set seed
    seed = config.get("seed", 42)
    pl.seed_everything(seed)

    # Create output directory
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data config
    data_config = config.get("data", {})
    data_dir = data_config.get("dir")

    if data_dir is None:
        print("Error: data.dir is required in config")
        return 1

    # Create data module
    datamodule = PhaseToFluorescenceDataModule(
        data_dir=data_dir,
        format="dual_channel_tiff",
        pc_channel=data_config.get("pc_channel", 0),
        fl_channel=data_config.get("fl_channel", 1),
        batch_size=data_config.get("batch_size", 8),
        patch_size=data_config.get("patch_size", 256),
        num_workers=data_config.get("num_workers", 4),
        val_split=data_config.get("val_split", 0.1),
        test_split=data_config.get("test_split", 0.1),
        augment=data_config.get("augment", True),
        seed=seed,
    )

    # Get model config
    model_config = config.get("model", {})

    # Get training config
    train_config = config.get("training", {})

    # Get loss config
    loss_config = config.get("loss", {})
    losses = loss_config.get("functions", ["mse", "ssim"])
    loss_weights = loss_config.get("weights", [0.8, 0.2])

    # Create model
    model = FluorescencePredictionModule(
        architecture=model_config.get("architecture", "attention_unet"),
        in_channels=1,
        out_channels=1,
        base_filters=model_config.get("base_filters", 64),
        depth=model_config.get("depth", 4),
        dropout=model_config.get("dropout", 0.0),
        losses=losses,
        loss_weights=loss_weights,
        optimizer=train_config.get("optimizer", "adamw"),
        learning_rate=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 1e-5),
        scheduler=train_config.get("scheduler", "cosine"),
    )

    # Print model summary
    print(f"\n{'=' * 60}")
    print(f"Model: {model_config.get('architecture', 'attention_unet')}")
    print(f"Parameters: {model.model.get_num_parameters():,}")
    print(f"Data: {data_dir}")
    print(f"{'=' * 60}\n")

    # Create callbacks
    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/ssim:.4f}",
            monitor="val/ssim",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
    ]

    # Add early stopping if enabled
    early_stopping = train_config.get("early_stopping", 20)
    if early_stopping > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss_total",
                patience=early_stopping,
                mode="min",
            )
        )

    # Create logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name=config.get("experiment_name", "fluorescence_prediction"),
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator=train_config.get("accelerator", "auto"),
        devices=train_config.get("devices", 1),
        precision=train_config.get("precision", "32"),
        max_epochs=train_config.get("max_epochs", 100),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Train
    trainer.fit(model, datamodule, ckpt_path=resume_from)

    # Test
    if datamodule.test_dataset is not None and len(datamodule.test_dataset) > 0:
        trainer.test(model, datamodule)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Best validation SSIM: {model.best_val_ssim:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"{'=' * 60}\n")

    return 0
