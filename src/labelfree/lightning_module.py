"""PyTorch Lightning module for fluorescence prediction."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from labelfree.models import UNet, AttentionUNet
from labelfree.losses import CombinedLoss
from labelfree.utils.metrics import compute_metrics


class FluorescencePredictionModule(pl.LightningModule):
    """Lightning module for PC -> FL prediction.

    Example:
        model = FluorescencePredictionModule(architecture="attention_unet")
        trainer = pl.Trainer(max_epochs=100)
        trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        architecture: str = "attention_unet",
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        losses: list[str] | None = None,
        loss_weights: list[float] | None = None,
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler: str = "cosine",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        model_cls = AttentionUNet if architecture == "attention_unet" else UNet
        self.model = model_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            depth=depth,
            dropout=dropout,
        )

        # Loss
        self.loss_fn = CombinedLoss(losses, loss_weights)

        # Config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler

        self.best_val_ssim = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        pc = batch["pc"]
        fl = batch["fl"]

        pred = self.model(pc)
        loss, loss_dict = self.loss_fn(pred, fl)

        # Log
        for name, val in loss_dict.items():
            self.log(f"{stage}/loss_{name}", val, on_epoch=True, prog_bar=(name == "total"))

        with torch.no_grad():
            metrics = compute_metrics(pred, fl)
            for name, val in metrics.items():
                self.log(f"{stage}/{name}", val, on_epoch=True, prog_bar=(name == "ssim"))

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name == "none":
            return {"optimizer": optimizer}

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs or 100,
                eta_min=1e-7,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if self.scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss_total"},
            }

        return {"optimizer": optimizer}

    def on_validation_epoch_end(self) -> None:
        val_ssim = self.trainer.callback_metrics.get("val/ssim")
        if val_ssim is not None and val_ssim > self.best_val_ssim:
            self.best_val_ssim = val_ssim.item()
