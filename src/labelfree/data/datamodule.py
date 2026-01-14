"""
PyTorch Lightning DataModule for dual-channel TIFF data.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from labelfree.data.dataset import DualChannelTiffDataset


class DualChannelDataModule(pl.LightningDataModule):
    """DataModule for dual-channel TIFF files.

    Example config.yaml:
    ```yaml
    data:
      dir: /path/to/data
      pc_channel: 0
      fl_channel: 1
      patch_size: 256
      batch_size: 8
    ```
    """

    def __init__(
        self,
        data_dir: str | Path,
        pc_channel: int = 0,
        fl_channel: int = 1,
        batch_size: int = 8,
        patch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        augment: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.pc_channel = pc_channel
        self.fl_channel = fl_channel
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.augment = augment
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets."""
        # Create full dataset without augmentation for splitting
        full_dataset = DualChannelTiffDataset(
            data_dir=self.data_dir,
            pc_channel=self.pc_channel,
            fl_channel=self.fl_channel,
            patch_size=self.patch_size,
            normalize=True,
            augment=False,
        )

        # Split
        total = len(full_dataset)
        test_size = int(total * self.test_split)
        val_size = int(total * self.val_split)
        train_size = total - val_size - test_size

        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        if stage == "fit" or stage is None:
            # Training dataset with augmentation
            train_files = [full_dataset.files[i] for i in train_subset.indices]
            self.train_dataset = _AugmentedSubset(
                files=train_files,
                pc_channel=self.pc_channel,
                fl_channel=self.fl_channel,
                patch_size=self.patch_size,
                augment=self.augment,
            )
            self.val_dataset = val_subset

        if stage == "test" or stage is None:
            self.test_dataset = test_subset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class _AugmentedSubset(DualChannelTiffDataset):
    """Subset with augmentation enabled."""

    def __init__(self, files: list[Path], **kwargs):
        # Skip parent __init__ file discovery
        self.files = files
        self.data_dir = files[0].parent if files else Path(".")
        self.pc_channel = kwargs.get("pc_channel", 0)
        self.fl_channel = kwargs.get("fl_channel", 1)
        self.patch_size = kwargs.get("patch_size", 256)
        self.normalize = True
        self.augment = kwargs.get("augment", True)

        # Setup augmentation
        self.aug_pipeline = None
        if self.augment:
            try:
                import albumentations as A
                self.aug_pipeline = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                    ],
                    additional_targets={"fl": "image"},
                )
            except ImportError:
                pass


# Alias for backwards compatibility
PhaseToFluorescenceDataModule = DualChannelDataModule
