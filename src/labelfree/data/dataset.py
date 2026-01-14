"""
Dataset for dual-channel TIFF files with phase contrast (PC) and fluorescence (FL).

Training Data Format
====================

This module expects dual-channel TIFF files where:
- Channel 0: Phase contrast (PC) - input
- Channel 1: Fluorescence (FL) - target

File Structure
--------------
```
data/
├── image_001.tif   # Shape: (2, H, W) - [PC, FL]
├── image_002.tif
└── ...
```

Tensor Shapes
-------------
- Input (PC): (B, 1, H, W) - batch of grayscale phase contrast
- Target (FL): (B, 1, H, W) - batch of fluorescence

Example Usage
-------------
```python
dataset = DualChannelTiffDataset(
    data_dir="data/train",
    patch_size=256,
)

sample = dataset[0]
pc = sample["pc"]  # (1, 256, 256)
fl = sample["fl"]  # (1, 256, 256)
```
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def percentile_normalize(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Normalize image using percentile-based scaling to [0, 1]."""
    p_low, p_high = np.percentile(image, [low, high])
    if p_high - p_low < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - p_low) / (p_high - p_low)
    return np.clip(normalized, 0, 1)


class DualChannelTiffDataset(Dataset):
    """Dataset for dual-channel TIFF files (PC + FL).

    Each TIFF file should have shape (2, H, W) where:
    - Channel 0: Phase contrast
    - Channel 1: Fluorescence
    """

    def __init__(
        self,
        data_dir: str | Path,
        pc_channel: int = 0,
        fl_channel: int = 1,
        patch_size: int | None = 256,
        normalize: bool = True,
        augment: bool = False,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing dual-channel TIFF files
            pc_channel: Channel index for phase contrast
            fl_channel: Channel index for fluorescence
            patch_size: Size of random patches (None for full images)
            normalize: Whether to apply percentile normalization
            augment: Whether to apply data augmentation
        """
        if tifffile is None:
            raise ImportError("tifffile is required: pip install tifffile")

        self.data_dir = Path(data_dir)
        self.pc_channel = pc_channel
        self.fl_channel = fl_channel
        self.patch_size = patch_size
        self.normalize = normalize
        self.augment = augment

        # Find all TIFF files
        self.files = sorted(
            list(self.data_dir.glob("*.tif")) + list(self.data_dir.glob("*.tiff"))
        )

        if len(self.files) == 0:
            raise ValueError(f"No TIFF files found in {data_dir}")

        # Setup augmentation
        self.aug_pipeline = None
        if augment and HAS_ALBUMENTATIONS:
            self.aug_pipeline = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                additional_targets={"fl": "image"},
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.

        Returns:
            Dictionary with 'pc', 'fl', and 'name' keys.
        """
        path = self.files[idx]

        # Load dual-channel TIFF
        img = tifffile.imread(str(path))

        # Handle different shapes
        if img.ndim == 2:
            # Single channel - use as both PC and FL (for testing)
            pc = img
            fl = img
        elif img.ndim == 3:
            if img.shape[0] == 2:
                # (2, H, W) format
                pc = img[self.pc_channel]
                fl = img[self.fl_channel]
            elif img.shape[2] == 2:
                # (H, W, 2) format
                pc = img[:, :, self.pc_channel]
                fl = img[:, :, self.fl_channel]
            else:
                raise ValueError(f"Unexpected shape {img.shape} for {path}")
        else:
            raise ValueError(f"Unexpected shape {img.shape} for {path}")

        # Extract random patch
        if self.patch_size is not None:
            pc, fl = self._extract_patch(pc, fl)

        # Normalize
        if self.normalize:
            pc = percentile_normalize(pc)
            fl = percentile_normalize(fl)
        else:
            pc = pc.astype(np.float32)
            fl = fl.astype(np.float32)

        # Augment
        if self.aug_pipeline is not None:
            augmented = self.aug_pipeline(image=pc, fl=fl)
            pc = augmented["image"]
            fl = augmented["fl"]

        # Convert to tensors (add channel dimension)
        pc_tensor = torch.from_numpy(pc.copy()).unsqueeze(0).float()
        fl_tensor = torch.from_numpy(fl.copy()).unsqueeze(0).float()

        return {
            "pc": pc_tensor,
            "fl": fl_tensor,
            "name": path.stem,
        }

    def _extract_patch(
        self, pc: np.ndarray, fl: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract random patch from images."""
        h, w = pc.shape
        ps = self.patch_size

        # Pad if image is smaller than patch
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            pc = np.pad(pc, ((0, pad_h), (0, pad_w)), mode="reflect")
            fl = np.pad(fl, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = pc.shape

        # Random crop
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)

        return pc[y : y + ps, x : x + ps], fl[y : y + ps, x : x + ps]
