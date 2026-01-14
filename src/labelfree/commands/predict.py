"""
Predict command for labelfree CLI.

Usage:
    labelfree predict -c config.yaml --checkpoint model.ckpt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import tifffile
except ImportError:
    tifffile = None

from labelfree.lightning_module import FluorescencePredictionModule


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_predict(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = "predictions",
) -> int:
    """Run prediction from config and checkpoint.

    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for predictions

    Returns:
        Exit code (0 for success)
    """
    if tifffile is None:
        print("Error: tifffile is required for prediction")
        return 1

    # Load config
    config = load_config(config_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    model = FluorescencePredictionModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get data config
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("dir", "."))
    pc_channel = data_config.get("pc_channel", 0)

    # Find all TIFF files
    tiff_files = sorted(data_dir.glob("*.tif")) + sorted(data_dir.glob("*.tiff"))

    if len(tiff_files) == 0:
        print(f"No TIFF files found in {data_dir}")
        return 1

    print(f"Processing {len(tiff_files)} files...")

    for tiff_path in tiff_files:
        # Load image
        img = tifffile.imread(str(tiff_path))

        # Extract phase contrast channel
        if img.ndim == 3:
            pc = img[pc_channel]
        else:
            pc = img

        # Normalize
        p_low, p_high = np.percentile(pc, [1, 99])
        if p_high - p_low > 1e-6:
            pc_norm = (pc.astype(np.float32) - p_low) / (p_high - p_low)
            pc_norm = np.clip(pc_norm, 0, 1)
        else:
            pc_norm = np.zeros_like(pc, dtype=np.float32)

        # Convert to tensor
        pc_tensor = torch.from_numpy(pc_norm).unsqueeze(0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(pc_tensor)

        # Convert back to numpy
        pred_np = pred.squeeze().cpu().numpy()

        # Save prediction
        output_file = output_path / f"{tiff_path.stem}_predicted.tif"
        tifffile.imwrite(str(output_file), pred_np.astype(np.float32))
        print(f"  Saved: {output_file.name}")

    print(f"\nPredictions saved to {output_path}")
    return 0
