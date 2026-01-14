"""Evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute MSE, MAE, PSNR, SSIM."""
    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        psnr = 10 * torch.log10(1.0 / max(mse, 1e-10)).item()

        # Simple SSIM
        mu_p = pred.mean()
        mu_t = target.mean()
        sigma_p = pred.var()
        sigma_t = target.var()
        sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2))
        ssim = ssim.item()

    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}
