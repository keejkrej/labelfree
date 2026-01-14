"""Loss functions for fluorescence prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """Structural Similarity loss (1 - SSIM)."""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-coords**2 / (2 * sigma**2))
        gauss /= gauss.sum()
        window = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        window = self.window.expand(channels, 1, -1, -1).to(pred.device)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_t = F.conv2d(target, window, padding=pad, groups=channels)

        mu_p_sq, mu_t_sq = mu_p**2, mu_t**2
        mu_pt = mu_p * mu_t

        sigma_p = F.conv2d(pred**2, window, padding=pad, groups=channels) - mu_p_sq
        sigma_t = F.conv2d(target**2, window, padding=pad, groups=channels) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

        C1, C2 = 0.01**2, 0.03**2
        ssim = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
        ssim /= (mu_p_sq + mu_t_sq + C1) * (sigma_p + sigma_t + C2)

        return 1 - ssim.mean()


class CombinedLoss(nn.Module):
    """Combined loss with configurable weights.

    Default: 0.8 * MSE + 0.2 * SSIM
    """

    def __init__(
        self,
        losses: list[str] | None = None,
        weights: list[float] | None = None,
    ):
        super().__init__()
        losses = losses or ["mse", "ssim"]
        weights = weights or [0.8, 0.2]

        loss_map = {"mse": MSELoss, "mae": MAELoss, "ssim": SSIMLoss}

        self.names = losses
        self.weights = weights
        self.loss_fns = nn.ModuleList([loss_map[n]() for n in losses])

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.tensor(0.0, device=pred.device)
        losses = {}
        for name, weight, fn in zip(self.names, self.weights, self.loss_fns):
            loss = fn(pred, target)
            losses[name] = loss
            total = total + weight * loss
        losses["total"] = total
        return total, losses
