"""
Standard UNet architecture for image-to-image translation.

Architecture: Encoder-decoder with skip connections
- Input: (B, 1, H, W) phase contrast
- Output: (B, 1, H, W) predicted fluorescence
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution: (Conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Standard UNet for fluorescence prediction.

    Example:
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(4, 1, 256, 256)  # PC images
        y = model(x)  # Predicted FL
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.depth = depth

        # Filter counts: [64, 128, 256, 512, 1024] for depth=4
        filters = [base_filters * (2**i) for i in range(depth + 1)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(ConvBlock(in_ch, filters[i], dropout if i >= depth - 2 else 0))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = filters[i]

        # Bottleneck
        self.bottleneck = ConvBlock(filters[depth - 1], filters[depth], dropout)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(filters[i + 1], filters[i], 2, stride=2))
            self.decoders.append(ConvBlock(filters[i] * 2, filters[i], dropout if i >= depth - 2 else 0))

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.output(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
