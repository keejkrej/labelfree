"""
Attention UNet - UNet with attention gates on skip connections.

Attention gates help focus on relevant regions, particularly useful
for sparse nuclear signals in fluorescence prediction.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from labelfree.models.unet import ConvBlock


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""

    def __init__(self, skip_ch: int, gate_ch: int):
        super().__init__()
        inter_ch = skip_ch // 2

        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        theta = self.W_skip(skip)
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=True)
        phi = self.W_gate(gate)
        attention = self.psi(self.relu(theta + phi))
        return skip * attention


class AttentionUNet(nn.Module):
    """Attention UNet for fluorescence prediction.

    Recommended architecture for nuclear fluorescence prediction.
    Better at focusing on sparse signals compared to standard UNet.

    Example:
        model = AttentionUNet(in_channels=1, out_channels=1)
        x = torch.randn(4, 1, 256, 256)
        y = model(x)
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

        # Decoder with attention
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(filters[i + 1], filters[i], 2, stride=2))
            self.attention_gates.append(AttentionGate(filters[i], filters[i + 1]))
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

        # Decoder with attention
        for up, attn, dec, skip in zip(
            self.upconvs, self.attention_gates, self.decoders, reversed(skips)
        ):
            x_up = up(x)
            skip_attn = attn(skip, x)
            if x_up.shape != skip_attn.shape:
                x_up = F.interpolate(x_up, size=skip_attn.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x_up, skip_attn], dim=1)
            x = dec(x)

        return self.output(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
