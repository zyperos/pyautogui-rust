"""Compact reference-conditioned network used by the product runtime.

PyTorch is a training/export dependency only. Production inference consumes
the custom TLN weight stream from the Rust extension and never imports this
module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, source: int, target: int, kernel: int, stride: int = 1, groups: int = 1) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(source, target, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(target),
            nn.Hardswish(inplace=True),
        )


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(value, 1)
        scale = F.hardswish(self.reduce(scale))
        return value * torch.sigmoid(self.expand(scale))


class InvertedResidual(nn.Module):
    def __init__(self, source: int, target: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden = source * expansion
        self.residual = stride == 1 and source == target
        self.expand = ConvNormAct(source, hidden, 1) if expansion != 1 else nn.Identity()
        self.depthwise = ConvNormAct(hidden, hidden, 3, stride, groups=hidden)
        self.attention = SqueezeExcite(hidden)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, target, 1, bias=False),
            nn.BatchNorm2d(target),
        )

    def forward(self, value: Tensor) -> Tensor:
        result = self.project(self.attention(self.depthwise(self.expand(value))))
        return value + result if self.residual else result


class TinyLocateEncoder(nn.Module):
    """Shared 0.8M parameter encoder with stride-8 feature output."""

    def __init__(self, feature_channels: int = 64) -> None:
        super().__init__()
        self.stem = ConvNormAct(3, 24, 3, 2)
        self.stage1 = nn.Sequential(
            InvertedResidual(24, 32, 2, 2),
            InvertedResidual(32, 32, 1, 3),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 64, 2, 3),
            InvertedResidual(64, 64, 1, 3),
            InvertedResidual(64, 64, 1, 3),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(64, 96, 1, 3),
            InvertedResidual(96, 96, 1, 3),
        )
        self.project = nn.Conv2d(96, feature_channels, 1, bias=False)

    def forward(self, image: Tensor) -> Tensor:
        value = self.stage1(self.stem(image))
        value = self.stage2(value)
        value = self.stage3(value)
        return F.normalize(self.project(value), dim=1, eps=1e-6)


class TinyLocateNet(nn.Module):
    """One-shot visual instance locator for arbitrary reference categories."""

    def __init__(self, feature_channels: int = 64) -> None:
        super().__init__()
        self.encoder = TinyLocateEncoder(feature_channels)
        self.objectness = nn.Sequential(
            ConvNormAct(feature_channels + 1, feature_channels, 3),
            nn.Conv2d(feature_channels, 1, 1),
        )
        self.box = nn.Sequential(
            ConvNormAct(feature_channels + 1, feature_channels, 3),
            nn.Conv2d(feature_channels, 4, 1),
        )

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def forward(self, reference: Tensor, search: Tensor) -> dict[str, Tensor]:
        reference_features = self.encoder(reference)
        search_features = self.encoder(search)
        query = F.normalize(F.adaptive_avg_pool2d(reference_features, 1), dim=1, eps=1e-6)
        correlation = (search_features * query).sum(dim=1, keepdim=True)
        fused = torch.cat((search_features, correlation), dim=1)
        return {
            "objectness": self.objectness(fused),
            "box": self.box(fused),
            "features": search_features,
            "correlation": correlation,
        }

