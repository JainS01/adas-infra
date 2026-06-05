"""FusionBaseline — multimodal biometric identification model.

Architecture:
  IrisEncoder        : Conv2d stack → AdaptiveAvgPool → 128-d embedding
  FingerprintEncoder : Conv2d stack → AdaptiveAvgPool → 128-d embedding
  FusionHead         : Concat(256-d) → LayerNorm → Linear(256) → ReLU → Dropout → Linear(classes)

Design rationale:
  - AdaptiveAvgPool makes the spatial resolution of the input irrelevant; the
    model accepts any HxW without recompilation (important for variable-size cameras).
  - LayerNorm (not BatchNorm) before the fusion head stabilises training when the
    two encoders output distributions with different scales. LayerNorm is chosen
    deliberately: it normalises per-sample, so it is batch-size-independent (a
    trailing batch of size 1 would crash BatchNorm1d in train mode) and behaves
    identically in train and eval — no running-statistics drift at inference.
  - The two-tower architecture allows independent per-modality fine-tuning and
    makes it straightforward to add a third modality (e.g., face) later by
    concatenating a third encoder embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU — repeated building block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class IrisEncoder(nn.Module):
    """Shallow CNN encoder for grayscale iris images.

    Input:  (B, 1, H, W) — any resolution
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(1, 32),
            nn.MaxPool2d(2),  # H/2
            _ConvBlock(32, 64),
            _ConvBlock(64, 64),
            nn.MaxPool2d(2),  # H/4
            _ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(128 * 4 * 4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (B, 128, 4, 4)
        feat = feat.flatten(1)  # (B, 128*4*4)
        return self.norm(self.proj(feat))  # type: ignore[no-any-return]  # (B, embed_dim)


class FingerprintEncoder(nn.Module):
    """Shallow CNN encoder for grayscale fingerprint images.

    Input:  (B, 1, H, W) — any resolution
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(1, 32),
            nn.MaxPool2d(2),
            _ConvBlock(32, 64),
            _ConvBlock(64, 64),
            nn.MaxPool2d(2),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(128 * 4 * 4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = feat.flatten(1)
        return self.norm(self.proj(feat))  # type: ignore[no-any-return]


class FusionBaseline(nn.Module):
    """Late-fusion biometric identification model.

    Fuses iris and fingerprint embeddings by concatenation, then classifies
    into *num_classes* subject identities.

    Input:  iris       (B, 1, H_iris, W_iris)
            fingerprint (B, 1, H_fp,   W_fp)
    Output: logits     (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        iris_embed_dim: int = 128,
        fp_embed_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.iris_enc = IrisEncoder(iris_embed_dim)
        self.fp_enc = FingerprintEncoder(fp_embed_dim)

        fused_dim = iris_embed_dim + fp_embed_dim
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),  # batch-size-independent; safe for B=1 batches
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._num_classes = num_classes
        self._init_weights()

    def forward(self, iris: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        iris_feat = self.iris_enc(iris)  # (B, iris_embed_dim)
        fp_feat = self.fp_enc(fingerprint)  # (B, fp_embed_dim)
        fused = torch.cat([iris_feat, fp_feat], dim=-1)  # (B, fused_dim)
        return self.fusion_head(fused)  # type: ignore[no-any-return]  # (B, num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
