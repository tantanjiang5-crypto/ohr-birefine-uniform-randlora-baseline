from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import LayerNorm2d


class CWSAMClasswiseDecoderHead(nn.Module):
    """CWSAM-style classwise mask decoder branch.

    Faithful default structure for SAM transformer_dim=256:
      input upscaled embedding: 32 channels
      ConvTranspose2d(32,32,k=2,s=2)
      LayerNorm2d + GELU
      Conv2d(32,32*num_classes,k=7,s=2,p=3)
      GELU
      mask hypernetwork multiplication -> [B,M,C,H,W]

    The original task is dense semantic segmentation. For GT-box instance
    classification, pool_instance_logits provides a deterministic adaptation.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        transformer_dim: int = 256,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or transformer_dim <= 0 or transformer_dim % 8 != 0:
            raise ValueError("num_classes>1 and transformer_dim divisible by 8 are required")
        self.num_classes = int(num_classes)
        self.transformer_dim = int(transformer_dim)
        self.mask_channel_dim = transformer_dim // 8

        c = self.mask_channel_dim
        self.classwise_upscaling = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
            LayerNorm2d(c),
            activation(),
            nn.Conv2d(c, c * num_classes, kernel_size=7, stride=2, padding=3),
            activation(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match the public CWSAM code: custom fan-out normal init is applied
        # to Conv2d, while ConvTranspose2d keeps PyTorch's default init.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                    // module.groups
                )
                nn.init.normal_(module.weight, mean=0.0, std=(2.0 / fan_out) ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, upscaled_embedding: Tensor, mask_hypernet: Tensor) -> Tensor:
        if upscaled_embedding.ndim != 4:
            raise ValueError(
                f"upscaled_embedding must be [B,C,H,W], got {tuple(upscaled_embedding.shape)}"
            )
        if mask_hypernet.ndim != 3:
            raise ValueError(f"mask_hypernet must be [B,M,C], got {tuple(mask_hypernet.shape)}")
        b, c, h, w = upscaled_embedding.shape
        if c != self.mask_channel_dim:
            raise ValueError(f"Expected {self.mask_channel_dim} channels, got {c}")
        if mask_hypernet.shape[0] != b or mask_hypernet.shape[2] != c:
            raise ValueError(
                "mask_hypernet batch/channel dimensions must match upscaled_embedding"
            )

        classwise_features = self.classwise_upscaling(upscaled_embedding)
        if classwise_features.shape[-2:] != (h, w):
            classwise_features = F.interpolate(
                classwise_features, size=(h, w), mode="bilinear", align_corners=False
            )
        classwise_features = classwise_features.reshape(
            b, c, self.num_classes, h, w
        )
        return torch.einsum("bmc,bckhw->bmkhw", mask_hypernet, classwise_features)

    @staticmethod
    def pool_instance_logits(
        classwise_mask_logits: Tensor,
        *,
        binary_mask_logits: Optional[Tensor] = None,
        mask_token_index: int = 0,
        eps: float = 1e-6,
    ) -> Tensor:
        """Convert CWSAM dense classwise masks to one class logit vector per instance.

        This is an adaptation for GT-box instance classification, not part of the
        original SAR semantic-segmentation task. Soft predicted masks are used as
        spatial weights when binary_mask_logits is supplied.
        """
        if classwise_mask_logits.ndim != 5:
            raise ValueError(
                "classwise_mask_logits must be [B,M,C,H,W], got "
                f"{tuple(classwise_mask_logits.shape)}"
            )
        b, m, _, h, w = classwise_mask_logits.shape
        if not 0 <= mask_token_index < m:
            raise IndexError("mask_token_index outside available mask tokens")
        selected = classwise_mask_logits[:, mask_token_index]

        if binary_mask_logits is None:
            return selected.mean(dim=(-2, -1))
        if binary_mask_logits.ndim == 3:
            binary_mask_logits = binary_mask_logits.unsqueeze(1)
        if binary_mask_logits.shape[:2] != (b, m):
            raise ValueError("binary_mask_logits must be [B,M,H,W] or [B,H,W]")
        weights = binary_mask_logits[:, mask_token_index]
        if weights.shape[-2:] != (h, w):
            weights = F.interpolate(
                weights.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
        weights = weights.sigmoid().unsqueeze(1)
        numerator = (selected * weights).sum(dim=(-2, -1))
        denominator = weights.sum(dim=(-2, -1)).clamp_min(eps)
        return numerator / denominator
