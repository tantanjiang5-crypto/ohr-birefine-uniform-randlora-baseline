from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import RegionDescriptorOutput


class RegionDescriptorSampler(nn.Module):
    """Mask-token-guided, quota-preserving descriptor sampler for one region type."""

    def __init__(
        self,
        *,
        feature_dim: int,
        token_dim: int,
        ot_dim: int,
        descriptor_count: int,
        temperature: float,
        token_similarity_weight: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.descriptor_count = int(descriptor_count)
        self.temperature = float(temperature)
        self.token_similarity_weight = float(token_similarity_weight)
        self.eps = float(eps)

        self.query = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, ot_dim))
        self.key = nn.Conv2d(feature_dim, ot_dim, kernel_size=1, bias=False)
        self.descriptor = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, ot_dim))

    def forward(self, features: Tensor, region_weights: Tensor, mask_token: Tensor) -> RegionDescriptorOutput:
        if features.ndim != 4:
            raise ValueError("features must have shape [N,C,H,W]")
        if region_weights.ndim == 3:
            region_weights = region_weights.unsqueeze(1)
        if region_weights.ndim != 4 or region_weights.shape[1] != 1:
            raise ValueError("region_weights must have shape [N,1,H,W] or [N,H,W]")
        if features.shape[0] != mask_token.shape[0] or features.shape[0] != region_weights.shape[0]:
            raise ValueError("instance count mismatch")
        if features.shape[-2:] != region_weights.shape[-2:]:
            raise ValueError("feature and region spatial sizes must match")

        n, channels, h, w = features.shape
        k = min(self.descriptor_count, h * w)
        if n == 0:
            return RegionDescriptorOutput(
                descriptors=features.new_empty((0, k, self.descriptor[-1].out_features)),
                source_mass=features.new_empty((0, k)),
                selected_indices=torch.empty((0, k), device=features.device, dtype=torch.long),
                selected_region_weights=features.new_empty((0, k)),
                region_coverage=features.new_empty((0, 1)),
            )

        feat_flat = features.flatten(2).transpose(1, 2)  # [N,HW,C]
        key_flat = F.normalize(self.key(features).flatten(2).transpose(1, 2), dim=-1)
        query = F.normalize(self.query(mask_token), dim=-1)
        similarity = torch.einsum("nd,nhd->nh", query, key_flat)

        weight_flat = region_weights.flatten(1).clamp(0.0, 1.0)
        score = torch.log(weight_flat + self.eps) + self.token_similarity_weight * similarity
        top_score, top_idx = torch.topk(score, k=k, dim=1, largest=True, sorted=True)

        gather_feat = top_idx.unsqueeze(-1).expand(-1, -1, channels)
        selected_raw = torch.gather(feat_flat, 1, gather_feat)
        descriptors = F.normalize(self.descriptor(selected_raw), dim=-1)
        selected_weights = torch.gather(weight_flat, 1, top_idx)

        mass_logits = top_score / max(self.temperature, self.eps)
        source_mass = torch.softmax(mass_logits, dim=-1)
        # Keep gradients finite when all region weights are approximately zero.
        source_mass = source_mass.clamp_min(self.eps)
        source_mass = source_mass / source_mass.sum(dim=-1, keepdim=True)

        return RegionDescriptorOutput(
            descriptors=descriptors,
            source_mass=source_mass,
            selected_indices=top_idx,
            selected_region_weights=selected_weights,
            region_coverage=weight_flat.mean(dim=-1, keepdim=True),
        )
