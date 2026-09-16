from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import TQMOTPConfig
from .roi import (
    BoxesInput,
    aligned_roi,
    canonicalize_boxes,
    expand_and_clip_boxes,
    scale_boxes,
)
from .sampling import RegionDescriptorSampler
from .sinkhorn import partial_sinkhorn_prototype_distance
from .types import PartialOTOutput, RegionDescriptorOutput, TQMOTPOutput

REGIONS: Tuple[str, str, str] = ("fg", "bd", "ctx")


def _logit(value: float) -> float:
    value = min(max(value, 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


class DetailStem(nn.Module):
    """Local image-detail encoder applied after image-space ROIAlign."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        hidden = max(32, out_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8 if out_channels % 8 == 0 else 1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaskShapeEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, mask: Tensor) -> Tensor:
        return self.net(mask)


class ScalarGate(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_logits: Optional[Tensor] = None) -> None:
        super().__init__()
        hidden = max(32, in_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        if init_logits is None:
            nn.init.zeros_(self.net[-1].bias)
        else:
            with torch.no_grad():
                self.net[-1].bias.copy_(init_logits.to(self.net[-1].bias))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TQMOTPHead(nn.Module):
    """Typed, quality-aware, mask-conditioned Partial-OT prototype head.

    Input contract:
      * image_embeddings: [B, C, Hf, Wf]
      * input_images: padded/normalized SAM inputs [B,3,Hin,Win], optional
      * pred_mask_logits: selected low-resolution masks [N,1,Hm,Wm]
      * mask_token / iou_token: raw decoder tokens [N,D]
      * predicted_iou: selected SAM mask-quality prediction [N] or [N,1]
      * boxes: boxes in padded SAM input coordinates
    """

    def __init__(self, config: Optional[TQMOTPConfig] = None) -> None:
        super().__init__()
        self.config = config or TQMOTPConfig()
        self.config.validate()
        c = self.config

        self.semantic_project = nn.Sequential(
            nn.Conv2d(c.image_channels, c.semantic_channels, 1, bias=False),
            nn.GroupNorm(8 if c.semantic_channels % 8 == 0 else 1, c.semantic_channels),
            nn.GELU(),
        )
        self.detail_stem = DetailStem(c.detail_channels) if c.use_detail_stem else None
        detail_in = c.detail_channels if c.use_detail_stem else 0
        self.fuse = nn.Sequential(
            nn.Conv2d(c.semantic_channels + detail_in, c.fused_channels, 1, bias=False),
            nn.GroupNorm(8 if c.fused_channels % 8 == 0 else 1, c.fused_channels),
            nn.GELU(),
            nn.Conv2d(c.fused_channels, c.fused_channels, 3, padding=1, groups=c.fused_channels, bias=False),
            nn.Conv2d(c.fused_channels, c.fused_channels, 1, bias=False),
            nn.GELU(),
        )

        self.samplers = nn.ModuleDict(
            {
                region: RegionDescriptorSampler(
                    feature_dim=c.fused_channels,
                    token_dim=c.token_dim,
                    ot_dim=c.ot_dim,
                    descriptor_count=c.descriptor_counts[region],
                    temperature=c.descriptor_temperature,
                    token_similarity_weight=c.token_similarity_weight,
                    eps=c.region_eps,
                )
                for region in REGIONS
            }
        )

        self.prototypes = nn.ParameterDict()
        for region in REGIONS:
            proto = torch.randn(c.num_classes, c.prototype_counts[region], c.ot_dim)
            proto = F.normalize(proto, dim=-1)
            self.prototypes[region] = nn.Parameter(proto)

        self.log_ot_temperature = nn.ParameterDict(
            {
                region: nn.Parameter(torch.tensor(math.log(math.expm1(c.ot_temperature_init))))
                for region in REGIONS
            }
        )

        self.quality_token = nn.Sequential(
            nn.LayerNorm(c.token_dim),
            nn.Linear(c.token_dim, c.quality_token_dim),
            nn.GELU(),
        )
        quality_dim = c.quality_token_dim + 7
        rho_logits = []
        for region in REGIONS:
            lo, hi = c.rho_ranges[region]
            normalized = (c.rho_init[region] - lo) / (hi - lo)
            rho_logits.append(_logit(normalized))
        self.rho_gate = ScalarGate(quality_dim, 3, torch.tensor(rho_logits))
        self.region_gate = ScalarGate(quality_dim, 3, torch.zeros(3))
        self.fusion_gate = ScalarGate(quality_dim, 1, torch.tensor([_logit(c.fusion_alpha_init)]))

        self.shape_encoder = MaskShapeEncoder(c.shape_dim)
        token_in_dim = 2 * c.token_dim + c.fused_channels + c.shape_dim + 6
        self.token_classifier = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, c.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.classifier_hidden_dim, c.classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.classifier_hidden_dim // 2, c.num_classes),
        )

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def normalized_prototype_bank(self) -> Dict[str, Tensor]:
        return {region: F.normalize(self.prototypes[region], dim=-1) for region in REGIONS}

    @torch.no_grad()
    def load_prototype_bank(self, source: Union[str, Path, Dict[str, Tensor]], strict: bool = True) -> None:
        bank = torch.load(source, map_location="cpu") if isinstance(source, (str, Path)) else source
        if "prototypes" in bank and isinstance(bank["prototypes"], dict):
            bank = bank["prototypes"]
        for region in REGIONS:
            if region not in bank:
                if strict:
                    raise KeyError(f"prototype bank missing region {region}")
                continue
            tensor = torch.as_tensor(bank[region], dtype=self.prototypes[region].dtype)
            if tensor.shape != self.prototypes[region].shape:
                raise ValueError(
                    f"prototype shape for {region}: expected {tuple(self.prototypes[region].shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            self.prototypes[region].copy_(F.normalize(tensor.to(self.prototypes[region].device), dim=-1))

    def _empty_output(self, reference: Tensor) -> TQMOTPOutput:
        c = self.config
        logits = reference.new_empty((0, c.num_classes))
        scalar = reference.new_empty((0, 1))
        return TQMOTPOutput(
            logits=logits,
            ot_logits=logits,
            token_logits=logits,
            ot_distances=logits,
            alpha=scalar,
            region_weights=reference.new_empty((0, 3)),
            rho=reference.new_empty((0, 3)),
            quality_features=reference.new_empty((0, c.quality_token_dim + 7)),
            mask_prob_roi=reference.new_empty((0, 1, c.roi_size, c.roi_size)),
            region_masks={region: reference.new_empty((0, 1, c.roi_size, c.roi_size)) for region in REGIONS},
            descriptor_outputs={},
            region_ot={},
            normalized_prototypes=self.normalized_prototype_bank(),
            diagnostics={},
        )

    def _mask_regions(self, mask_prob: Tensor) -> Dict[str, Tensor]:
        dilated_5 = F.max_pool2d(mask_prob, kernel_size=5, stride=1, padding=2)
        eroded_5 = -F.max_pool2d(-mask_prob, kernel_size=5, stride=1, padding=2)
        boundary = (dilated_5 - eroded_5).clamp(0.0, 1.0)
        dilated_7 = F.max_pool2d(mask_prob, kernel_size=7, stride=1, padding=3)
        context = (1.0 - dilated_7).clamp(0.0, 1.0)
        return {"fg": mask_prob, "bd": boundary, "ctx": context}

    @staticmethod
    def _normalized_entropy(prob: Tensor) -> Tensor:
        p = prob.clamp(1.0e-6, 1.0 - 1.0e-6)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)

    def _mask_shape_stats(self, mask_prob: Tensor, boundary: Tensor) -> Tensor:
        n, _, h, w = mask_prob.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=mask_prob.device, dtype=mask_prob.dtype),
            torch.linspace(-1.0, 1.0, w, device=mask_prob.device, dtype=mask_prob.dtype),
            indexing="ij",
        )
        weight = mask_prob[:, 0]
        mass = weight.sum(dim=(-1, -2)).clamp_min(1.0e-6)
        mean_x = (weight * xx).sum(dim=(-1, -2)) / mass
        mean_y = (weight * yy).sum(dim=(-1, -2)) / mass
        var_x = (weight * (xx - mean_x[:, None, None]).square()).sum(dim=(-1, -2)) / mass
        var_y = (weight * (yy - mean_y[:, None, None]).square()).sum(dim=(-1, -2)) / mass
        elongation = torch.sqrt(torch.maximum(var_x, var_y).clamp_min(1.0e-6)) / torch.sqrt(
            torch.minimum(var_x, var_y).clamp_min(1.0e-6)
        )
        area = weight.mean(dim=(-1, -2))
        perimeter = boundary[:, 0].mean(dim=(-1, -2))
        compactness = area / (perimeter.square() + 1.0e-6)
        entropy = self._normalized_entropy(mask_prob).mean(dim=(-1, -2, -3))
        return torch.stack([area, perimeter, elongation, compactness, entropy, mass / float(h * w)], dim=-1)

    def _quality_features(
        self,
        iou_token: Tensor,
        predicted_iou: Tensor,
        mask_prob: Tensor,
        regions: Dict[str, Tensor],
        descriptors: Dict[str, RegionDescriptorOutput],
    ) -> Tensor:
        if predicted_iou.ndim == 2:
            predicted_iou = predicted_iou.squeeze(-1)
        if self.config.pred_iou_is_logit:
            predicted_iou = predicted_iou.sigmoid()
        predicted_iou = predicted_iou.clamp(0.0, 1.0)
        entropy_map = self._normalized_entropy(mask_prob)
        mask_entropy = entropy_map.mean(dim=(-1, -2, -3))
        bd = regions["bd"]
        boundary_entropy = (entropy_map * bd).sum(dim=(-1, -2, -3)) / bd.sum(dim=(-1, -2, -3)).clamp_min(1.0e-6)
        area = mask_prob.mean(dim=(-1, -2, -3))
        coverages = [descriptors[r].region_coverage.squeeze(-1) for r in REGIONS]
        scalar = torch.stack([predicted_iou, mask_entropy, boundary_entropy, area, *coverages], dim=-1)
        return torch.cat([self.quality_token(iou_token), scalar], dim=-1)

    def _rho_from_quality(self, quality: Tensor) -> Tensor:
        raw = torch.sigmoid(self.rho_gate(quality))
        values = []
        for idx, region in enumerate(REGIONS):
            lo, hi = self.config.rho_ranges[region]
            values.append(lo + (hi - lo) * raw[:, idx])
        return torch.stack(values, dim=-1)

    def _temperature(self, region: str) -> Tensor:
        return F.softplus(self.log_ot_temperature[region]) + self.config.ot_temperature_min

    def forward(
        self,
        *,
        image_embeddings: Tensor,
        input_images: Optional[Tensor],
        pred_mask_logits: Tensor,
        mask_token: Tensor,
        iou_token: Tensor,
        predicted_iou: Tensor,
        boxes: BoxesInput,
        batch_indices: Optional[Tensor] = None,
        input_image_hw: Optional[Tuple[int, int]] = None,
    ) -> TQMOTPOutput:
        c = self.config
        if image_embeddings.ndim != 4:
            raise ValueError("image_embeddings must have shape [B,C,H,W]")
        if pred_mask_logits.ndim == 3:
            pred_mask_logits = pred_mask_logits.unsqueeze(1)
        if pred_mask_logits.ndim != 4 or pred_mask_logits.shape[1] != 1:
            raise ValueError("pred_mask_logits must have shape [N,1,Hm,Wm]")
        if mask_token.ndim != 2 or iou_token.ndim != 2:
            raise ValueError("mask_token and iou_token must have shape [N,D]")
        if mask_token.shape != iou_token.shape or mask_token.shape[1] != c.token_dim:
            raise ValueError("token shapes are incompatible with configuration")

        b = image_embeddings.shape[0]
        if input_image_hw is None:
            if input_images is not None:
                input_image_hw = tuple(input_images.shape[-2:])
            else:
                # SAM default; callers using another input size should pass it explicitly.
                input_image_hw = (1024, 1024)
        input_h, input_w = input_image_hw

        flat_boxes, flat_batch = canonicalize_boxes(
            boxes,
            batch_indices=batch_indices,
            batch_size=b,
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )
        n = flat_boxes.shape[0]
        if n == 0:
            return self._empty_output(image_embeddings)
        for name, tensor in {
            "pred_mask_logits": pred_mask_logits,
            "mask_token": mask_token,
            "iou_token": iou_token,
        }.items():
            if tensor.shape[0] != n:
                raise ValueError(f"{name} has {tensor.shape[0]} instances but boxes have {n}")
        if predicted_iou.shape[0] != n:
            raise ValueError("predicted_iou instance count mismatch")

        expanded = expand_and_clip_boxes(flat_boxes, c.context_scale, input_image_hw)

        sem_map = self.semantic_project(image_embeddings)
        sem_boxes = scale_boxes(expanded, input_image_hw, tuple(sem_map.shape[-2:]))
        sem_roi = aligned_roi(sem_map, sem_boxes, flat_batch, c.roi_size)

        feature_parts = [sem_roi]
        if self.detail_stem is not None:
            if input_images is None:
                detail_roi = sem_roi.new_zeros((n, c.detail_channels, c.roi_size, c.roi_size))
            else:
                if input_images.ndim != 4 or input_images.shape[0] != b or input_images.shape[1] != 3:
                    raise ValueError("input_images must have shape [B,3,H,W]")
                detail_raw = aligned_roi(
                    input_images,
                    expanded,
                    flat_batch,
                    c.roi_size * c.detail_roi_scale,
                )
                detail_roi = self.detail_stem(detail_raw)
                if detail_roi.shape[-2:] != (c.roi_size, c.roi_size):
                    detail_roi = F.interpolate(detail_roi, size=(c.roi_size, c.roi_size), mode="bilinear", align_corners=False)
            feature_parts.append(detail_roi)
        roi_features = self.fuse(torch.cat(feature_parts, dim=1))

        mask_source = pred_mask_logits.detach() if c.detach_mask_for_classification else pred_mask_logits
        mask_boxes = scale_boxes(expanded, input_image_hw, tuple(mask_source.shape[-2:]))
        own_indices = torch.arange(n, device=mask_source.device, dtype=torch.long)
        mask_roi_logits = aligned_roi(mask_source, mask_boxes, own_indices, c.roi_size)
        mask_prob = torch.sigmoid(mask_roi_logits / c.mask_temperature)
        regions = self._mask_regions(mask_prob)

        descriptor_outputs: Dict[str, RegionDescriptorOutput] = {
            region: self.samplers[region](roi_features, regions[region], mask_token)
            for region in REGIONS
        }
        quality = self._quality_features(iou_token, predicted_iou, mask_prob, regions, descriptor_outputs)
        rho = self._rho_from_quality(quality)
        region_weights = torch.softmax(self.region_gate(quality), dim=-1)
        alpha = torch.sigmoid(self.fusion_gate(quality))

        prototypes = self.normalized_prototype_bank()
        region_ot: Dict[str, PartialOTOutput] = {}
        region_logits = []
        for idx, region in enumerate(REGIONS):
            ot = partial_sinkhorn_prototype_distance(
                descriptor_outputs[region].descriptors,
                descriptor_outputs[region].source_mass,
                prototypes[region],
                rho[:, idx : idx + 1],
                epsilon=c.sinkhorn_epsilon,
                iterations=c.sinkhorn_iterations,
                dustbin_cost=c.dustbin_cost,
                tolerance=c.sinkhorn_tolerance,
                return_plan=c.return_transport,
            )
            region_ot[region] = ot
            region_logits.append(-ot.distance / self._temperature(region))

        stacked_region_logits = torch.stack(region_logits, dim=1)  # [N,3,C]
        ot_logits = (region_weights.unsqueeze(-1) * stacked_region_logits).sum(dim=1)
        stacked_distances = torch.stack([region_ot[r].distance for r in REGIONS], dim=1)
        ot_distances = (region_weights.unsqueeze(-1) * stacked_distances).sum(dim=1)

        fg_weight = regions["fg"]
        pooled = (roi_features * fg_weight).sum(dim=(-1, -2)) / fg_weight.sum(dim=(-1, -2)).clamp_min(1.0e-6)
        shape_feature = self.shape_encoder(mask_prob)
        shape_stats = self._mask_shape_stats(mask_prob, regions["bd"])
        token_feature = torch.cat([mask_token, iou_token, pooled, shape_feature, shape_stats], dim=-1)
        token_logits = self.token_classifier(token_feature)
        logits = alpha * ot_logits + (1.0 - alpha) * token_logits

        diagnostics: Dict[str, Tensor] = {
            "roi_feature_mean_abs": roi_features.detach().abs().mean(),
            "mask_area_mean": mask_prob.detach().mean(),
            "alpha_mean": alpha.detach().mean(),
            "rho_mean": rho.detach().mean(dim=0),
            "region_weight_mean": region_weights.detach().mean(dim=0),
        }
        for region in REGIONS:
            ot = region_ot[region]
            if ot.row_error is not None:
                diagnostics[f"{region}_row_error_max"] = ot.row_error.detach().max()
            if ot.col_error is not None:
                diagnostics[f"{region}_col_error_max"] = ot.col_error.detach().max()
            if ot.dustbin_mass is not None:
                diagnostics[f"{region}_dustbin_mass_mean"] = ot.dustbin_mass.detach().mean()
            if ot.entropy is not None:
                diagnostics[f"{region}_transport_entropy_mean"] = ot.entropy.detach().mean()

        return TQMOTPOutput(
            logits=logits,
            ot_logits=ot_logits,
            token_logits=token_logits,
            ot_distances=ot_distances,
            alpha=alpha,
            region_weights=region_weights,
            rho=rho,
            quality_features=quality,
            mask_prob_roi=mask_prob,
            region_masks=regions,
            descriptor_outputs=descriptor_outputs,
            region_ot=region_ot,
            normalized_prototypes=prototypes,
            diagnostics=diagnostics,
        )
