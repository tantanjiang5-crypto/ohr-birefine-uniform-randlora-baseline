#!/usr/bin/env python3
"""Unified SAM1 GT-box model: TQ-MOTP classification plus QJ-2 quality.

This file is a non-training integration program.  It runs the SAM encoder,
prompt encoder and mask decoder once, then consumes the exact selected SAM
mask token and IoU token through two independent heads:

  * TQ-MOTP V1: vehicle-damage class logits.
  * QJ-2: DualToken quality logit and sigmoid quality.

The original project-owned TQ-MOTP implementation is imported directly rather
than reimplemented.  No checkpoint is loaded implicitly and no QJ-2 weight is
bundled.  Call load_state_dict explicitly with compatible model weights.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

LOCAL_CODE = Path(__file__).resolve().parent
ROOT = LOCAL_CODE.parent
TQMOTP_SOURCE = ROOT / "vendor" / "tq-motp"
SAM_SOURCE = ROOT / "vendor" / "segment-anything"
# insert(0) reverses the iterable priority, so keep LOCAL_CODE last: this
# forces the isolated src/ copy ahead of the workspace-wide legacy src/.
for source in (TQMOTP_SOURCE, SAM_SOURCE, ROOT, ROOT / "vendor" / "classification-heads"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
if str(LOCAL_CODE) in sys.path:
    sys.path.remove(str(LOCAL_CODE))
sys.path.insert(0, str(LOCAL_CODE))

from src.models.sam1_wrapper import MASK_TOKEN_INDEX, NUM_DAMAGE_CLASSES, create_model
from tq_motp import TQMOTPConfig, TQMOTPHead, tqmotp_optimizer_groups


def _native_roi_align(feature_map, rois, output_size, spatial_scale=1.0,
                      sampling_ratio=-1, aligned=False):
    """Use torchvision's native ROIAlign op without its Triton fallback.

    torchvision 0.20 routes ROIAlign through a runtime Triton compilation when
    deterministic algorithms are enabled.  CUDA 11.8/Triton cannot compile an
    SM86 image on this host's 470 driver, while the installed native torchvision
    operator is available and has the same ROIAlign arguments.  Calling that
    operator directly keeps the tensor operation on GPU and avoids changing
    data, prompts, targets, thresholds, or evaluation semantics.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return torch.ops.torchvision.roi_align(
        feature_map, rois, float(spatial_scale), int(output_size[0]),
        int(output_size[1]), int(sampling_ratio), bool(aligned),
    )


# `aligned_roi` resolves this module global at call time, so patch only the
# external helper's operator binding; the TQ-MOTP architecture is unchanged.
import tq_motp.roi as _tqmotp_roi
_tqmotp_roi.roi_align = _native_roi_align


@dataclass
class QJ2QualityOutput:
    quality_logit: torch.Tensor
    quality: torch.Tensor
    hidden: torch.Tensor


class DualTokenQualityHead(nn.Module):
    """Exact QJ-2 DualToken quality head from Q-series quality_heads.py."""

    def __init__(self, dim: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.mn = nn.LayerNorm(dim)
        self.qn = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, mask_token: torch.Tensor, iou_token: torch.Tensor) -> QJ2QualityOutput:
        hidden = torch.relu(self.fc1(torch.cat([self.mn(mask_token), self.qn(iou_token)], dim=-1)))
        logit = self.fc2(hidden).squeeze(-1)
        return QJ2QualityOutput(quality_logit=logit, quality=logit.sigmoid(), hidden=hidden)


@dataclass
class TQMOTPQJ2Output:
    pred_masks: torch.Tensor
    original_pred_iou: torch.Tensor
    class_logits: torch.Tensor
    predicted_class: torch.Tensor
    class_probability: torch.Tensor
    qj2_quality_logit: torch.Tensor
    effective_quality: torch.Tensor
    d_score: torch.Tensor
    selected_mask_token: torch.Tensor
    iou_token: torch.Tensor
    tqmotp_output: object | None
    boxes_1024: torch.Tensor
    batch_indices: torch.Tensor


class TQMOTPQJ2SAMModel(nn.Module):
    """One-SAM-forward composition of formal TQ-MOTP V1 and QJ-2.

    Inputs are SAM-normalized/right-bottom-padded images and GT-box prompts in
    the matching SAM padded-1024 coordinate system.  The caller must reuse the
    validated CorrectSamPreprocessor; this module deliberately does not resize
    images or construct mask targets.
    """

    mask_index = MASK_TOKEN_INDEX

    def __init__(self, num_classes: int = NUM_DAMAGE_CLASSES, *, lora_rank: int = 4,
                 lora_alpha: float = 4.0, lora_dropout: float = 0.05,
                 adapter_type: str = "lora", randlora_config: dict | None = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.sam_base = create_model(
            head_name="mask_token", lora_rank=lora_rank,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            num_classes=self.num_classes, adapter_type=adapter_type,
            randlora_config=randlora_config,
        )
        # The constructor creates legacy H1; the unified model never calls it.
        for parameter in self.sam_base.classification_head.parameters():
            parameter.requires_grad = False
        self.adapter_type = str(adapter_type)
        self.tqmotp = TQMOTPHead(TQMOTPConfig(num_classes=self.num_classes))
        self.qj2_quality_head = DualTokenQualityHead()

    @property
    def sam(self):
        return self.sam_base.sam

    def set_lora_dropout(self, enabled: bool) -> None:
        self.sam_base.set_lora_dropout(enabled)

    def forward(self, images: torch.Tensor, boxes_list: Sequence[torch.Tensor]) -> TQMOTPQJ2Output:
        if images.ndim != 4 or images.shape[1:] != (3, 1024, 1024):
            raise ValueError(
                "images must be SAM-normalized/right-bottom-padded [B,3,1024,1024], "
                f"got {tuple(images.shape)}"
            )
        if len(boxes_list) != images.shape[0]:
            raise ValueError("boxes_list must contain exactly one [Ni,4] tensor per input image")
        image_embeddings = self.sam_base.forward_encoder(images)
        masks, pred_iou, mask_tokens, iou_tokens, boxes, batch_indices = [], [], [], [], [], []
        for image_index, boxes_i in enumerate(boxes_list):
            if boxes_i.ndim != 2 or boxes_i.shape[-1] != 4:
                raise ValueError(f"boxes_list[{image_index}] must have shape [Ni,4]")
            if boxes_i.device != images.device:
                raise ValueError(f"boxes_list[{image_index}] must be on {images.device}, got {boxes_i.device}")
            if not boxes_i.numel():
                continue
            sparse, dense = self.sam_base.forward_prompt_encoder(boxes_i)
            decoded = self.sam_base.forward_decoder(
                image_embeddings[image_index:image_index + 1], sparse, dense,
                multimask_output=False,
            )
            selected_mask = decoded["masks"].squeeze(1)
            selected_iou = decoded["iou_pred"].squeeze(-1)
            selected_mask_token = decoded["mask_tokens_out"][:, self.mask_index, :]
            selected_iou_token = decoded["iou_token_out"]
            count = selected_mask.shape[0]
            if not (
                count == selected_iou.shape[0] == selected_mask_token.shape[0]
                == selected_iou_token.shape[0] == boxes_i.shape[0]
            ):
                raise RuntimeError("SAM selected mask/token/IoU instance alignment failure")
            masks.append(selected_mask); pred_iou.append(selected_iou)
            mask_tokens.append(selected_mask_token); iou_tokens.append(selected_iou_token)
            boxes.append(boxes_i)
            batch_indices.append(torch.full((count,), image_index, device=images.device, dtype=torch.long))
        if not masks:
            empty = images.new_empty((0,))
            return TQMOTPQJ2Output(images.new_empty((0, 256, 256)), empty, images.new_empty((0, self.num_classes)), empty.long(), empty, empty, empty, empty, images.new_empty((0, 256)), images.new_empty((0, 256)), None, images.new_empty((0, 4)), empty.long())

        pred_masks = torch.cat(masks); native_iou = torch.cat(pred_iou)
        mask_token = torch.cat(mask_tokens); iou_token = torch.cat(iou_tokens)
        boxes_1024 = torch.cat(boxes); instance_to_image = torch.cat(batch_indices)
        # Sinkhorn/OT stays FP32 as in the formal TQ-MOTP implementation.
        with torch.autocast(device_type=images.device.type, enabled=False):
            tq = self.tqmotp(
                image_embeddings=image_embeddings.float(), input_images=images.float(),
                pred_mask_logits=pred_masks.float().unsqueeze(1), mask_token=mask_token.float(),
                iou_token=iou_token.float(), predicted_iou=native_iou.float(),
                boxes=boxes_1024.float(), batch_indices=instance_to_image,
                input_image_hw=tuple(images.shape[-2:]),
            )
        quality = self.qj2_quality_head(mask_token, iou_token)
        probability = tq.logits.softmax(dim=-1)
        predicted_class = probability.argmax(dim=-1)
        class_probability = probability.gather(1, predicted_class[:, None]).squeeze(1)
        d_score = class_probability.clamp(0, 1).pow(0.5) * quality.quality.clamp(0, 1).pow(1.5)
        return TQMOTPQJ2Output(
            pred_masks=pred_masks, original_pred_iou=native_iou,
            class_logits=tq.logits, predicted_class=predicted_class,
            class_probability=class_probability, qj2_quality_logit=quality.quality_logit,
            effective_quality=quality.quality, d_score=d_score,
            selected_mask_token=mask_token, iou_token=iou_token,
            tqmotp_output=tq, boxes_1024=boxes_1024, batch_indices=instance_to_image,
        )

    def get_param_groups(self, *, lora_lr: float = 1e-4, mask_decoder_lr: float = 5e-5,
                         tq_head_lr: float = 3e-4, prototype_lr: float = 3e-4,
                         qj2_head_lr: float = 1e-4, weight_decay: float = 1e-4):
        groups = self.sam_base.get_param_groups(lora_lr, mask_decoder_lr, tq_head_lr, weight_decay)
        groups += tqmotp_optimizer_groups(self.tqmotp, head_lr=tq_head_lr,
                                          prototype_lr=prototype_lr, weight_decay=weight_decay)
        groups.append({"name": "qj2_quality_head", "params": list(self.qj2_quality_head.parameters()),
                       "lr": qj2_head_lr, "weight_decay": weight_decay})
        seen = set()
        for group in groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
            for parameter in group["params"]:
                if id(parameter) in seen:
                    raise RuntimeError("duplicate optimizer parameter")
                seen.add(id(parameter))
        groups = [group for group in groups if group["params"]]
        # RandLoRA is appended by the trainer with the package's safe helper,
        # after preserving all existing TQ-MOTP/QJ-2 groups verbatim.
        if self.adapter_type != "randlora":
            self.assert_optimizer_parameter_coverage(groups)
        return groups

    def assert_optimizer_parameter_coverage(self, groups) -> None:
        """Assert that every and only trainable parameter enters one group.

        This guards the intended composition: LoRA, SAM mask decoder, TQ-MOTP
        and QJ-2 are trainable; the legacy H1 head and frozen SAM modules are
        excluded.  It can be called before constructing ``torch.optim``.
        """
        grouped = [parameter for group in groups for parameter in group["params"]]
        grouped_ids = [id(parameter) for parameter in grouped]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("optimizer groups contain duplicate parameters")
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if set(grouped_ids) != {id(parameter) for parameter in trainable}:
            missing = len({id(parameter) for parameter in trainable} - set(grouped_ids))
            unexpected = len(set(grouped_ids) - {id(parameter) for parameter in trainable})
            raise RuntimeError(
                "optimizer parameter coverage mismatch: "
                f"missing_trainable={missing}, unexpected_or_frozen={unexpected}"
            )


if __name__ == "__main__":
    model = TQMOTPQJ2SAMModel()
    print({"class": type(model).__name__, "tqmotp_classification": type(model.tqmotp).__name__,
           "qj2_quality": type(model.qj2_quality_head).__name__,
           "d_score": "class_probability**0.5 * qj2_quality**1.5"})
