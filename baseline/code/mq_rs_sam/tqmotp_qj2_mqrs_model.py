#!/usr/bin/env python3
"""Revised project integration: SAM1 + TQ-MOTP + QJ-2, with MQ-RS external.

MQ-RS is deliberately not executed in ``forward`` because it requires GT masks
and labels and must never be part of inference. Use ``MQRSRegularizer`` from the
training loop. This file removes hard-coded import side effects where possible,
keeps one SAM forward, and exposes configurable D-score fusion exponents.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .fusion import fuse_class_and_quality_scores


def configure_project_imports(project_root: str | Path | None = None) -> Path:
    root = Path(project_root or os.environ.get("SAM44_ROOT", "/workspace/sam44")).resolve()
    sources = (
        root,
        root / "models/segment-anything",
        root / "external/tq_motp_head_code/tq_motp_head_code",
    )
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project source path(s) do not exist: " + ", ".join(str(path) for path in missing)
        )
    for source in sources:
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    return root


@dataclass
class QJ2QualityOutput:
    quality_logit: torch.Tensor
    quality: torch.Tensor
    hidden: torch.Tensor


class DualTokenQualityHead(nn.Module):
    """QJ-2 DualToken quality head."""

    def __init__(self, dim: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.mask_norm = nn.LayerNorm(dim)
        self.iou_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, mask_token: torch.Tensor, iou_token: torch.Tensor) -> QJ2QualityOutput:
        if mask_token.shape != iou_token.shape:
            raise ValueError(
                f"mask_token and iou_token must be shape-aligned, got {mask_token.shape} and {iou_token.shape}"
            )
        hidden = torch.relu(
            self.fc1(torch.cat([self.mask_norm(mask_token), self.iou_norm(iou_token)], dim=-1))
        )
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


def build_project_model_class(project_root: str | Path | None = None):
    """Create the project-bound model class after validating import paths."""

    configure_project_imports(project_root)
    from src.models.sam1_wrapper import MASK_TOKEN_INDEX, create_model
    from tq_motp import TQMOTPConfig, TQMOTPHead, tqmotp_optimizer_groups

    class TQMOTPQJ2SAMModel(nn.Module):
        mask_index = MASK_TOKEN_INDEX

        def __init__(
            self,
            num_classes: int = 8,
            *,
            lora_rank: int = 4,
            lora_alpha: float = 4.0,
            lora_dropout: float = 0.05,
            d_score_class_exponent: float = 0.5,
            d_score_quality_exponent: float = 1.5,
        ) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.d_score_class_exponent = d_score_class_exponent
            self.d_score_quality_exponent = d_score_quality_exponent
            self.sam_base = create_model(
                head_name="mask_token",
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            # Legacy H1 is constructed by the wrapper but never used here.
            for parameter in self.sam_base.classification_head.parameters():
                parameter.requires_grad = False
            self.tqmotp = TQMOTPHead(TQMOTPConfig(num_classes=num_classes))
            self.qj2_quality_head = DualTokenQualityHead()

        @property
        def sam(self):
            return self.sam_base.sam

        def set_lora_dropout(self, enabled: bool) -> None:
            self.sam_base.set_lora_dropout(enabled)

        def _empty_output(self, images: torch.Tensor) -> TQMOTPQJ2Output:
            scalar_empty = images.new_empty((0,))
            return TQMOTPQJ2Output(
                pred_masks=images.new_empty((0, 256, 256)),
                original_pred_iou=scalar_empty,
                class_logits=images.new_empty((0, self.num_classes)),
                predicted_class=torch.empty(0, device=images.device, dtype=torch.long),
                class_probability=scalar_empty,
                qj2_quality_logit=scalar_empty,
                effective_quality=scalar_empty,
                d_score=scalar_empty,
                selected_mask_token=images.new_empty((0, 256)),
                iou_token=images.new_empty((0, 256)),
                tqmotp_output=None,
                boxes_1024=images.new_empty((0, 4)),
                batch_indices=torch.empty(0, device=images.device, dtype=torch.long),
            )

        def forward(self, images: torch.Tensor, boxes_list: Sequence[torch.Tensor]) -> TQMOTPQJ2Output:
            if len(boxes_list) != images.shape[0]:
                raise ValueError("boxes_list length must equal image batch size")
            image_embeddings = self.sam_base.forward_encoder(images)
            masks: list[torch.Tensor] = []
            pred_iou: list[torch.Tensor] = []
            mask_tokens: list[torch.Tensor] = []
            iou_tokens: list[torch.Tensor] = []
            boxes: list[torch.Tensor] = []
            batch_indices: list[torch.Tensor] = []

            for image_index, boxes_i in enumerate(boxes_list):
                if boxes_i.ndim != 2 or boxes_i.shape[-1] != 4:
                    raise ValueError(f"boxes_list[{image_index}] must have shape [N,4]")
                if not boxes_i.numel():
                    continue
                sparse, dense = self.sam_base.forward_prompt_encoder(boxes_i)
                decoded = self.sam_base.forward_decoder(
                    image_embeddings[image_index : image_index + 1],
                    sparse,
                    dense,
                    multimask_output=False,
                )
                selected_mask = decoded["masks"].squeeze(1)
                selected_iou = decoded["iou_pred"].squeeze(-1)
                selected_mask_token = decoded["mask_tokens_out"][:, self.mask_index, :]
                selected_iou_token = decoded["iou_token_out"]
                count = selected_mask.shape[0]
                if not (
                    count
                    == selected_iou.shape[0]
                    == selected_mask_token.shape[0]
                    == selected_iou_token.shape[0]
                    == boxes_i.shape[0]
                ):
                    raise RuntimeError("SAM selected mask/token/IoU/box instance alignment failure")
                masks.append(selected_mask)
                pred_iou.append(selected_iou)
                mask_tokens.append(selected_mask_token)
                iou_tokens.append(selected_iou_token)
                boxes.append(boxes_i)
                batch_indices.append(
                    torch.full((count,), image_index, device=images.device, dtype=torch.long)
                )

            if not masks:
                return self._empty_output(images)

            pred_masks = torch.cat(masks, dim=0)
            native_iou = torch.cat(pred_iou, dim=0)
            mask_token = torch.cat(mask_tokens, dim=0)
            iou_token = torch.cat(iou_tokens, dim=0)
            boxes_1024 = torch.cat(boxes, dim=0)
            instance_to_image = torch.cat(batch_indices, dim=0)

            with torch.autocast(device_type=images.device.type, enabled=False):
                tq = self.tqmotp(
                    image_embeddings=image_embeddings.float(),
                    input_images=images.float(),
                    pred_mask_logits=pred_masks.float().unsqueeze(1),
                    mask_token=mask_token.float(),
                    iou_token=iou_token.float(),
                    predicted_iou=native_iou.float(),
                    boxes=boxes_1024.float(),
                    batch_indices=instance_to_image,
                    input_image_hw=tuple(images.shape[-2:]),
                )

            quality = self.qj2_quality_head(mask_token, iou_token)
            probability = tq.logits.float().softmax(dim=-1)
            predicted_class = probability.argmax(dim=-1)
            class_probability = probability.gather(1, predicted_class[:, None]).squeeze(1)
            d_score = fuse_class_and_quality_scores(
                class_probability,
                quality.quality,
                class_exponent=self.d_score_class_exponent,
                quality_exponent=self.d_score_quality_exponent,
            )
            return TQMOTPQJ2Output(
                pred_masks=pred_masks,
                original_pred_iou=native_iou,
                class_logits=tq.logits,
                predicted_class=predicted_class,
                class_probability=class_probability,
                qj2_quality_logit=quality.quality_logit,
                effective_quality=quality.quality,
                d_score=d_score,
                selected_mask_token=mask_token,
                iou_token=iou_token,
                tqmotp_output=tq,
                boxes_1024=boxes_1024,
                batch_indices=instance_to_image,
            )

        def get_param_groups(
            self,
            *,
            lora_lr: float = 1e-4,
            mask_decoder_lr: float = 5e-5,
            tq_head_lr: float = 3e-4,
            prototype_lr: float = 3e-4,
            qj2_head_lr: float = 1e-4,
            weight_decay: float = 1e-4,
        ):
            groups = self.sam_base.get_param_groups(
                lora_lr, mask_decoder_lr, tq_head_lr, weight_decay
            )
            groups += tqmotp_optimizer_groups(
                self.tqmotp,
                head_lr=tq_head_lr,
                prototype_lr=prototype_lr,
                weight_decay=weight_decay,
            )
            groups.append(
                {
                    "name": "qj2_quality_head",
                    "params": list(self.qj2_quality_head.parameters()),
                    "lr": qj2_head_lr,
                    "weight_decay": weight_decay,
                }
            )
            seen: set[int] = set()
            clean_groups = []
            for group in groups:
                trainable = [parameter for parameter in group["params"] if parameter.requires_grad]
                for parameter in trainable:
                    if id(parameter) in seen:
                        raise RuntimeError(f"duplicate optimizer parameter in group {group.get('name')}")
                    seen.add(id(parameter))
                if trainable:
                    clean_group = dict(group)
                    clean_group["params"] = trainable
                    clean_groups.append(clean_group)
            return clean_groups

    return TQMOTPQJ2SAMModel


if __name__ == "__main__":
    Model = build_project_model_class()
    model = Model()
    print(
        {
            "class": type(model).__name__,
            "tqmotp_classification": type(model.tqmotp).__name__,
            "qj2_quality": type(model.qj2_quality_head).__name__,
            "mqrs": "training-only external regularizer",
            "d_score": (
                f"class_probability**{model.d_score_class_exponent} * "
                f"qj2_quality**{model.d_score_quality_exponent}"
            ),
        }
    )
