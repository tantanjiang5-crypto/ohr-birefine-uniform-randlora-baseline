#!/usr/bin/env python3
"""TQMOTP alpha=.25 wrapper using the isolated Final-DAG RandLoRA runtime.

The validated TQMOTP/QJ-2 implementation is retained verbatim.  Only the
``create_model`` factory is redirected to this run's DAG-enabled SAM wrapper,
which supports the per-block rank allocation while preserving the same
forward graph and shared-gradient scaling used by F0.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import torch

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parent
ORIGINAL = LOCAL / "TQMOTP_QJ2_UNIFIED.py"

# Preload the isolated DAG package before importing the legacy unified module;
# its SAM wrapper then resolves the same package/class objects as the profiler.
_dag_vendor = LOCAL.parent / "vendor" / "DAG-RandLoRA-v1.0.0"
if str(_dag_vendor) in sys.path:
    sys.path.remove(str(_dag_vendor))
sys.path.insert(0, str(_dag_vendor))
import randlora_damage  # noqa: F401,E402


def _load_original():
    name = "tqmotp_qj2_original_for_final_dag"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ORIGINAL)
    if spec is None or spec.loader is None:
        raise ImportError(ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = _load_original()

# Load the DAG-enabled SAM wrapper from this isolated experiment.  The parent
# TQMOTP class resolves ``create_model`` in its defining module, so replacing
# that symbol for construction is sufficient and leaves all head/loss code
# unchanged.
_src_name = "final_dag_sam1_wrapper"
_src_path = LOCAL / "src" / "models" / "sam1_wrapper.py"
_src_spec = importlib.util.spec_from_file_location(_src_name, _src_path)
if _src_spec is None or _src_spec.loader is None:
    raise ImportError(_src_path)
_src_mod = importlib.util.module_from_spec(_src_spec)
sys.modules[_src_name] = _src_mod
_src_spec.loader.exec_module(_src_mod)
_dag_create_model = _src_mod.create_model


def _scale(value: torch.Tensor, alpha: float) -> torch.Tensor:
    detached = value.detach()
    return detached + float(alpha) * (value - detached)


class TQMOTPQJ2SAMModel(_base.TQMOTPQJ2SAMModel):
    def __init__(self, *args, **kwargs) -> None:
        # The shared project factory forwards these protocol markers; the
        # original unified constructor predates them, so consume them here.
        kwargs.pop("classification_gradient_mode", None)
        kwargs.pop("classification_gradient_alpha", None)
        kwargs.pop("quality_gradient_mode", None)
        kwargs.pop("enable_quality_head", None)
        kwargs.pop("use_qj2_quality", None)
        mode = os.environ.get("TQMOTP_CLASSIFICATION_GRADIENT_MODE", "scaled")
        alpha_text = os.environ.get("TQMOTP_CLASSIFICATION_GRADIENT_ALPHA", "0.25")
        quality_mode = os.environ.get("TQMOTP_QUALITY_GRADIENT_MODE", "detached")
        if mode != "scaled" or quality_mode != "detached":
            raise RuntimeError("Final-DAG requires scaled TQMOTP alpha=.25 and detached QJ-2")
        alpha = float(alpha_text)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"invalid classification gradient alpha {alpha}")
        old_factory = _base.create_model
        _base.create_model = _dag_create_model
        try:
            super().__init__(*args, **kwargs)
        finally:
            _base.create_model = old_factory
        self.classification_gradient_mode = mode
        self.classification_gradient_alpha = alpha
        self.quality_gradient_mode = quality_mode

    def forward(self, images: torch.Tensor, boxes_list: Sequence[torch.Tensor]):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 1024, 1024):
            raise ValueError(f"images must be [B,3,1024,1024], got {tuple(images.shape)}")
        if len(boxes_list) != images.shape[0]:
            raise ValueError("boxes_list must contain one tensor per image")
        image_embeddings = self.sam_base.forward_encoder(images)
        masks, pred_iou, mask_tokens, iou_tokens, boxes, batch_indices = [], [], [], [], [], []
        for image_index, boxes_i in enumerate(boxes_list):
            if boxes_i.ndim != 2 or boxes_i.shape[-1] != 4:
                raise ValueError(f"boxes_list[{image_index}] must be [N,4]")
            if boxes_i.device != images.device:
                raise ValueError("box/device mismatch")
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
            if not (count == selected_iou.shape[0] == selected_mask_token.shape[0] == selected_iou_token.shape[0] == boxes_i.shape[0]):
                raise RuntimeError("SAM output alignment failure")
            masks.append(selected_mask); pred_iou.append(selected_iou)
            mask_tokens.append(selected_mask_token); iou_tokens.append(selected_iou_token)
            boxes.append(boxes_i)
            batch_indices.append(torch.full((count,), image_index, device=images.device, dtype=torch.long))
        if not masks:
            empty = images.new_empty((0,))
            return _base.TQMOTPQJ2Output(
                images.new_empty((0, 256, 256)), empty,
                images.new_empty((0, self.num_classes)), empty.long(), empty, empty, empty,
                empty, images.new_empty((0, 256)), images.new_empty((0, 256)), None,
                images.new_empty((0, 4)), empty.long(),
            )
        pred_masks = torch.cat(masks); native_iou = torch.cat(pred_iou)
        mask_token = torch.cat(mask_tokens); iou_token = torch.cat(iou_tokens)
        boxes_1024 = torch.cat(boxes); instance_to_image = torch.cat(batch_indices)
        alpha = self.classification_gradient_alpha
        with torch.autocast(device_type=images.device.type, enabled=False):
            tq = self.tqmotp(
                image_embeddings=_scale(image_embeddings, alpha).float(),
                input_images=images.float(),
                pred_mask_logits=_scale(pred_masks, alpha).float().unsqueeze(1),
                mask_token=_scale(mask_token, alpha).float(),
                iou_token=_scale(iou_token, alpha).float(),
                predicted_iou=_scale(native_iou, alpha).float(),
                boxes=boxes_1024.float(), batch_indices=instance_to_image,
                input_image_hw=tuple(images.shape[-2:]),
            )
        quality = self.qj2_quality_head(mask_token.detach(), iou_token.detach())
        probability = tq.logits.softmax(dim=-1)
        predicted_class = probability.argmax(dim=-1)
        class_probability = probability.gather(1, predicted_class[:, None]).squeeze(1)
        d_score = class_probability.clamp(0, 1).pow(0.5) * quality.quality.clamp(0, 1).pow(1.5)
        return _base.TQMOTPQJ2Output(
            pred_masks=pred_masks, original_pred_iou=native_iou,
            class_logits=tq.logits, predicted_class=predicted_class,
            class_probability=class_probability, qj2_quality_logit=quality.quality_logit,
            effective_quality=quality.quality, d_score=d_score,
            selected_mask_token=mask_token, iou_token=iou_token,
            tqmotp_output=tq, boxes_1024=boxes_1024, batch_indices=instance_to_image,
        )


def build_project_model_class(project_root=None):
    return TQMOTPQJ2SAMModel
