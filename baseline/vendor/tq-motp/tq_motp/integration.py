from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .head import TQMOTPHead
from .types import TQMOTPOutput


def select_sam_decoder_outputs(
    *,
    low_res_masks: Tensor,
    iou_predictions: Tensor,
    mask_tokens: Tensor,
    iou_token: Tensor,
    mask_index: int = 0,
) -> Dict[str, Tensor]:
    """Normalize common SAM1 decoder outputs to the TQ-MOTP input contract."""
    if low_res_masks.ndim != 4:
        raise ValueError("low_res_masks must be [N,M,H,W]")
    if mask_tokens.ndim == 2:
        selected_mask_token = mask_tokens
    elif mask_tokens.ndim == 3:
        selected_mask_token = mask_tokens[:, mask_index]
    else:
        raise ValueError("mask_tokens must be [N,D] or [N,M,D]")
    if iou_predictions.ndim == 1:
        selected_iou = iou_predictions
    elif iou_predictions.ndim == 2:
        selected_iou = iou_predictions[:, mask_index]
    else:
        raise ValueError("iou_predictions must be [N] or [N,M]")
    return {
        "pred_mask_logits": low_res_masks[:, mask_index : mask_index + 1],
        "mask_token": selected_mask_token,
        "iou_token": iou_token,
        "predicted_iou": selected_iou,
    }


def forward_tqmotp_one_image(
    head: TQMOTPHead,
    *,
    image_embedding: Tensor,
    processed_image: Optional[Tensor],
    boxes_1024: Tensor,
    low_res_masks: Tensor,
    iou_predictions: Tensor,
    mask_tokens: Tensor,
    iou_token: Tensor,
    mask_index: int = 0,
) -> TQMOTPOutput:
    """Drop-in helper for the current per-image SAM1 GT-box loop."""
    selected = select_sam_decoder_outputs(
        low_res_masks=low_res_masks,
        iou_predictions=iou_predictions,
        mask_tokens=mask_tokens,
        iou_token=iou_token,
        mask_index=mask_index,
    )
    if image_embedding.ndim == 3:
        image_embedding = image_embedding.unsqueeze(0)
    if processed_image is not None and processed_image.ndim == 3:
        processed_image = processed_image.unsqueeze(0)
    return head(
        image_embeddings=image_embedding,
        input_images=processed_image,
        boxes=boxes_1024,
        batch_indices=torch.zeros(boxes_1024.shape[0], device=boxes_1024.device, dtype=torch.long),
        input_image_hw=tuple(processed_image.shape[-2:]) if processed_image is not None else (1024, 1024),
        **selected,
    )


def tqmotp_optimizer_groups(
    head: TQMOTPHead,
    *,
    head_lr: float,
    prototype_lr: Optional[float] = None,
    weight_decay: float = 1.0e-4,
) -> List[Dict[str, object]]:
    """Separate prototypes/temperatures/norms from decayed head parameters."""
    prototype_lr = head_lr if prototype_lr is None else prototype_lr
    prototype_ids = {id(p) for p in head.prototypes.parameters()}
    no_decay_ids = set(prototype_ids)
    for name, param in head.named_parameters():
        if name.startswith("log_ot_temperature") or name.endswith("bias") or "norm" in name.lower():
            no_decay_ids.add(id(param))

    regular = [p for p in head.parameters() if p.requires_grad and id(p) not in no_decay_ids]
    no_decay = [p for p in head.parameters() if p.requires_grad and id(p) in no_decay_ids and id(p) not in prototype_ids]
    prototypes = [p for p in head.prototypes.parameters() if p.requires_grad]
    groups: List[Dict[str, object]] = []
    if regular:
        groups.append({"params": regular, "lr": head_lr, "weight_decay": weight_decay, "name": "tqmotp_regular"})
    if no_decay:
        groups.append({"params": no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "tqmotp_no_decay"})
    if prototypes:
        groups.append({"params": prototypes, "lr": prototype_lr, "weight_decay": 0.0, "name": "tqmotp_prototypes"})
    return groups
