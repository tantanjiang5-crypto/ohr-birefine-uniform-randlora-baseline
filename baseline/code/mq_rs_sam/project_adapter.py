from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .regularizer import MQRSRegularizer, MQRSRegularizerOutput


@dataclass
class FlattenedInstanceTargets:
    labels: Tensor
    gt_masks: Tensor
    batch_indices: Tensor


def flatten_instance_targets(
    class_targets_list: Sequence[Tensor],
    gt_masks_list: Sequence[Tensor],
    *,
    device: torch.device | str,
) -> FlattenedInstanceTargets:
    """Flatten targets in the same image-major, box-order used by the model.

    Each ``class_targets_list[i]`` and ``gt_masks_list[i]`` must follow the
    exact order of ``boxes_list[i]`` passed to ``TQMOTPQJ2SAMModel.forward``.
    GT masks must already be transformed into the selected SAM low-resolution
    mask frame, normally [N_i, 256, 256].
    """

    if len(class_targets_list) != len(gt_masks_list):
        raise ValueError("class_targets_list and gt_masks_list must have the same number of images")
    labels_parts: list[Tensor] = []
    mask_parts: list[Tensor] = []
    batch_parts: list[Tensor] = []
    for image_index, (labels_i, masks_i) in enumerate(zip(class_targets_list, gt_masks_list)):
        labels_i = labels_i.reshape(-1)
        if masks_i.ndim == 4 and masks_i.shape[1] == 1:
            masks_i = masks_i[:, 0]
        if masks_i.ndim != 3:
            raise ValueError(f"gt_masks_list[{image_index}] must be [N,H,W] or [N,1,H,W]")
        if labels_i.numel() != masks_i.shape[0]:
            raise ValueError(
                f"target count mismatch for image {image_index}: "
                f"{labels_i.numel()} labels vs {masks_i.shape[0]} masks"
            )
        if labels_i.numel() == 0:
            continue
        labels_parts.append(labels_i.to(device=device, dtype=torch.long))
        mask_parts.append(masks_i.to(device=device, dtype=torch.float32))
        batch_parts.append(torch.full((labels_i.numel(),), image_index, device=device, dtype=torch.long))

    if not labels_parts:
        return FlattenedInstanceTargets(
            labels=torch.empty(0, device=device, dtype=torch.long),
            gt_masks=torch.empty((0, 256, 256), device=device, dtype=torch.float32),
            batch_indices=torch.empty(0, device=device, dtype=torch.long),
        )
    spatial_shapes = {tuple(part.shape[-2:]) for part in mask_parts}
    if len(spatial_shapes) != 1:
        raise ValueError(f"all GT masks must share a spatial shape, got {sorted(spatial_shapes)}")
    return FlattenedInstanceTargets(
        labels=torch.cat(labels_parts, dim=0),
        gt_masks=torch.cat(mask_parts, dim=0),
        batch_indices=torch.cat(batch_parts, dim=0),
    )


def assert_model_target_alignment(
    *,
    model_batch_indices: Tensor,
    target_batch_indices: Tensor,
    class_logits: Tensor,
    pred_masks: Tensor,
    labels: Tensor,
    gt_masks: Tensor,
) -> None:
    n = class_logits.shape[0]
    if pred_masks.shape[0] != n or labels.numel() != n or gt_masks.shape[0] != n:
        raise RuntimeError(
            "MQ-RS instance alignment failure: class logits, predicted masks, labels and GT masks "
            f"must share N; got {n}, {pred_masks.shape[0]}, {labels.numel()}, {gt_masks.shape[0]}"
        )
    if model_batch_indices.shape != target_batch_indices.shape or not torch.equal(
        model_batch_indices.to(target_batch_indices.device), target_batch_indices
    ):
        raise RuntimeError("MQ-RS image-major target order does not match the model box/token order")
    if tuple(pred_masks.shape[-2:]) != tuple(gt_masks.shape[-2:]):
        raise RuntimeError(
            "MQ-RS spatial target mismatch: GT masks must use the exact low-resolution SAM mask frame; "
            f"got pred {tuple(pred_masks.shape[-2:])} vs GT {tuple(gt_masks.shape[-2:])}"
        )


def add_regularizer_to_base_loss(base_loss: Tensor, regularizer_output: MQRSRegularizerOutput) -> Tensor:
    if base_loss.ndim != 0:
        raise ValueError("base_loss must be a scalar tensor")
    return base_loss + regularizer_output.loss


def merge_loss_dict(
    existing_losses: Mapping[str, Tensor],
    regularizer_output: MQRSRegularizerOutput,
) -> dict[str, Tensor]:
    merged = dict(existing_losses)
    if "loss_mqrs" in merged or "loss_qj2_mqrs_target" in merged:
        raise KeyError("existing loss dictionary already contains an MQ-RS key")
    merged["loss_mqrs"] = regularizer_output.loss_mqrs
    if regularizer_output.loss_qj2.detach().abs().item() != 0:
        merged["loss_qj2_mqrs_target"] = regularizer_output.loss_qj2
    return merged


def compute_for_tqmotp_qj2_output(
    regularizer: MQRSRegularizer,
    model_output: object,
    targets: FlattenedInstanceTargets,
    *,
    global_step: int | None = None,
) -> MQRSRegularizerOutput:
    """Duck-typed bridge for the user's TQMOTPQJ2Output dataclass."""

    required = (
        "class_logits",
        "pred_masks",
        "qj2_quality_logit",
        "batch_indices",
    )
    missing = [name for name in required if not hasattr(model_output, name)]
    if missing:
        raise AttributeError(f"model_output is missing fields required by MQ-RS: {missing}")
    assert_model_target_alignment(
        model_batch_indices=model_output.batch_indices,
        target_batch_indices=targets.batch_indices,
        class_logits=model_output.class_logits,
        pred_masks=model_output.pred_masks,
        labels=targets.labels,
        gt_masks=targets.gt_masks,
    )
    return regularizer(
        class_logits=model_output.class_logits,
        labels=targets.labels,
        pred_mask_logits=model_output.pred_masks,
        gt_masks=targets.gt_masks,
        qj2_quality_logit=model_output.qj2_quality_logit,
        global_step=global_step,
    )
