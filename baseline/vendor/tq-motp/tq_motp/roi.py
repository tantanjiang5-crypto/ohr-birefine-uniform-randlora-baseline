from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchvision.ops import roi_align

BoxesInput = Union[Tensor, Sequence[Tensor]]


def canonicalize_boxes(
    boxes: BoxesInput,
    *,
    batch_indices: Optional[Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Return flattened boxes [N,4] and batch indices [N]."""
    if isinstance(boxes, Tensor):
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("tensor boxes must have shape [N,4]")
        flat = boxes.to(device=device, dtype=dtype)
        if batch_indices is None:
            if batch_size != 1:
                raise ValueError("batch_indices are required for tensor boxes when B > 1")
            indices = torch.zeros(flat.shape[0], device=device, dtype=torch.long)
        else:
            indices = batch_indices.to(device=device, dtype=torch.long)
            if indices.shape != (flat.shape[0],):
                raise ValueError("batch_indices must have shape [N]")
        return flat, indices

    if len(boxes) != batch_size:
        raise ValueError(f"boxes list length {len(boxes)} != batch size {batch_size}")
    flat_list: List[Tensor] = []
    idx_list: List[Tensor] = []
    for batch_idx, current in enumerate(boxes):
        current = torch.as_tensor(current, device=device, dtype=dtype)
        if current.numel() == 0:
            continue
        if current.ndim != 2 or current.shape[-1] != 4:
            raise ValueError("each boxes-list entry must have shape [Ni,4]")
        flat_list.append(current)
        idx_list.append(torch.full((current.shape[0],), batch_idx, device=device, dtype=torch.long))
    if not flat_list:
        return (
            torch.empty((0, 4), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )
    return torch.cat(flat_list, dim=0), torch.cat(idx_list, dim=0)


def expand_and_clip_boxes(boxes: Tensor, scale: float, image_hw: Tuple[int, int]) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    if scale <= 0:
        raise ValueError("scale must be positive")
    h, w = image_hw
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1).clamp_min(1.0) * scale
    bh = (y2 - y1).clamp_min(1.0) * scale
    out = torch.stack(
        [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], dim=-1
    )
    out[:, 0::2] = out[:, 0::2].clamp(0.0, float(w))
    out[:, 1::2] = out[:, 1::2].clamp(0.0, float(h))
    # Avoid zero-area ROIAlign inputs.
    out[:, 2] = torch.maximum(out[:, 2], out[:, 0] + 1.0)
    out[:, 3] = torch.maximum(out[:, 3], out[:, 1] + 1.0)
    out[:, 2] = out[:, 2].clamp_max(float(w))
    out[:, 3] = out[:, 3].clamp_max(float(h))
    return out


def scale_boxes(boxes: Tensor, from_hw: Tuple[int, int], to_hw: Tuple[int, int]) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    scaled = boxes.clone()
    scaled[:, 0::2] *= float(to_w) / float(from_w)
    scaled[:, 1::2] *= float(to_h) / float(from_h)
    return scaled


def make_rois(boxes: Tensor, batch_indices: Tensor) -> Tensor:
    if boxes.shape[0] != batch_indices.shape[0]:
        raise ValueError("boxes and batch_indices disagree")
    if boxes.numel() == 0:
        return boxes.new_empty((0, 5))
    return torch.cat([batch_indices.to(boxes.dtype).unsqueeze(1), boxes], dim=1)


def aligned_roi(
    feature_map: Tensor,
    boxes_in_feature_coords: Tensor,
    batch_indices: Tensor,
    output_size: Union[int, Tuple[int, int]],
) -> Tensor:
    """ROIAlign where the boxes are already expressed in feature coordinates."""
    rois = make_rois(boxes_in_feature_coords, batch_indices)
    if rois.numel() == 0:
        if isinstance(output_size, int):
            oh = ow = output_size
        else:
            oh, ow = output_size
        return feature_map.new_empty((0, feature_map.shape[1], oh, ow))
    return roi_align(
        feature_map,
        rois,
        output_size=output_size,
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
