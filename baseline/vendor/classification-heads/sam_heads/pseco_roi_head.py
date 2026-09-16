from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align


@dataclass
class PseCoROIOutput:
    embeddings: Tensor
    logits: Optional[Tensor]
    boxes_per_image: Tuple[int, ...]


class PseCoROIClassificationHead(nn.Module):
    """PseCo-style ROIAlign + heavy MLP head.

    The default embedding branch faithfully follows the official PseCo head:
      ROIAlign(7x7) -> Linear(256*7*7, 4096) -> ReLU -> Linear(4096, 512).

    Modes:
      closed_set: adds Linear(512, num_classes) for the vehicle-damage task.
      prototype: scores supplied 512-D CLIP/example prototypes by dot product,
                 matching the original PseCo classification mechanism.
      embedding: returns only the 512-D ROI embeddings.
    """

    VALID_MODES = {"closed_set", "prototype", "embedding"}

    def __init__(
        self,
        num_classes: Optional[int],
        *,
        in_channels: int = 256,
        roi_size: int = 7,
        hidden_dim: int = 4096,
        embedding_dim: int = 512,
        spatial_scale: float = 1.0 / 16.0,
        sampling_ratio: int = -1,
        aligned: bool = True,
        mode: str = "closed_set",
        normalize_prototypes: bool = False,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        if mode == "closed_set" and (num_classes is None or num_classes <= 1):
            raise ValueError("closed_set mode requires num_classes > 1")
        if min(in_channels, roi_size, hidden_dim, embedding_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if spatial_scale <= 0:
            raise ValueError("spatial_scale must be positive")

        self.num_classes = num_classes
        self.in_channels = int(in_channels)
        self.roi_size = int(roi_size)
        self.embedding_dim = int(embedding_dim)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = bool(aligned)
        self.mode = mode
        self.normalize_prototypes = bool(normalize_prototypes)

        flattened_dim = in_channels * roi_size * roi_size
        self.embedding_mlp = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.closed_set_classifier = (
            nn.Linear(embedding_dim, int(num_classes)) if mode == "closed_set" else None
        )

    @staticmethod
    def _validate_boxes(boxes: Sequence[Tensor], batch_size: int) -> Tuple[int, ...]:
        if len(boxes) != batch_size:
            raise ValueError(f"Expected {batch_size} box tensors, received {len(boxes)}")
        counts: List[int] = []
        for image_index, image_boxes in enumerate(boxes):
            if image_boxes.ndim != 2 or image_boxes.shape[-1] != 4:
                raise ValueError(
                    f"boxes[{image_index}] must be [Ni,4] XYXY, got {tuple(image_boxes.shape)}"
                )
            if image_boxes.numel() and not torch.isfinite(image_boxes).all():
                raise ValueError(f"boxes[{image_index}] contains NaN or Inf")
            if image_boxes.numel() and (
                (image_boxes[:, 2] < image_boxes[:, 0]).any()
                or (image_boxes[:, 3] < image_boxes[:, 1]).any()
            ):
                raise ValueError(f"boxes[{image_index}] contains invalid XYXY coordinates")
            counts.append(int(image_boxes.shape[0]))
        return tuple(counts)

    def forward(
        self,
        feature_map: Tensor,
        boxes: Sequence[Tensor],
        *,
        prototypes: Optional[Tensor] = None,
    ) -> PseCoROIOutput:
        if feature_map.ndim != 4:
            raise ValueError(f"feature_map must be [B,C,H,W], got {tuple(feature_map.shape)}")
        if feature_map.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} feature channels, got {feature_map.shape[1]}"
            )
        counts = self._validate_boxes(boxes, feature_map.shape[0])
        total_boxes = sum(counts)
        if total_boxes == 0:
            embeddings = feature_map.new_empty((0, self.embedding_dim))
            logits = None
            if self.mode == "closed_set":
                logits = feature_map.new_empty((0, int(self.num_classes)))
            elif self.mode == "prototype" and prototypes is not None:
                logits = feature_map.new_empty((0, prototypes.shape[-2]))
            return PseCoROIOutput(embeddings, logits, counts)

        roi_features = roi_align(
            feature_map,
            [box.to(device=feature_map.device, dtype=feature_map.dtype) for box in boxes],
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )
        embeddings = self.embedding_mlp(roi_features.flatten(start_dim=1))

        if self.mode == "embedding":
            logits = None
        elif self.mode == "closed_set":
            assert self.closed_set_classifier is not None
            logits = self.closed_set_classifier(embeddings)
        else:
            if prototypes is None:
                raise ValueError("prototype mode requires prototypes")
            if prototypes.ndim != 2 or prototypes.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"prototypes must be [K,{self.embedding_dim}], got {tuple(prototypes.shape)}"
                )
            prototypes = prototypes.to(device=embeddings.device, dtype=embeddings.dtype)
            if self.normalize_prototypes:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
            logits = embeddings @ prototypes.transpose(0, 1)

        return PseCoROIOutput(embeddings=embeddings, logits=logits, boxes_per_image=counts)
