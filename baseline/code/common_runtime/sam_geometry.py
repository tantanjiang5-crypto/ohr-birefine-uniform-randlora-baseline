from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PreparedBatch:
    images: Tensor
    boxes_list: list[Tensor]
    masks_256_list: list[Tensor]
    labels_list: list[Tensor]
    input_sizes: list[tuple[int, int]]
    original_sizes: list[tuple[int, int]]
    image_ids: list[int]
    annotation_ids_list: list[list[int]]


def get_preprocess_shape(old_h: int, old_w: int, long_side_length: int = 1024) -> tuple[int, int]:
    scale = long_side_length * 1.0 / max(old_h, old_w)
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    return new_h, new_w


def build_padded_lowres_mask_targets(
    masks: Tensor,
    input_size: tuple[int, int],
    *,
    encoder_size: int = 1024,
    lowres_size: int = 256,
) -> Tensor:
    """Map original-resolution masks into SAM's padded low-resolution frame.

    Correct chain: original mask -> aspect-ratio-preserving resized input_size ->
    right/bottom pad to 1024 -> nearest downsample to 256. This must match the
    low-resolution logits emitted by SAM's selected mask token.
    """

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"masks must be [N,H,W], got {tuple(masks.shape)}")
    n = masks.shape[0]
    if n == 0:
        return masks.new_empty((0, lowres_size, lowres_size), dtype=torch.float32)
    resized_h, resized_w = input_size
    if not (0 < resized_h <= encoder_size and 0 < resized_w <= encoder_size):
        raise ValueError(f"invalid SAM input_size={input_size}")
    masks_float = masks.float().unsqueeze(1)
    resized = F.interpolate(masks_float, size=input_size, mode="nearest")
    padded = F.pad(resized, (0, encoder_size - resized_w, 0, encoder_size - resized_h))
    lowres = F.interpolate(padded, size=(lowres_size, lowres_size), mode="nearest")
    return lowres[:, 0].clamp(0, 1)


class CorrectSamPreprocessor:
    """Official SAM resize/normalize/pad flow plus aligned 256-mask targets."""

    def __init__(self, sam: torch.nn.Module, device: torch.device | str, image_size: int = 1024) -> None:
        try:
            from segment_anything.utils.transforms import ResizeLongestSide
        except ImportError as error:
            raise ImportError("segment_anything must be importable before constructing the preprocessor") from error
        self.sam = sam
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.transform = ResizeLongestSide(self.image_size)
        encoder_size = int(getattr(sam.image_encoder, "img_size", self.image_size))
        if encoder_size != self.image_size:
            raise RuntimeError(f"expected SAM encoder size {self.image_size}, got {encoder_size}")

    @staticmethod
    def _image_to_numpy_rgb(image: Any) -> np.ndarray:
        if isinstance(image, np.ndarray):
            array = image
        elif torch.is_tensor(image):
            tensor = image.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"image tensor must be CHW or HWC, got {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        else:
            array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"image must resolve to HWC RGB/RGBA, got {array.shape}")
        array = array[..., :3]
        if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
            array = array * 255.0
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))

    def process(self, batch: Sequence[dict[str, Any]]) -> PreparedBatch:
        images: list[Tensor] = []
        boxes_list: list[Tensor] = []
        masks_256_list: list[Tensor] = []
        labels_list: list[Tensor] = []
        input_sizes: list[tuple[int, int]] = []
        original_sizes: list[tuple[int, int]] = []
        image_ids: list[int] = []
        annotation_ids_list: list[list[int]] = []

        for item_index, item in enumerate(batch):
            image_np = self._image_to_numpy_rgb(item["image"])
            original_h, original_w = image_np.shape[:2]
            declared = tuple(int(value) for value in item.get("original_size", (original_h, original_w)))
            if declared != (original_h, original_w):
                raise RuntimeError(
                    f"item {item_index} original_size mismatch: declared={declared}, image={(original_h, original_w)}"
                )
            resized_np = self.transform.apply_image(image_np)
            input_size = tuple(int(value) for value in resized_np.shape[:2])
            expected = get_preprocess_shape(original_h, original_w, self.image_size)
            if input_size != expected:
                raise RuntimeError(f"ResizeLongestSide mismatch: got {input_size}, expected {expected}")

            resized_tensor = torch.as_tensor(resized_np, device=self.device).permute(2, 0, 1).float()
            # Official SAM preprocess: normalize first, then right/bottom pad.
            preprocessed = self.sam.preprocess(resized_tensor)
            if tuple(preprocessed.shape) != (3, self.image_size, self.image_size):
                raise RuntimeError(f"SAM preprocess produced {tuple(preprocessed.shape)}")
            images.append(preprocessed)

            boxes_original = torch.as_tensor(item["boxes"], dtype=torch.float32, device=self.device).reshape(-1, 4)
            boxes_resized = self.transform.apply_boxes_torch(boxes_original, (original_h, original_w))
            # ResizeLongestSide rounds the shorter resized dimension to an
            # integer.  Verify against those official, per-axis dimensions,
            # rather than an ideal continuous longest-side scale: the latter
            # differs slightly whenever the rounded dimension is not exact.
            # This remains a single original->ResizeLongestSide box transform.
            scale_x = float(input_size[1]) / float(original_w)
            scale_y = float(input_size[0]) / float(original_h)
            expected_boxes = boxes_original * boxes_original.new_tensor(
                [scale_x, scale_y, scale_x, scale_y]
            )
            if not torch.allclose(boxes_resized, expected_boxes, rtol=0.0, atol=1e-4):
                raise RuntimeError("GT boxes are not exactly in the single ResizeLongestSide frame")
            if boxes_resized.numel():
                if not torch.isfinite(boxes_resized).all():
                    raise RuntimeError("non-finite transformed box")
                tolerance = 1e-3
                if float(boxes_resized.min()) < -tolerance or float(boxes_resized.max()) > self.image_size + tolerance:
                    raise RuntimeError(f"transformed boxes outside SAM frame: min={boxes_resized.min()}, max={boxes_resized.max()}")
            boxes_list.append(boxes_resized)

            masks_original = torch.as_tensor(item["masks"], dtype=torch.float32, device=self.device)
            if masks_original.ndim == 2:
                masks_original = masks_original.unsqueeze(0)
            if masks_original.shape[0] != boxes_original.shape[0]:
                raise RuntimeError(
                    f"item {item_index} instance mismatch: masks={masks_original.shape[0]}, boxes={boxes_original.shape[0]}"
                )
            masks_256 = build_padded_lowres_mask_targets(masks_original, input_size)
            if tuple(masks_256.shape) != (boxes_original.shape[0], 256, 256):
                raise RuntimeError(f"low-resolution mask target must be [N,256,256], got {tuple(masks_256.shape)}")
            masks_256_list.append(masks_256)

            labels = torch.as_tensor(item["labels"], dtype=torch.long, device=self.device).reshape(-1)
            if labels.numel() != boxes_original.shape[0]:
                raise RuntimeError(
                    f"item {item_index} instance mismatch: labels={labels.numel()}, boxes={boxes_original.shape[0]}"
                )
            labels_list.append(labels)
            input_sizes.append(input_size)
            original_sizes.append((original_h, original_w))
            image_ids.append(int(item["image_id"]))
            annotation_ids_list.append([int(value) for value in item.get("annotation_ids", [])])

        return PreparedBatch(
            images=torch.stack(images, dim=0),
            boxes_list=boxes_list,
            masks_256_list=masks_256_list,
            labels_list=labels_list,
            input_sizes=input_sizes,
            original_sizes=original_sizes,
            image_ids=image_ids,
            annotation_ids_list=annotation_ids_list,
        )

    def postprocess_instance_logits(
        self,
        lowres_logits: Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> Tensor:
        if lowres_logits.ndim == 2:
            lowres_logits = lowres_logits[None, None]
        elif lowres_logits.ndim == 3:
            lowres_logits = lowres_logits[:, None]
        elif lowres_logits.ndim != 4:
            raise ValueError(f"lowres_logits must be [H,W], [N,H,W], or [N,1,H,W], got {tuple(lowres_logits.shape)}")
        if lowres_logits.shape[1] != 1:
            raise ValueError("postprocess expects one selected mask per instance")
        if tuple(lowres_logits.shape[-2:]) != (256, 256):
            raise ValueError(f"decoder logits must be 256x256, got {tuple(lowres_logits.shape)}")
        postprocessed = self.sam.postprocess_masks(lowres_logits, input_size, original_size)
        restored = postprocessed[:, 0]
        if tuple(restored.shape[-2:]) != tuple(original_size):
            raise RuntimeError(
                f"SAM postprocess did not restore original HxW: got {tuple(restored.shape[-2:])}, expected {tuple(original_size)}"
            )
        return restored
