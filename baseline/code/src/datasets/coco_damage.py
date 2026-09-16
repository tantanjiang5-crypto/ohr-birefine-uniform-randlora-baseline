"""COCO damage dataset for SAM1 instance segmentation with GT box prompts.

Reads COCO-format instance annotations and prepares per-image data with
boxes, masks, and class labels.  Each __getitem__ returns one image with
all its GT instances.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CocoDamageDataset(Dataset):
    """COCO instance-segmentation dataset for vehicle damage.

    Reads images + polygons, converts masks to binary numpy arrays on the fly.
    Applies SAM ResizeLongestSide + pad during collation, NOT here.
    """

    def __init__(
        self,
        root: str,
        annotation_path: str,
        category_mapping: Dict[int, int],
        *,
        split: str = "train",
    ) -> None:
        """
        Args:
            root: Dataset root, e.g. /workspace/datasets/coco-damage-open.
            annotation_path: Path to instances_*.json.
            category_mapping: Original COCO cat_id → contiguous 0-indexed label.
            split: 'train' or 'val'.
        """
        self.root = Path(root)
        self.split = split
        self.category_mapping = category_mapping

        with open(annotation_path) as f:
            self.coco = json.load(f)

        # Build indices
        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.categories = {c["id"]: c["name"] for c in self.coco["categories"]}

        # image_id → list of annotations
        self.anns_by_image: Dict[int, List[Dict]] = {}
        for ann in self.annotations:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Image id → image info
        self.image_info: Dict[int, Dict] = {img["id"]: img for img in self.images}

        # Images that have at least one annotation
        self._valid_image_ids = sorted(
            set(img["id"] for img in self.images)
        )

    # ------------------------------------------------------------------
    # Properties for reporting
    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(set(self.category_mapping.values()))

    @property
    def class_names(self) -> List[str]:
        return [self.categories[cid] for cid in sorted(self.categories)]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._valid_image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self._valid_image_ids[index]
        info = self.image_info[image_id]
        anns = self.anns_by_image.get(image_id, [])
        file_path = os.path.join(self.root, f"{self.split}2017", info["file_name"])

        # Load image as PIL
        image_pil = Image.open(file_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        image_np = np.array(image_pil)

        boxes_xywh = []
        masks = []
        labels = []
        ann_ids = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.category_mapping:
                continue
            bbox = ann["bbox"]  # COCO XYWH
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            # COCO XYWH → XYXY (in original image coords)
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            boxes_xywh.append([x1, y1, x2, y2])

            # Polygon → binary mask (original resolution)
            seg = ann["segmentation"]
            mask = self._poly_to_mask(seg, orig_h, orig_w)
            masks.append(mask)

            labels.append(self.category_mapping[cat_id])
            ann_ids.append(ann["id"])

        return {
            "image_id": image_id,
            "image": image_np,                     # HxWx3 uint8 numpy
            "original_size": (orig_h, orig_w),
            "boxes": np.array(boxes_xywh, dtype=np.float32) if boxes_xywh else np.zeros((0, 4), dtype=np.float32),
            "masks": np.stack(masks, axis=0) if masks else np.zeros((0, orig_h, orig_w), dtype=np.uint8),
            "labels": np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            "annotation_ids": np.array(ann_ids, dtype=np.int64) if ann_ids else np.zeros((0,), dtype=np.int64),
        }

    @staticmethod
    def _poly_to_mask(segmentation: List, h: int, w: int) -> np.ndarray:
        """Convert COCO polygon(s) to a binary mask."""
        from pycocotools import mask as cocomask

        if isinstance(segmentation, dict) and "counts" in segmentation:
            # RLE
            return cocomask.decode(segmentation).astype(np.uint8)

        # Polygon(s)
        rles = cocomask.frPyObjects(segmentation, h, w)
        if isinstance(rles, list):
            rle = cocomask.merge(rles)
        else:
            rle = rles
        return cocomask.decode(rle).astype(np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_zero_instance_image(self) -> int:
        """Find an image ID with zero instances, or return -1."""
        for img_id in self._valid_image_ids:
            if len(self.anns_by_image.get(img_id, [])) == 0:
                return img_id
        return -1

    def get_class_distribution(self) -> Dict[str, int]:
        from collections import Counter
        counter: Counter = Counter()
        for ann in self.annotations:
            name = self.categories.get(ann["category_id"], "UNKNOWN")
            counter[name] += 1
        return dict(counter)
