from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class CocoGTBoxDataset(Dataset):
    """COCO instance dataset preserving annotation-id order for oracle GT boxes."""

    def __init__(
        self,
        annotation_json: str | Path,
        image_root: str | Path,
        *,
        category_ids: list[int] | None = None,
        filter_empty: bool = False,
        ignore_crowd: bool = True,
    ) -> None:
        try:
            from pycocotools.coco import COCO
        except ImportError as error:
            # The validated SAM/Q-series environment vendors pycocotools in
            # this project-local dependency directory.  Do not install or
            # upgrade anything at runtime; make that existing dependency
            # discoverable for the common trainer and evaluator.
            project_root = Path(os.environ.get("SAM44_ROOT", Path(__file__).resolve().parents[3]))
            project_deps = project_root / "baseline" / "vendor" / "python-deps"
            if project_deps.is_dir() and str(project_deps) not in sys.path:
                sys.path.insert(0, str(project_deps))
            try:
                from pycocotools.coco import COCO
            except ImportError:
                raise ImportError(
                    "pycocotools is required; expected the validated project dependency at "
                    f"{project_deps}"
                ) from error
        self.annotation_json = str(Path(annotation_json).resolve())
        self.image_root = Path(image_root).resolve()
        self.coco = COCO(self.annotation_json)
        available_categories = sorted(self.coco.getCatIds())
        self.category_ids = list(category_ids) if category_ids is not None else available_categories
        unknown = sorted(set(self.category_ids) - set(available_categories))
        if unknown:
            raise ValueError(f"category ids not found in COCO JSON: {unknown}")
        self.category_to_contiguous = {category_id: index for index, category_id in enumerate(self.category_ids)}
        self.contiguous_to_category = {index: category_id for category_id, index in self.category_to_contiguous.items()}
        self.category_records = {record["id"]: record for record in self.coco.loadCats(self.category_ids)}
        self.class_names = [str(self.category_records[category_id]["name"]) for category_id in self.category_ids]
        self.ignore_crowd = bool(ignore_crowd)

        image_ids = sorted(self.coco.getImgIds())
        if filter_empty:
            retained: list[int] = []
            for image_id in image_ids:
                annotation_ids = self.coco.getAnnIds(
                    imgIds=[image_id], catIds=self.category_ids, iscrowd=False if self.ignore_crowd else None
                )
                if annotation_ids:
                    retained.append(image_id)
            image_ids = retained
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def class_counts(self) -> list[int]:
        counts = [0 for _ in self.category_ids]
        for annotation in self.coco.dataset.get("annotations", []):
            if self.ignore_crowd and int(annotation.get("iscrowd", 0)):
                continue
            category_id = int(annotation["category_id"])
            if category_id in self.category_to_contiguous:
                counts[self.category_to_contiguous[category_id]] += 1
        return counts

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = int(self.image_ids[index])
        image_record = self.coco.loadImgs([image_id])[0]
        image_path = self.image_root / image_record["file_name"]
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB").copy()
        original_size = (int(image_record["height"]), int(image_record["width"]))
        if image.size != (original_size[1], original_size[0]):
            raise RuntimeError(f"image metadata mismatch for {image_path}: PIL={image.size}, COCO={original_size[::-1]}")

        annotation_ids = self.coco.getAnnIds(
            imgIds=[image_id], catIds=self.category_ids, iscrowd=False if self.ignore_crowd else None
        )
        annotations = sorted(self.coco.loadAnns(annotation_ids), key=lambda record: int(record["id"]))
        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []
        kept_annotation_ids: list[int] = []
        category_ids: list[int] = []
        height, width = original_size
        for annotation in annotations:
            x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
            x1 = min(max(x, 0.0), float(width))
            y1 = min(max(y, 0.0), float(height))
            x2 = min(max(x + box_width, 0.0), float(width))
            y2 = min(max(y + box_height, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            mask = self.coco.annToMask(annotation).astype(np.uint8, copy=False)
            if mask.shape != original_size:
                raise RuntimeError(f"mask shape mismatch for annotation {annotation['id']}: {mask.shape} vs {original_size}")
            category_id = int(annotation["category_id"])
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_to_contiguous[category_id])
            masks.append(mask)
            kept_annotation_ids.append(int(annotation["id"]))
            category_ids.append(category_id)

        if masks:
            masks_array = np.stack(masks, axis=0)
        else:
            masks_array = np.zeros((0, original_size[0], original_size[1]), dtype=np.uint8)
        return {
            "image": image,
            "image_id": image_id,
            "original_size": original_size,
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.from_numpy(masks_array),
            "annotation_ids": kept_annotation_ids,
            "category_ids": category_ids,
            "image_path": str(image_path),
        }


class DeterministicEpochSampler(Sampler[int]):
    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order())

    def __len__(self) -> int:
        return len(self.data_source)
