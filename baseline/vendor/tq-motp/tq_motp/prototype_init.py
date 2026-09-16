from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def torch_kmeans(
    samples: Tensor,
    clusters: int,
    *,
    iterations: int = 50,
    seed: int = 42,
) -> Tensor:
    """Small dependency-free cosine K-means for offline prototype initialization."""
    if samples.ndim != 2:
        raise ValueError("samples must be [N,D]")
    if samples.shape[0] == 0:
        raise ValueError("cannot cluster an empty tensor")
    x = F.normalize(samples.float(), dim=-1)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    if x.shape[0] >= clusters:
        initial = torch.randperm(x.shape[0], generator=generator, device=x.device)[:clusters]
        centers = x[initial].clone()
    else:
        repeat = (clusters + x.shape[0] - 1) // x.shape[0]
        centers = x.repeat(repeat, 1)[:clusters].clone()
        centers += 1.0e-3 * torch.randn(centers.shape, generator=generator, device=x.device)
        centers = F.normalize(centers, dim=-1)

    for _ in range(iterations):
        assignment = torch.matmul(x, centers.t()).argmax(dim=1)
        new_centers = []
        for idx in range(clusters):
            selected = x[assignment == idx]
            if selected.numel() == 0:
                fallback = x[torch.randint(x.shape[0], (1,), generator=generator, device=x.device)][0]
                new_centers.append(fallback)
            else:
                new_centers.append(F.normalize(selected.mean(dim=0), dim=0))
        updated = torch.stack(new_centers, dim=0)
        if torch.allclose(updated, centers, atol=1.0e-5, rtol=1.0e-5):
            centers = updated
            break
        centers = updated
    return F.normalize(centers, dim=-1)


def initialize_prototype_bank_from_dump(
    dump: Mapping[str, Tensor],
    *,
    num_classes: int,
    prototype_counts: Mapping[str, int],
    min_selected_weight: float = 0.05,
    max_samples_per_class_region: int = 5000,
    seed: int = 42,
) -> Dict[str, Tensor]:
    """Create typed class prototype tensors from a descriptor dump.

    Required dump keys:
      labels [N]
      <region>_descriptors [N,K,D]
      optional <region>_weights [N,K]
    """
    labels = torch.as_tensor(dump["labels"], dtype=torch.long)
    result: Dict[str, Tensor] = {}
    for region, kp in prototype_counts.items():
        descriptors = torch.as_tensor(dump[f"{region}_descriptors"], dtype=torch.float32)
        weights = dump.get(f"{region}_weights")
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)
        class_centers = []
        for class_idx in range(num_classes):
            selected = descriptors[labels == class_idx]
            if selected.numel() == 0:
                raise ValueError(f"no descriptors for class {class_idx}, region {region}")
            selected = selected.reshape(-1, selected.shape[-1])
            if weights is not None:
                selected_weights = weights[labels == class_idx].reshape(-1)
                selected = selected[selected_weights >= min_selected_weight]
            if selected.shape[0] == 0:
                selected = descriptors[labels == class_idx].reshape(-1, descriptors.shape[-1])
            if selected.shape[0] > max_samples_per_class_region:
                generator = torch.Generator().manual_seed(seed + class_idx)
                idx = torch.randperm(selected.shape[0], generator=generator)[:max_samples_per_class_region]
                selected = selected[idx]
            class_centers.append(torch_kmeans(selected, kp, seed=seed + class_idx))
        result[region] = torch.stack(class_centers, dim=0)
    return result


def save_prototype_bank(bank: Mapping[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": {k: v.cpu() for k, v in bank.items()}}, path)
