from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import RankSortConfig


@dataclass
class RankSortOutput:
    loss: Tensor
    ranking_error: Tensor
    sorting_error: Tensor
    positive_count: int
    negative_count: int
    relevant_negative_count: int
    active_sort_classes: int
    score_mode: str
    sort_scope: str


def _smooth_relation(difference: Tensor, delta: float) -> Tensor:
    if delta > 0:
        return torch.clamp(difference / (2.0 * delta) + 0.5, min=0.0, max=1.0)
    return (difference >= 0).to(difference.dtype)


class _RankSortIdentityUpdate(torch.autograd.Function):
    """Device-agnostic Identity Update implementation adapted for MQ-RS.

    The forward value follows Rank & Sort error definitions. The backward pass
    uses the error-driven Identity Update from Oksuz et al. The sorting group
    can be global or restricted to positives with the same class label.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        flat_scores: Tensor,
        flat_targets: Tensor,
        flat_class_ids: Tensor,
        delta: float,
        eps: float,
        sort_scope_code: int,
        rank_weight: float,
        sort_weight: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if flat_scores.ndim != 1 or flat_targets.shape != flat_scores.shape:
            raise ValueError("flat_scores and flat_targets must be one-dimensional and shape-aligned")
        if flat_class_ids.shape != flat_scores.shape:
            raise ValueError("flat_class_ids must match flat_scores")

        device = flat_scores.device
        dtype = flat_scores.dtype
        grads = torch.zeros_like(flat_scores)
        positive_mask = flat_targets > 0
        positive_indices = positive_mask.nonzero(as_tuple=False).flatten()
        positive_count = int(positive_indices.numel())

        if positive_count == 0:
            zero = flat_scores.sum() * 0.0
            ctx.save_for_backward(grads)
            return zero, zero.detach(), zero.detach()

        positive_scores = flat_scores[positive_mask]
        positive_targets = flat_targets[positive_mask]
        positive_classes = flat_class_ids[positive_mask]

        threshold = positive_scores.min() - float(delta)
        relevant_negative_mask = (flat_targets == 0) & (flat_scores >= threshold)
        relevant_negative_scores = flat_scores[relevant_negative_mask]

        positive_grads = torch.zeros_like(positive_scores)
        negative_grads = torch.zeros_like(relevant_negative_scores)
        ranking_errors = torch.zeros(positive_count, device=device, dtype=dtype)
        sorting_errors = torch.zeros(positive_count, device=device, dtype=dtype)

        # Following the official implementation, process positives from low to high score.
        order = torch.argsort(positive_scores)
        for index_tensor in order:
            index = int(index_tensor.item())
            score_i = positive_scores[index]

            all_positive_relations = _smooth_relation(positive_scores - score_i, float(delta))
            negative_relations = _smooth_relation(relevant_negative_scores - score_i, float(delta))

            rank_positive = all_positive_relations.sum()
            false_positive_count = negative_relations.sum()
            total_rank = (rank_positive + false_positive_count).clamp_min(float(eps))
            ranking_error_i = false_positive_count / total_rank
            ranking_errors[index] = ranking_error_i

            if sort_scope_code == 1:  # class-local sorting
                group_mask = positive_classes == positive_classes[index]
            else:
                group_mask = torch.ones_like(positive_classes, dtype=torch.bool)
            group_indices = group_mask.nonzero(as_tuple=False).flatten()
            group_scores = positive_scores[group_mask]
            group_targets = positive_targets[group_mask]
            group_relations = _smooth_relation(group_scores - score_i, float(delta))
            group_rank = group_relations.sum().clamp_min(float(eps))

            current_sorting_error = (group_relations * (1.0 - group_targets)).sum() / group_rank
            target_order = (group_targets >= positive_targets[index]).to(dtype) * group_relations
            target_rank = target_order.sum().clamp_min(float(eps))
            target_sorting_error = (target_order * (1.0 - group_targets)).sum() / target_rank
            sorting_error_i = (current_sorting_error - target_sorting_error).clamp_min(0.0)
            sorting_errors[index] = sorting_error_i

            if float(false_positive_count.detach().item()) > eps and rank_weight > 0:
                weighted_rank_error = float(rank_weight) * ranking_error_i
                positive_grads[index] -= weighted_rank_error
                negative_grads += negative_relations * (weighted_rank_error / false_positive_count)

            if sort_weight > 0:
                lower_quality = group_targets < positive_targets[index]
                missorted = lower_quality.to(dtype) * group_relations
                missorted_denominator = missorted.sum()
                if float(missorted_denominator.detach().item()) > eps:
                    weighted_sort_error = float(sort_weight) * sorting_error_i
                    positive_grads[index] -= weighted_sort_error
                    positive_grads[group_indices] += missorted * (weighted_sort_error / missorted_denominator)

        normalization = float(positive_count)
        grads[positive_mask] = positive_grads / normalization
        grads[relevant_negative_mask] = negative_grads / normalization
        ctx.save_for_backward(grads)

        ranking_mean = ranking_errors.mean()
        sorting_mean = sorting_errors.mean()
        total = float(rank_weight) * ranking_mean + float(sort_weight) * sorting_mean
        return total, ranking_mean.detach(), sorting_mean.detach()

    @staticmethod
    def backward(ctx, grad_total: Tensor, grad_rank: Tensor, grad_sort: Tensor):  # type: ignore[override]
        del grad_rank, grad_sort
        (saved_grads,) = ctx.saved_tensors
        return saved_grads * grad_total, None, None, None, None, None, None, None


class MultiClassRankSortLoss(nn.Module):
    """Rank true-class scores above wrong classes and sort by mask quality.

    For an ``N x C`` multi-class head, each true-class entry is a positive with
    a continuous target quality Q_i, while all ``N*(C-1)`` wrong-class entries
    are negatives. This preserves the positive/negative ranking objective even
    in a GT-box protocol where every prompt corresponds to a real object.

    ``score_mode='log_probability'`` is recommended for softmax heads because
    raw logits are only defined up to an arbitrary per-instance additive shift.
    """

    def __init__(self, config: RankSortConfig | None = None) -> None:
        super().__init__()
        self.config = config or RankSortConfig()

    def _scores(self, logits: Tensor) -> Tensor:
        mode = self.config.score_mode
        if mode == "log_probability":
            return logits.float().log_softmax(dim=-1)
        if mode == "probability":
            return logits.float().softmax(dim=-1)
        if mode == "raw_logit":
            return logits.float()
        raise RuntimeError(f"unsupported score mode: {mode}")

    def forward(self, logits: Tensor, labels: Tensor, qualities: Tensor) -> RankSortOutput:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N,C], got {tuple(logits.shape)}")
        n, c = logits.shape
        labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
        qualities = qualities.to(device=logits.device, dtype=torch.float32).reshape(-1)
        if labels.numel() != n or qualities.numel() != n:
            raise ValueError("labels and qualities must each contain N entries")
        if n == 0:
            zero = logits.sum() * 0.0
            return RankSortOutput(zero, zero.detach(), zero.detach(), 0, 0, 0, 0,
                                  self.config.score_mode, self.config.sort_scope)
        if c < 2:
            raise ValueError("Rank & Sort requires at least two classes")
        if torch.any((labels < 0) | (labels >= c)):
            raise ValueError("labels contain an out-of-range class index")
        if not torch.isfinite(logits).all():
            raise ValueError("logits contain NaN or Inf")
        if not torch.isfinite(qualities).all():
            raise ValueError("qualities contain NaN or Inf")

        qualities = qualities.clamp(min=1e-8, max=1.0)
        scores = self._scores(logits)
        targets = torch.zeros_like(scores)
        targets.scatter_(1, labels[:, None], qualities[:, None])
        class_ids = torch.full_like(labels[:, None].expand(n, c), -1)
        class_ids.scatter_(1, labels[:, None], labels[:, None])

        flat_scores = scores.reshape(-1)
        flat_targets = targets.reshape(-1)
        flat_class_ids = class_ids.reshape(-1)
        sort_scope_code = 1 if self.config.sort_scope == "class" else 0

        # Always execute the ranking kernel in FP32; gradients propagate back
        # through log_softmax/softmax to the original logits.
        total, rank_error, sort_error = _RankSortIdentityUpdate.apply(
            flat_scores,
            flat_targets,
            flat_class_ids,
            float(self.config.delta),
            float(self.config.eps),
            sort_scope_code,
            float(self.config.rank_weight),
            float(self.config.sort_weight),
        )

        with torch.no_grad():
            positive_scores = flat_scores[flat_targets > 0]
            threshold = positive_scores.min() - self.config.delta
            relevant_negatives = int(((flat_targets == 0) & (flat_scores >= threshold)).sum().item())
            counts = torch.bincount(labels, minlength=c)
            active_sort_classes = int((counts >= 2).sum().item())

        return RankSortOutput(
            loss=total,
            ranking_error=rank_error,
            sorting_error=sort_error,
            positive_count=n,
            negative_count=n * (c - 1),
            relevant_negative_count=relevant_negatives,
            active_sort_classes=active_sort_classes,
            score_mode=self.config.score_mode,
            sort_scope=self.config.sort_scope,
        )
