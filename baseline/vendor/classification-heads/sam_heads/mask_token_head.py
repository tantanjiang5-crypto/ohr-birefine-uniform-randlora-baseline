from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .common import TokenMLPClassifier, select_mask_token


class MaskTokenClassificationHead(nn.Module):
    """Closed-set classifier over a raw SAM mask token.

    For GT-box experiments with multimask_output=False, use strategy='first'.
    For true multimask inference, strategy='best_iou' is normally preferable.
    """

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        *,
        hidden_dim: Optional[int] = None,
        num_linear_layers: int = 3,
        dropout: float = 0.1,
        selection_strategy: str = "first",
        fixed_index: int = 0,
    ) -> None:
        super().__init__()
        self.selection_strategy = selection_strategy
        self.fixed_index = int(fixed_index)
        self.classifier = TokenMLPClassifier(
            input_dim=token_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_linear_layers=num_linear_layers,
            dropout=dropout,
        )

    def forward(self, mask_tokens: Tensor, iou_scores: Optional[Tensor] = None) -> Tensor:
        selected = select_mask_token(
            mask_tokens,
            iou_scores=iou_scores,
            strategy=self.selection_strategy,
            fixed_index=self.fixed_index,
        ).token
        return self.classifier(selected)
