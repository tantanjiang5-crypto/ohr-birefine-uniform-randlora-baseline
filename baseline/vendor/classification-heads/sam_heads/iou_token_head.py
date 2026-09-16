from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .common import TokenMLPClassifier


class IoUTokenClassificationHead(nn.Module):
    """Closed-set classifier over the raw IoU token from the SAM transformer.

    Do not feed the output of SAM's existing iou_prediction_head here; doing so
    would give this baseline an extra MLP compared with MaskTokenClassificationHead.
    """

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        *,
        hidden_dim: Optional[int] = None,
        num_linear_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = TokenMLPClassifier(
            input_dim=token_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_linear_layers=num_linear_layers,
            dropout=dropout,
        )

    def forward(self, iou_token: Tensor) -> Tensor:
        return self.classifier(iou_token)
