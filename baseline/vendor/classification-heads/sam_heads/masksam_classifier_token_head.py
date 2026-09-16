from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn


class MaskSAMClassifierTokenHead(nn.Module):
    """MaskSAM-style learnable global classifier token and linear classifier.

    Faithful core behavior:
      1. Create one learnable global classifier token.
      2. Optionally add an auxiliary classifier token from a prompt generator.
      3. Append the resulting token to SAM decoder input tokens.
      4. Read the final transformed classifier token and apply one Linear layer.

    This module intentionally does not hide the decoder call. The token MUST pass
    through the same two-way transformer as the mask/IoU tokens.
    """

    def __init__(
        self,
        transformer_dim: int,
        num_classes: int,
        *,
        include_no_object: bool = True,
    ) -> None:
        super().__init__()
        if transformer_dim <= 0 or num_classes <= 1:
            raise ValueError("transformer_dim must be positive and num_classes must exceed one")
        self.transformer_dim = int(transformer_dim)
        self.num_classes = int(num_classes)
        self.include_no_object = bool(include_no_object)
        output_classes = num_classes + int(include_no_object)

        self.global_classifier_token = nn.Embedding(1, transformer_dim)
        self.classifier = nn.Linear(transformer_dim, output_classes)

    def build_classifier_token(
        self,
        batch_size: int,
        *,
        auxiliary_classifier_token: Optional[Tensor] = None,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        token = self.global_classifier_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if auxiliary_classifier_token is None:
            return token

        auxiliary = auxiliary_classifier_token
        if auxiliary.ndim == 2:
            auxiliary = auxiliary.unsqueeze(1)
        expected = (batch_size, 1, self.transformer_dim)
        if auxiliary.shape != expected:
            raise ValueError(f"Expected auxiliary classifier token {expected}, got {tuple(auxiliary.shape)}")
        return token + auxiliary.to(device=token.device, dtype=token.dtype)

    def append_to_decoder_tokens(
        self,
        decoder_tokens: Tensor,
        *,
        auxiliary_classifier_token: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """Append the classifier token and return its sequence index."""
        if decoder_tokens.ndim != 3:
            raise ValueError(f"decoder_tokens must be [B,T,D], got {tuple(decoder_tokens.shape)}")
        if decoder_tokens.shape[-1] != self.transformer_dim:
            raise ValueError(
                f"Expected transformer dim {self.transformer_dim}, got {decoder_tokens.shape[-1]}"
            )
        classifier_token = self.build_classifier_token(
            decoder_tokens.shape[0], auxiliary_classifier_token=auxiliary_classifier_token
        )
        token_index = decoder_tokens.shape[1]
        from torch import cat
        return cat((decoder_tokens, classifier_token), dim=1), token_index

    def append_to_decoder_tokens_simple(
        self,
        decoder_tokens: Tensor,
        *,
        auxiliary_classifier_token: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """Backward-compatible alias for append_to_decoder_tokens."""
        return self.append_to_decoder_tokens(
            decoder_tokens,
            auxiliary_classifier_token=auxiliary_classifier_token,
        )

    def classify_transformed_tokens(self, transformed_tokens: Tensor, token_index: int = -1) -> Tensor:
        if transformed_tokens.ndim != 3:
            raise ValueError(
                f"transformed_tokens must be [B,T,D], got {tuple(transformed_tokens.shape)}"
            )
        if transformed_tokens.shape[-1] != self.transformer_dim:
            raise ValueError(
                f"Expected transformer dim {self.transformer_dim}, got {transformed_tokens.shape[-1]}"
            )
        if not -transformed_tokens.shape[1] <= token_index < transformed_tokens.shape[1]:
            raise IndexError("token_index is outside transformed token sequence")
        return self.classifier(transformed_tokens[:, token_index, :])

    def forward(self, transformed_classifier_token: Tensor) -> Tensor:
        if transformed_classifier_token.ndim != 2:
            raise ValueError(
                "forward expects the transformed classifier token [B,D]; use "
                "classify_transformed_tokens for a full token sequence"
            )
        if transformed_classifier_token.shape[-1] != self.transformer_dim:
            raise ValueError(
                f"Expected transformer dim {self.transformer_dim}, got {transformed_classifier_token.shape[-1]}"
            )
        return self.classifier(transformed_classifier_token)
