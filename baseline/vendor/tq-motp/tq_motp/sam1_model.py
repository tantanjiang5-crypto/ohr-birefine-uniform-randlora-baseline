from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .head import TQMOTPHead
from .integration import forward_tqmotp_one_image
from .types import TQMOTPOutput


def predict_masks_with_raw_tokens(
    mask_decoder: nn.Module,
    *,
    image_embeddings: Tensor,
    image_pe: Tensor,
    sparse_prompt_embeddings: Tensor,
    dense_prompt_embeddings: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Official SAM1 MaskDecoder.predict_masks logic with raw token outputs.

    This avoids changing Meta's mask decoder source. It assumes the standard
    SAM1 MaskDecoder attributes and returns masks, IoU predictions, mask tokens,
    and the raw IoU token.
    """
    output_tokens = torch.cat(
        [mask_decoder.iou_token.weight, mask_decoder.mask_tokens.weight], dim=0
    )
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape
    hs, src = mask_decoder.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : 1 + mask_decoder.num_mask_tokens, :]

    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = mask_decoder.output_upscaling(src)
    hyper_in_list = [
        mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
        for i in range(mask_decoder.num_mask_tokens)
    ]
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
    iou_pred = mask_decoder.iou_prediction_head(iou_token_out)
    return masks, iou_pred, mask_tokens_out, iou_token_out


class Sam1TQMOTPModel(nn.Module):
    """Independent SAM1 + TQ-MOTP wrapper for already preprocessed images.

    It can be used directly or its per-image forward body can be copied into the
    existing sam44 wrapper. Images must already be ResizeLongestSide-normalized-
    padded exactly like the current experiment code, and boxes must be transformed
    to the padded SAM coordinate system.
    """

    def __init__(self, sam: nn.Module, head: TQMOTPHead, mask_index: int = 0) -> None:
        super().__init__()
        self.sam = sam
        self.head = head
        self.mask_index = int(mask_index)

    def forward(self, images: Tensor, boxes: Sequence[Tensor]) -> Dict[str, object]:
        if images.ndim != 4:
            raise ValueError("images must be [B,3,H,W]")
        if len(boxes) != images.shape[0]:
            raise ValueError("boxes list must match image batch")
        image_embeddings = self.sam.image_encoder(images)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        low_res_all: List[Tensor] = []
        iou_all: List[Tensor] = []
        class_all: List[Tensor] = []
        mask_token_all: List[Tensor] = []
        iou_token_all: List[Tensor] = []
        head_outputs: List[TQMOTPOutput] = []

        for batch_idx, boxes_i in enumerate(boxes):
            if boxes_i.numel() == 0:
                continue
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes_i, masks=None)
            low_res, iou_pred, mask_tokens, iou_token = predict_masks_with_raw_tokens(
                self.sam.mask_decoder,
                image_embeddings=image_embeddings[batch_idx : batch_idx + 1],
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
            )
            head_out = forward_tqmotp_one_image(
                self.head,
                image_embedding=image_embeddings[batch_idx : batch_idx + 1],
                processed_image=images[batch_idx : batch_idx + 1],
                boxes_1024=boxes_i,
                low_res_masks=low_res,
                iou_predictions=iou_pred,
                mask_tokens=mask_tokens,
                iou_token=iou_token,
                mask_index=self.mask_index,
            )
            low_res_all.append(low_res[:, self.mask_index : self.mask_index + 1])
            iou_all.append(iou_pred[:, self.mask_index : self.mask_index + 1])
            class_all.append(head_out.logits)
            mask_token_all.append(mask_tokens[:, self.mask_index])
            iou_token_all.append(iou_token)
            head_outputs.append(head_out)

        if not low_res_all:
            empty = images.new_empty((0, 1, 256, 256))
            return {
                "low_res_masks": empty,
                "iou_pred": images.new_empty((0, 1)),
                "class_logits": images.new_empty((0, self.head.num_classes)),
                "raw_mask_token": images.new_empty((0, self.head.config.token_dim)),
                "raw_iou_token": images.new_empty((0, self.head.config.token_dim)),
                "tqmotp_outputs": [],
                "image_embeddings": image_embeddings,
            }
        return {
            "low_res_masks": torch.cat(low_res_all, dim=0),
            "iou_pred": torch.cat(iou_all, dim=0),
            "class_logits": torch.cat(class_all, dim=0),
            "raw_mask_token": torch.cat(mask_token_all, dim=0),
            "raw_iou_token": torch.cat(iou_token_all, dim=0),
            "tqmotp_outputs": head_outputs,
            "image_embeddings": image_embeddings,
        }
