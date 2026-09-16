"""Unified SAM1 ViT-B training wrapper with Q/V LoRA + one classification head.

Supported heads (select via --head):
  mask_token    H1  MaskTokenClassificationHead
  iou_token     H2  IoUTokenClassificationHead
  pseco_roi     H3  PseCoROIClassificationHead
  masksam_token H4  MaskSAMClassifierTokenHead
  cwsam_decoder H5  CWSAMClasswiseDecoderHead

One model instance = one head.  Five experiments = five independent runs.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored dependencies resolve relative to the portable baseline root.
_PORTABLE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PORTABLE_ROOT / "vendor" / "segment-anything"))
from segment_anything.modeling import Sam, ImageEncoderViT, MaskDecoder, PromptEncoder
from segment_anything import sam_model_registry

# Classification heads
sys.path.insert(0, str(_PORTABLE_ROOT / "vendor" / "classification-heads"))
from sam_heads import (
    MaskTokenClassificationHead,
    IoUTokenClassificationHead,
    PseCoROIClassificationHead,
    MaskSAMClassifierTokenHead,
    CWSAMClasswiseDecoderHead,
)

# LoRA
from src.lora import inject_qv_lora_into_sam1, count_lora_parameters, FusedQKVLoRA

# The experiment carries an immutable, audited copy of RandLoRA v1.2.0.  It is
# deliberately imported from this isolated run rather than a globally installed
# package so that the run manifest can fingerprint the exact source used.
_RUN_ROOT = Path(__file__).resolve().parents[3]
_RANDLORA_ROOT = _RUN_ROOT / "vendor" / "DAG-RandLoRA-v1.0.0"
if str(_RANDLORA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RANDLORA_ROOT))
from randlora_damage import RandLoRADamageConfig, inject_randlora_damage_encoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAM1_CHECKPOINT = os.environ.get(
    "SAM1_CHECKPOINT",
    str(_PORTABLE_ROOT / "artifacts" / "sam_vit_b_01ec64.pth"),
)
NUM_DAMAGE_CLASSES = 8
TRANSFORMER_DIM = 256
MASK_CHANNEL_DIM = 32  # transformer_dim // 8
NUM_MASK_TOKENS = 4     # SAM default: 3 multimask + 1 output
MASK_TOKEN_INDEX = 0    # token 0 for multimask_output=False

HEAD_NAMES = {
    "mask_token":    "H1 MaskTokenClassificationHead",
    "iou_token":     "H2 IoUTokenClassificationHead",
    "pseco_roi":     "H3 PseCoROIClassificationHead",
    "masksam_token": "H4 MaskSAMClassifierTokenHead",
    "cwsam_decoder": "H5 CWSAMClasswiseDecoderHead",
}


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class Sam1TrainWrapper(nn.Module):
    """SAM1 ViT-B + Encoder Q/V LoRA + trainable Mask Decoder + one classif. head."""

    def __init__(
        self,
        head_name: str,
        *,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.05,
        adapter_type: str = "lora",
        randlora_config: Optional[Dict[str, Any]] = None,
        num_classes: int = NUM_DAMAGE_CLASSES,
        checkpoint_path: str = SAM1_CHECKPOINT,
    ):
        super().__init__()
        if head_name not in HEAD_NAMES:
            raise ValueError(f"Unknown head: {head_name}.  Choose from {list(HEAD_NAMES)}")
        self.head_name = head_name
        self.num_classes = int(num_classes)

        # ---- 1. Load SAM1 ViT-B ----
        sam: Sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.sam = sam
        self.image_encoder: ImageEncoderViT = sam.image_encoder
        self.prompt_encoder: PromptEncoder = sam.prompt_encoder
        self.mask_decoder: MaskDecoder = sam.mask_decoder

        # ---- 2. Freeze image encoder, prompt encoder ----
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        # ---- 3. Inject the selected encoder adapter *after* Meta SAM loads ----
        # V0 preserves the verified all-block Q/V LoRA baseline.  V1 removes it
        # completely and injects RandLoRA only into blocks 4--11 Q/V slices.
        self.adapter_type = str(adapter_type)
        self._lora_map = {}
        self.randlora_report = None
        if self.adapter_type == "lora":
            self._lora_map = inject_qv_lora_into_sam1(
                sam,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_blocks=list(range(12)),
            )
            self.lora_rank = lora_rank
            self.lora_total, self.lora_per_block = count_lora_parameters(sam)
        elif self.adapter_type == "randlora":
            cfg = RandLoRADamageConfig.from_dict(dict(randlora_config or {}))
            self.randlora_report = inject_randlora_damage_encoder(sam, cfg)
            self.lora_rank = 0
            self.lora_total = 0
            self.lora_per_block = {}
        else:
            raise ValueError(f"unsupported adapter_type={adapter_type!r}; expected 'lora' or 'randlora'")

        # ---- 4. Make mask decoder trainable ----
        for p in self.mask_decoder.parameters():
            p.requires_grad = True

        # ---- 5. Create classification head ----
        self.classification_head = self._build_head(head_name)

        # ---- 6. Count parameters ----
        self._param_counts = self._compute_param_counts()

    # ------------------------------------------------------------------
    # Head factory
    # ------------------------------------------------------------------
    def _build_head(self, name: str) -> nn.Module:
        if name == "mask_token":
            return MaskTokenClassificationHead(
                token_dim=TRANSFORMER_DIM,
                num_classes=self.num_classes,
            )
        elif name == "iou_token":
            return IoUTokenClassificationHead(
                token_dim=TRANSFORMER_DIM,
                num_classes=self.num_classes,
            )
        elif name == "pseco_roi":
            return PseCoROIClassificationHead(
                num_classes=self.num_classes,
                in_channels=TRANSFORMER_DIM,
                mode="closed_set",
            )
        elif name == "masksam_token":
            return MaskSAMClassifierTokenHead(
                transformer_dim=TRANSFORMER_DIM,
                num_classes=self.num_classes,
                include_no_object=False,
            )
        elif name == "cwsam_decoder":
            return CWSAMClasswiseDecoderHead(
                num_classes=self.num_classes,
                transformer_dim=TRANSFORMER_DIM,
            )
        raise ValueError(f"Unknown head: {name}")

    # ------------------------------------------------------------------
    # Core forward — returns all intermediates needed by any head
    # ------------------------------------------------------------------
    def forward_encoder(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, 1024, 1024] (already preprocessed) → [B, 256, 64, 64]."""
        return self.image_encoder(images)

    def forward_prompt_encoder(
        self,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """boxes: [N, 4] in XYXY, 1024 coordinate space → sparse, dense embeddings."""
        return self.prompt_encoder(points=None, boxes=boxes, masks=None)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        sparse_emb: torch.Tensor,
        dense_emb: torch.Tensor,
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run mask_decoder and return ALL intermediate tensors.

        Returns dict with:
          masks           [N, 1_or_3, 256, 256]
          iou_pred        [N, 1_or_3]
          iou_token_out   [N, 256]  raw token from transformer
          mask_tokens_out [N, M, 256]  raw mask tokens (M=4)
          upscaled_embedding [N, 32, 256, 256]
          mask_hypernet   [N, M, 32]
        """
        # We reproduce predict_masks to capture intermediates
        iou_tok = self.mask_decoder.iou_token.weight      # [1, 256]
        mask_toks = self.mask_decoder.mask_tokens.weight   # [M, 256]
        M = mask_toks.shape[0]
        output_tokens = torch.cat([iou_tok, mask_toks], dim=0)  # [1+M, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_emb.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_emb), dim=1)   # [N, 1+M+prompts, 256]

        # Expand image embeddings — one per prompt
        N = tokens.shape[0]
        src = image_embeddings.repeat(N, 1, 1, 1) if image_embeddings.shape[0] == 1 else image_embeddings
        src = src + dense_emb
        image_pe = self.prompt_encoder.get_dense_pe()
        pos_src = image_pe.repeat(N, 1, 1, 1) if image_pe.shape[0] == 1 else image_pe
        B, C, H, W = src.shape

        # Run transformer
        hs, src_out = self.mask_decoder.transformer(src, pos_src, tokens)

        # Split tokens
        iou_token_out = hs[:, 0, :]                        # [N, 256]
        mask_tokens_out = hs[:, 1 : (1 + M), :]            # [N, M, 256]

        # Upscale + hypernetworks
        src_img = src_out.transpose(1, 2).view(B, C, H, W)
        upscaled_embedding = self.mask_decoder.output_upscaling(src_img)  # [N, 32, 256, 256]

        hyper_in_list = []
        for i in range(M):
            hyper_in_list.append(self.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        mask_hypernet = torch.stack(hyper_in_list, dim=1)  # [N, M, 32]

        # Predicted masks
        _, C_up, H_up, W_up = upscaled_embedding.shape
        masks = (mask_hypernet @ upscaled_embedding.view(B, C_up, H_up * W_up)).view(B, M, H_up, W_up)

        # IoU prediction
        iou_pred_full = self.mask_decoder.iou_prediction_head(iou_token_out)  # [N, M]

        # Select output mask
        if multimask_output:
            masks_out = masks[:, 1:, :, :]        # [N, 3, 256, 256]
            iou_out = iou_pred_full[:, 1:]         # [N, 3]
        else:
            masks_out = masks[:, 0:1, :, :]        # [N, 1, 256, 256]
            iou_out = iou_pred_full[:, 0:1]         # [N, 1]

        return {
            "masks": masks_out,
            "iou_pred": iou_out,
            "iou_token_out": iou_token_out,
            "mask_tokens_out": mask_tokens_out,
            "upscaled_embedding": upscaled_embedding,
            "mask_hypernet": mask_hypernet,
            "masks_full": masks,                   # [N, M, 256, 256] — all tokens
            "iou_pred_full": iou_pred_full,        # [N, M]
        }

    def forward_decoder_with_classifier_token(
        self,
        image_embeddings: torch.Tensor,
        sparse_emb: torch.Tensor,
        dense_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """H4-specific forward: inject classifier token into the two-way transformer.

        Returns same keys as forward_decoder plus:
          transformed_classifier_token  [N, 256]
          classifier_token_index        int
        """
        iou_tok = self.mask_decoder.iou_token.weight
        mask_toks = self.mask_decoder.mask_tokens.weight
        M = mask_toks.shape[0]
        output_tokens = torch.cat([iou_tok, mask_toks], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_emb.shape[0], -1, -1)

        # Append classifier token BEFORE transformer
        head: MaskSAMClassifierTokenHead = self.classification_head
        tokens_with_cls, cls_index = head.append_to_decoder_tokens(
            torch.cat((output_tokens, sparse_emb), dim=1)
        )

        N = tokens_with_cls.shape[0]
        src = image_embeddings.repeat(N, 1, 1, 1) if image_embeddings.shape[0] == 1 else image_embeddings
        src = src + dense_emb
        image_pe = self.prompt_encoder.get_dense_pe()
        pos_src = image_pe.repeat(N, 1, 1, 1) if image_pe.shape[0] == 1 else image_pe
        B, C, H, W = src.shape

        hs, src_out = self.mask_decoder.transformer(src, pos_src, tokens_with_cls)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + M), :]
        classifier_token_out = hs[:, cls_index, :]

        src_img = src_out.transpose(1, 2).view(B, C, H, W)
        upscaled_embedding = self.mask_decoder.output_upscaling(src_img)

        hyper_in_list = []
        for i in range(M):
            hyper_in_list.append(self.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        mask_hypernet = torch.stack(hyper_in_list, dim=1)

        _, C_up, H_up, W_up = upscaled_embedding.shape
        masks = (mask_hypernet @ upscaled_embedding.view(B, C_up, H_up * W_up)).view(B, M, H_up, W_up)
        iou_pred_full = self.mask_decoder.iou_prediction_head(iou_token_out)

        # multimask_output=False for GT-box experiments
        masks_out = masks[:, 0:1, :, :]
        iou_out = iou_pred_full[:, 0:1]

        return {
            "masks": masks_out,
            "iou_pred": iou_out,
            "iou_token_out": iou_token_out,
            "mask_tokens_out": mask_tokens_out,
            "upscaled_embedding": upscaled_embedding,
            "mask_hypernet": mask_hypernet,
            "masks_full": masks,
            "iou_pred_full": iou_pred_full,
            "transformed_classifier_token": classifier_token_out,
            "classifier_token_index": cls_index,
        }

    # ------------------------------------------------------------------
    # Complete training forward (image → all outputs + classification logits)
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,           # [B, 3, 1024, 1024]
        boxes_list: List[torch.Tensor], # List of [Ni, 4] XYXY in 1024 space
        gt_masks_256: Optional[torch.Tensor] = None,  # [total_N, 256, 256] for mask loss
        labels: Optional[torch.Tensor] = None,         # [total_N]
    ) -> Dict[str, Any]:
        """Training forward for one batch.

        images:    [B, 3, 1024, 1024] preprocessed images
        boxes_list: per-image GT box tensors in 1024×1024 coordinate space
        gt_masks_256: GT masks at 256×256 (for mask loss)
        labels: contiguous class labels [total_N]

        Returns dict with logits, masks, iou_pred, and any intermediate tensors.
        """
        B = images.shape[0]
        image_embeddings = self.forward_encoder(images)  # [B, 256, 64, 64]

        is_h4 = self.head_name == "masksam_token"

        all_class_logits = []
        all_masks = []
        all_iou_pred = []
        # Provenance-preserving side channels for Q-series joint quality heads.
        # They are collected from the exact decoder invocation used for the
        # final mask/classification output; legacy consumers ignore them.
        all_selected_mask_tokens = []
        all_iou_tokens = []
        all_mask_uncertainty = []
        all_mask_guided_features = []
        all_mask_guided_weight_sums = []

        # Process each image's instances independently
        idx_offset = 0
        for img_idx in range(B):
            boxes_i = boxes_list[img_idx]  # [Ni, 4]
            Ni = boxes_i.shape[0]

            if Ni == 0:
                continue  # zero-instance image → skip (no prompt)

            # Single-image embedding
            img_emb_i = image_embeddings[img_idx:img_idx+1]  # [1, 256, 64, 64]

            # Prompt encoding (one prompt per instance, batched)
            sparse_emb, dense_emb = self.forward_prompt_encoder(boxes_i)  # [Ni, P, 256], [Ni, 256, 64, 64]

            # ------------------------------------------------------------------
            # Decoder forward
            # ------------------------------------------------------------------
            if is_h4:
                dec_out = self.forward_decoder_with_classifier_token(
                    img_emb_i, sparse_emb, dense_emb,
                )
                # H4 classification from transformed classifier token
                head: MaskSAMClassifierTokenHead = self.classification_head
                cls_logits_i = head.forward(dec_out["transformed_classifier_token"])  # [Ni, 8]
            else:
                dec_out = self.forward_decoder(
                    img_emb_i, sparse_emb, dense_emb, multimask_output=False,
                )

                if self.head_name == "mask_token":
                    cls_logits_i = self.classification_head(dec_out["mask_tokens_out"][:, MASK_TOKEN_INDEX, :])
                elif self.head_name == "iou_token":
                    cls_logits_i = self.classification_head(dec_out["iou_token_out"])
                elif self.head_name == "pseco_roi":
                    roi_out = self.classification_head(
                        img_emb_i, [boxes_i],
                    )
                    cls_logits_i = roi_out.logits  # [Ni, 8]
                elif self.head_name == "cwsam_decoder":
                    classwise = self.classification_head(
                        dec_out["upscaled_embedding"],
                        dec_out["mask_hypernet"],
                    )
                    # pool_instance_logits expects binary_mask_logits matching
                    # classwise [B,M,C,H,W].  Pass all M mask tokens and select
                    # token 0 inside pool_instance_logits.
                    cls_logits_i = self.classification_head.pool_instance_logits(
                        classwise,
                        binary_mask_logits=dec_out["masks_full"],
                        mask_token_index=0,
                    )
                else:
                    raise RuntimeError(f"Unhandled head: {self.head_name}")

            all_class_logits.append(cls_logits_i)
            all_masks.append(dec_out["masks"].squeeze(1))   # [Ni, 256, 256]
            all_iou_pred.append(dec_out["iou_pred"].squeeze(-1))  # [Ni]
            if not is_h4 and getattr(self, "enable_joint_tokens", False):
                all_selected_mask_tokens.append(dec_out["mask_tokens_out"][:, MASK_TOKEN_INDEX, :])
                all_iou_tokens.append(dec_out["iou_token_out"])
                if getattr(self, "enable_joint_features", False):
                    # Frozen-consistent uncertainty is explicitly detached by
                    # design (hard thresholds/stability statistics).  Import is
                    # lazy so legacy H1 users do not require Q-series paths.
                    from features.mask_uncertainty import extract_mask_uncertainty
                    mask_logits_i = dec_out["masks"]  # exact selected-mask tensor
                    all_mask_uncertainty.append(extract_mask_uncertainty(mask_logits_i))
                    soft = F.interpolate(mask_logits_i.sigmoid(), size=img_emb_i.shape[-2:],
                                         mode="bilinear", align_corners=False).squeeze(1)
                    weights = soft / soft.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                    # No repeat/expand of [C,H,W]; this path remains differentiable.
                    all_mask_guided_features.append(torch.einsum("nhw,chw->nc", weights, img_emb_i[0]))
                    all_mask_guided_weight_sums.append(weights.sum(dim=(-2, -1)))

            idx_offset += Ni

        # Concatenate across all images
        if all_class_logits:
            class_logits = torch.cat(all_class_logits, dim=0)       # [total_N, 8]
            pred_masks = torch.cat(all_masks, dim=0)                # [total_N, 256, 256]
            pred_iou = torch.cat(all_iou_pred, dim=0)               # [total_N]
            selected_mask_token = torch.cat(all_selected_mask_tokens, dim=0) if all_selected_mask_tokens else None
            iou_token = torch.cat(all_iou_tokens, dim=0) if all_iou_tokens else None
            mask_uncertainty = torch.cat(all_mask_uncertainty, dim=0) if all_mask_uncertainty else images.new_zeros((0, 8))
            mask_guided_feature = torch.cat(all_mask_guided_features, dim=0) if all_mask_guided_features else images.new_zeros((0, 256))
            mask_guided_weight_sum = torch.cat(all_mask_guided_weight_sums, dim=0) if all_mask_guided_weight_sums else images.new_zeros((0,))
        else:
            class_logits = images.new_zeros((0, self.num_classes))
            pred_masks = images.new_zeros((0, 256, 256))
            pred_iou = images.new_zeros((0,))
            selected_mask_token = None
            iou_token = None
            mask_uncertainty = images.new_zeros((0, 8))
            mask_guided_feature = images.new_zeros((0, 256))
            mask_guided_weight_sum = images.new_zeros((0,))

        return {
            "class_logits": class_logits,
            "pred_masks": pred_masks,
            "pred_iou": pred_iou,
            "selected_mask_token": selected_mask_token,
            "iou_token": iou_token,
            "mask_uncertainty": mask_uncertainty,
            "mask_guided_feature": mask_guided_feature,
            "mask_guided_weight_sum": mask_guided_weight_sum,
            "total_instances": class_logits.shape[0],
        }

    # ------------------------------------------------------------------
    # Training-mode helpers
    # ------------------------------------------------------------------
    def set_lora_dropout(self, enabled: bool):
        for blk in self.image_encoder.blocks:
            if isinstance(blk.attn.qkv, FusedQKVLoRA):
                blk.attn.qkv.set_lora_dropout_enabled(enabled)

    def get_param_groups(
        self,
        lora_lr: float = 1e-4,
        mask_decoder_lr: float = 5e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Return parameter groups for optimizer with per-group learning rates."""
        lora_params = []
        decoder_params = []
        head_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "lora" in name.lower() or (isinstance(
                getattr(self.image_encoder.blocks[int(name.split(".")[0])].attn if name.startswith(("image_encoder.blocks.")) else None, "qkv", None),
                FusedQKVLoRA,
            ) if False else False):
                pass  # handled below

        # Walk blocks for LoRA params
        for blk in self.image_encoder.blocks:
            if isinstance(blk.attn.qkv, FusedQKVLoRA):
                lora_params.extend(
                    p for p in blk.attn.qkv.q_lora.parameters() if p.requires_grad
                )
                lora_params.extend(
                    p for p in blk.attn.qkv.v_lora.parameters() if p.requires_grad
                )

        # Mask decoder params
        decoder_params.extend(
            p for p in self.mask_decoder.parameters() if p.requires_grad
        )

        # Classification head params
        head_params.extend(
            p for p in self.classification_head.parameters() if p.requires_grad
        )

        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay, "name": "lora"})
        if decoder_params:
            groups.append({"params": decoder_params, "lr": mask_decoder_lr, "weight_decay": weight_decay, "name": "mask_decoder"})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "classification_head"})

        return groups

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------
    def _compute_param_counts(self) -> Dict[str, int]:
        lora_total = self.lora_total
        decoder = sum(p.numel() for p in self.mask_decoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.classification_head.parameters() if p.requires_grad)
        encoder_frozen = sum(p.numel() for p in self.image_encoder.parameters() if not p.requires_grad)
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": lora_total + decoder + head,
            "frozen_encoder": encoder_frozen,
            "lora": lora_total,
            "mask_decoder": decoder,
            "classification_head": head,
        }

    @property
    def param_summary(self) -> Dict[str, int]:
        return dict(self._param_counts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_model(
    head_name: str,
    *,
    lora_rank: int = 4,
    lora_alpha: float = 4.0,
    lora_dropout: float = 0.05,
    adapter_type: str = "lora",
    randlora_config: Optional[Dict[str, Any]] = None,
    num_classes: int = NUM_DAMAGE_CLASSES,
    checkpoint_path: str = SAM1_CHECKPOINT,
) -> Sam1TrainWrapper:
    """Create a fresh, independently-initialised SAM1 + LoRA + head model."""
    return Sam1TrainWrapper(
        head_name=head_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapter_type=adapter_type,
        randlora_config=randlora_config,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )
