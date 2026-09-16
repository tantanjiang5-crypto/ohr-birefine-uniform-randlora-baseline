from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from randlora_damage import (
    RandLoRADamageConfig,
    append_randlora_param_group,
    box_background_false_positive_rate,
    damage_priority_scores,
    inject_randlora_damage_encoder,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    set_adapters_enabled,
)
from randlora_damage.core import FusedQKVRandLoRA, RandLoRABasisBank, RandLoRACoefficients
from tests.mock_sam import MockSAM


def make_coeff(bank, dim=8, rank=2, mode="materialized"):
    return RandLoRACoefficients(
        in_features=dim,
        out_features=dim,
        rank=rank,
        alpha=2 * rank,
        dropout=0.0,
        forward_mode=mode,
        cache_eval_delta=True,
        basis_bank=bank,
    )


def test_eval_cache_invalidates_after_parameter_change():
    bank = RandLoRABasisBank(rank=2, max_dim=8, min_dim=8, seed=1)
    coeff = make_coeff(bank)
    coeff.eval()
    first = coeff.delta_weight().clone()
    with torch.no_grad():
        coeff.randlora_lambda.normal_()
    second = coeff.delta_weight().clone()
    assert not torch.equal(first, second)


def test_eval_cache_invalidates_after_state_dict_load():
    bank = RandLoRABasisBank(rank=2, max_dim=8, min_dim=8, seed=1)
    coeff = make_coeff(bank)
    coeff.eval()
    _ = coeff.delta_weight()
    state = coeff.state_dict()
    state["randlora_lambda"] = torch.randn_like(state["randlora_lambda"])
    coeff.load_state_dict(state)
    expected = coeff.delta_weight().clone()
    coeff.clear_cache()
    torch.testing.assert_close(coeff.delta_weight(), expected)


def test_unmerge_subtracts_exact_original_delta_after_coeff_change():
    torch.manual_seed(0)
    dim = 8
    base = nn.Linear(dim, 3 * dim)
    original = base.weight.detach().clone()
    bank = RandLoRABasisBank(rank=2, max_dim=dim, min_dim=dim, seed=2)
    wrapped = FusedQKVRandLoRA(base, {"q": make_coeff(bank), "v": make_coeff(bank)})
    with torch.no_grad():
        for coeff in wrapped.coefficients.values():
            coeff.randlora_lambda.normal_()
    wrapped.merge()
    with torch.no_grad():
        for coeff in wrapped.coefficients.values():
            coeff.randlora_lambda.add_(3.0)
    wrapped.unmerge()
    torch.testing.assert_close(base.weight, original, rtol=0, atol=1e-6)


def test_disabling_merged_adapter_unmerges_it():
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(
        model,
        RandLoRADamageConfig(
            block_indices=(4,), rank=4, target_trainable_params=None, auto_match_budget=False
        ),
    )
    wrapper = model.image_encoder.blocks[4].attn.qkv
    with torch.no_grad():
        wrapper.coefficients["q"].randlora_lambda.normal_()
    wrapper.merge()
    assert wrapper.merged
    set_adapters_enabled(model, False)
    assert not wrapper.merged and not wrapper.adapters_enabled


def test_checkpoint_is_portable_between_full_model_and_encoder(tmp_path: Path):
    config = RandLoRADamageConfig(
        block_indices=(4, 5), rank=4, target_trainable_params=None, auto_match_budget=False
    )
    source = MockSAM(dim=16)
    inject_randlora_damage_encoder(source, config)
    with torch.no_grad():
        for name, p in source.named_parameters():
            if "randlora_lambda" in name:
                p.normal_()
    path = save_adapter_checkpoint(source, tmp_path / "adapter.pt")

    target = MockSAM(dim=16)
    inject_randlora_damage_encoder(target, config)
    info = load_adapter_checkpoint(target.image_encoder, path, strict=True)
    assert info["loaded_keys"]
    for key, value in source.image_encoder.state_dict().items():
        if "randlora_" in key or "_randlora_basis_banks" in key:
            torch.testing.assert_close(target.image_encoder.state_dict()[key], value)


def test_strict_checkpoint_detects_missing_current_adapter_key(tmp_path: Path):
    config = RandLoRADamageConfig(
        block_indices=(4,), rank=4, target_trainable_params=None, auto_match_budget=False
    )
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, config)
    path = save_adapter_checkpoint(model, tmp_path / "complete.pt")
    payload = torch.load(path, weights_only=True)
    removed = next(key for key in payload["state_dict"] if "randlora_lambda" in key)
    payload["state_dict"].pop(removed)
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(RuntimeError, match="integrity failure"):
        load_adapter_checkpoint(model, broken, strict=True)


def test_float_logits_require_explicit_flag_and_are_thresholded_correctly():
    logits = torch.tensor([[[[-10.0, 10.0], [-10.0, 10.0]]]])
    gt = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.bool)
    box = torch.ones_like(gt)
    with pytest.raises(ValueError, match="from_logits=True"):
        box_background_false_positive_rate(logits, gt, box)
    result = box_background_false_positive_rate(logits, gt, box, pred_from_logits=True)
    # One false positive among three box-background pixels.
    torch.testing.assert_close(result, torch.tensor([1.0 / 3.0]))


def test_append_optimizer_group_preserves_existing_groups_and_rejects_duplicates():
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(
        model,
        RandLoRADamageConfig(
            block_indices=(4,), rank=4, target_trainable_params=None, auto_match_budget=False
        ),
    )
    groups = [{"params": list(model.classification_head.parameters()), "lr": 3e-4}]
    append_randlora_param_group(groups, model, encoder_lr=1e-4)
    assert groups[0]["lr"] == 3e-4
    assert groups[1]["group_name"] == "randlora_encoder"
    with pytest.raises(ValueError, match="already exist"):
        append_randlora_param_group(groups, model, encoder_lr=1e-4)


def test_damage_priority_penalizes_classification_sensitive_layer():
    scores = damage_priority_scores(
        {4: 0.8, 5: 0.2},
        classification_scores={4: 0.9, 5: 0.1},
        classification_penalty=0.8,
    )
    assert scores[5] > scores[4]


def test_auto_mode_training_and_eval_agree():
    torch.manual_seed(1)
    bank = RandLoRABasisBank(rank=2, max_dim=8, min_dim=8, seed=3)
    coeff = make_coeff(bank, mode="auto")
    with torch.no_grad():
        coeff.randlora_lambda.normal_()
        coeff.randlora_gamma.normal_()
    x = torch.randn(2, 5, 8)
    coeff.train()
    train_out = coeff(x)
    coeff.eval()
    eval_out = coeff(x)
    torch.testing.assert_close(train_out, eval_out, rtol=2e-4, atol=1e-5)
