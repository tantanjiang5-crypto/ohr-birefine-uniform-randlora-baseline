from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from randlora_damage import (
    ExperimentMetrics,
    RandLoRADamageConfig,
    add_randlora_to_optimizer,
    adapter_state_dict,
    allocate_block_ranks,
    append_randlora_param_group,
    box_background_false_positive_rate,
    evaluate_experiment_gate,
    inject_randlora_damage_encoder,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from randlora_damage.checkpoint import (
    _basis_fingerprints,
    _manifest,
    _state_digest,
    _validate_copied_basis_links,
)
from randlora_damage.core import FusedQKVRandLoRA, RandLoRABasisBank, RandLoRACoefficients
from tests.mock_sam import MockSAM


def fixed_config(**kwargs):
    values = dict(
        block_indices=(4,),
        rank=4,
        target_trainable_params=None,
        auto_match_budget=False,
    )
    values.update(kwargs)
    return RandLoRADamageConfig(**values)


def make_coeff(bank, dim=8, rank=2, mode="materialized", preserve_float32=False):
    return RandLoRACoefficients(
        in_features=dim,
        out_features=dim,
        rank=rank,
        alpha=2 * rank,
        dropout=0.0,
        forward_mode=mode,
        cache_eval_delta=True,
        basis_bank=bank,
        preserve_float32=preserve_float32,
    )


def rewrite_digest(payload):
    payload["state_digest"] = _state_digest(
        payload["state_dict"],
        config=payload["config"],
        manifest=tuple(dict(x) for x in payload["manifest"]),
        resolved_default_rank=payload["resolved_default_rank"],
        basis_fingerprints=payload["basis_fingerprints"],
        metadata=payload.get("metadata", {}),
    )


def test_eval_no_grad_cache_does_not_detach_later_eval_backward():
    bank = RandLoRABasisBank(rank=2, max_dim=8, min_dim=8, seed=1)
    coeff = make_coeff(bank)
    with torch.no_grad():
        coeff.randlora_lambda.normal_()
    coeff.eval()
    with torch.no_grad():
        cached = coeff.delta_weight()
    assert not cached.requires_grad

    x = torch.randn(2, 3, 8)
    loss = coeff(x).square().mean()
    loss.backward()
    assert coeff.randlora_lambda.grad is not None
    assert coeff.randlora_gamma.grad is not None
    assert coeff.randlora_lambda.grad.abs().sum() > 0
    assert coeff.randlora_gamma.grad.abs().sum() > 0


def test_strict_checkpoint_rejects_base_encoder_weight_even_with_valid_digest(tmp_path: Path):
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config())
    path = save_adapter_checkpoint(model, tmp_path / "adapter.pt")
    payload = torch.load(path, weights_only=True)
    base_key = next(
        key
        for key in model.image_encoder.state_dict()
        if key.endswith("attn.qkv.base_layer.weight")
    )
    payload["state_dict"][base_key] = model.image_encoder.state_dict()[base_key].clone()
    rewrite_digest(payload)
    tampered = tmp_path / "contains_base.pt"
    torch.save(payload, tampered)
    original = model.image_encoder.state_dict()[base_key].clone()
    with pytest.raises(RuntimeError, match="forbidden_non_adapter_keys"):
        load_adapter_checkpoint(model, tampered, strict=True)
    torch.testing.assert_close(model.image_encoder.state_dict()[base_key], original, rtol=0, atol=0)


def test_missing_adapter_key_detected_even_if_digest_is_recomputed(tmp_path: Path):
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config())
    path = save_adapter_checkpoint(model, tmp_path / "adapter.pt")
    payload = torch.load(path, weights_only=True)
    removed = next(key for key in payload["state_dict"] if "randlora_lambda" in key)
    payload["state_dict"].pop(removed)
    rewrite_digest(payload)
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(RuntimeError, match="missing_from_checkpoint"):
        load_adapter_checkpoint(model, broken, strict=True)


def test_config_scaling_mismatch_is_strictly_rejected(tmp_path: Path):
    source = MockSAM(dim=16)
    inject_randlora_damage_encoder(source, fixed_config(alpha_multiplier=2.0))
    path = save_adapter_checkpoint(source, tmp_path / "adapter.pt")
    target = MockSAM(dim=16)
    inject_randlora_damage_encoder(target, fixed_config(alpha_multiplier=3.0))
    with pytest.raises(RuntimeError, match="config_mismatch"):
        load_adapter_checkpoint(target, path, strict=True)


def test_deepcopy_and_whole_model_pickle_keep_copied_basis_links(tmp_path: Path):
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config())
    copied = copy.deepcopy(model)
    _validate_copied_basis_links(copied)
    source_bank = next(iter(model.image_encoder._randlora_basis_banks.values()))
    copied_bank = next(iter(copied.image_encoder._randlora_basis_banks.values()))
    assert source_bank is not copied_bank

    path = tmp_path / "whole_model.pt"
    torch.save(model, path)
    restored = torch.load(path, weights_only=False)
    _validate_copied_basis_links(restored)


def test_adapter_float32_survives_parent_half_and_forward():
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config(adapter_dtype="float32"))
    model.half()
    wrapper = model.image_encoder.blocks[4].attn.qkv
    assert wrapper.base_layer.weight.dtype == torch.float16
    for coeff in wrapper.coefficients.values():
        assert coeff.randlora_lambda.dtype == torch.float32
        assert coeff.randlora_gamma.dtype == torch.float32
        assert coeff.basis_bank.basis_a.dtype == torch.float32
        assert coeff.basis_bank.basis_b.dtype == torch.float32
    output = model(torch.randn(2, 3, 16, dtype=torch.float16))[0]
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_rank_allocation_actually_uses_priority_within_tolerance():
    ranks = allocate_block_ranks(
        {4: 0.9, 5: 0.1},
        dim=16,
        adapters_per_block=2,
        baseline_rank=4,
        budget_tolerance=0.25,
        rank_candidates=(2, 4, 8),
    )
    assert ranks == {4: 2, 5: 8}


def test_adapter_state_dict_is_a_snapshot_not_cpu_alias():
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config())
    snapshot = adapter_state_dict(model)
    key = next(key for key in snapshot if "randlora_lambda" in key)
    before = snapshot[key].clone()
    with torch.no_grad():
        dict(model.image_encoder.named_parameters())[key].add_(1.0)
    torch.testing.assert_close(snapshot[key], before, rtol=0, atol=0)


def test_optimizer_generator_is_not_consumed_and_live_optimizer_api_works():
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config())
    head_params = list(model.classification_head.parameters())
    groups = [{"params": (param for param in head_params), "lr": 3e-4}]
    append_randlora_param_group(groups, model)
    assert isinstance(groups[0]["params"], list)
    assert groups[0]["params"] == head_params

    optimizer = torch.optim.AdamW(model.quality_head.parameters(), lr=2e-4)
    add_randlora_to_optimizer(optimizer, model, encoder_lr=1e-4)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[1]["group_name"] == "randlora_encoder"


def test_merge_unmerge_is_bitwise_exact_and_train_auto_unmerges():
    dim = 8
    base = nn.Linear(dim, 3 * dim).half()
    original = base.weight.detach().clone()
    bank = RandLoRABasisBank(rank=2, max_dim=dim, min_dim=dim, seed=2)
    wrapped = FusedQKVRandLoRA(
        base,
        {"q": make_coeff(bank), "v": make_coeff(bank)},
    )
    with torch.no_grad():
        for coeff in wrapped.coefficients.values():
            coeff.randlora_lambda.normal_()
    wrapped.merge()
    wrapped.train(True)
    assert not wrapped.merged
    assert torch.equal(base.weight, original)


def test_injection_failure_is_transactional():
    model = MockSAM(dim=16)
    original_requires_grad = {
        name: param.requires_grad for name, param in model.image_encoder.named_parameters()
    }
    model.image_encoder.blocks[5].attn.qkv = nn.Linear(16, 16)
    with pytest.raises(TypeError, match="fused qkv"):
        inject_randlora_damage_encoder(
            model,
            fixed_config(block_indices=(4, 5)),
        )
    assert isinstance(model.image_encoder.blocks[4].attn.qkv, nn.Linear)
    assert not hasattr(model.image_encoder, "_randlora_basis_banks")
    assert {
        name: param.requires_grad for name, param in model.image_encoder.named_parameters()
    } == original_requires_grad


def test_diagnostics_and_gate_reject_ambiguous_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="batch"):
        box_background_false_positive_rate(
            torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, 2)
        )
    mask = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="threshold"):
        box_background_false_positive_rate(mask, mask, mask, threshold=1.5)

    baseline = ExperimentMetrics(0.2, 0.4, 0.8, 0.1, 1.0)
    candidate = ExperimentMetrics(math.nan, 0.4, 0.8, 0.1, 1.0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_experiment_gate(baseline, candidate, epoch=1)


def test_duplicate_targets_and_ambiguous_rank_budget_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        fixed_config(qkv_targets=("q", "q")).validate(num_blocks=12)
    with pytest.raises(ValueError, match="ambiguous"):
        RandLoRADamageConfig(rank=8, target_trainable_params=1000, auto_match_budget=True).validate(
            num_blocks=12
        )


def test_custom_factor_scaling_backward_matches_plain_autograd():
    from randlora_damage.core import _ScaleSharedBasis

    torch.manual_seed(3)
    basis = torch.randn(2, 1, 4, dtype=torch.double)
    lamb_custom = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    gamma_custom = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    lamb_plain = lamb_custom.detach().clone().requires_grad_(True)
    gamma_plain = gamma_custom.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 2, 4, dtype=torch.double)

    (_ScaleSharedBasis.apply(basis, lamb_custom, gamma_custom) * upstream).sum().backward()
    (basis * lamb_plain[:, :, None] * gamma_plain[None, :, :] * upstream).sum().backward()
    torch.testing.assert_close(lamb_custom.grad, lamb_plain.grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gamma_custom.grad, gamma_plain.grad, rtol=1e-10, atol=1e-12)


def test_zero_lambda_gives_lambda_gradient_but_initial_gamma_gradient_zero():
    bank = RandLoRABasisBank(rank=2, max_dim=8, min_dim=8, seed=5)
    coeff = make_coeff(bank, mode="factorized")
    loss = coeff(torch.randn(2, 3, 8)).square().sum()
    # A square loss at exact zero output also has zero upstream. Use a linear
    # objective to verify the intended no-op initialization gradient behavior.
    coeff.zero_grad(set_to_none=True)
    coeff(torch.randn(2, 3, 8)).sum().backward()
    assert coeff.randlora_lambda.grad is not None
    assert coeff.randlora_lambda.grad.abs().sum() > 0
    assert coeff.randlora_gamma.grad is not None
    assert torch.equal(coeff.randlora_gamma.grad, torch.zeros_like(coeff.randlora_gamma.grad))


def test_torch_compile_eager_backend_smoke():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(model, fixed_config())
    compiled = torch.compile(model, backend="eager")
    loss = compiled(torch.randn(2, 3, 16))[0].sum()
    loss.backward()
    assert torch.isfinite(loss)
