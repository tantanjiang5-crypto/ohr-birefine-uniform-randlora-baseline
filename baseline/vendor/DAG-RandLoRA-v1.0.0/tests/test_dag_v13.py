from __future__ import annotations

import math

import torch

from randlora_damage import (
    FusedQKVWeightGradientProfiler,
    RandLoRADamageConfig,
    allocate_target_ranks,
    build_r2_dag_plan,
    canonical_target_key,
    entropy_effective_rank,
    inject_randlora_damage_encoder,
    randlora_trainable_params,
)
from randlora_damage.core import FusedQKVRandLoRA
from tests.mock_sam import MockSAM


def test_target_specific_qv_rank_injection():
    model = MockSAM(dim=16, depth=12)
    cfg = RandLoRADamageConfig(
        block_indices=(4,),
        qkv_targets=("q", "v"),
        auto_match_budget=False,
        rank=8,
        target_trainable_params=None,
        target_rank_pattern={"4:q": 4, "4:v": 8},
    )
    report = inject_randlora_damage_encoder(model, cfg)
    wrapper = model.image_encoder.blocks[4].attn.qkv
    assert isinstance(wrapper, FusedQKVRandLoRA)
    assert wrapper.coefficients["q"].rank == 4
    assert wrapper.coefficients["v"].rank == 8
    assert report.target_ranks == {"4:q": 4, "4:v": 8}
    expected = randlora_trainable_params(16, 16, 4) + randlora_trainable_params(16, 16, 8)
    assert report.actual_adapter_params == expected
    assert wrapper.coefficients["q"].scaling == wrapper.coefficients["v"].scaling == 2.0


def test_gradient_profiler_matches_exact_weight_gradient():
    torch.manual_seed(0)
    model = MockSAM(dim=8, depth=2)
    model.image_encoder.requires_grad_(False)
    linear = model.image_encoder.blocks[0].attn.qkv
    assert not linear.weight.requires_grad

    with FusedQKVWeightGradientProfiler(model, block_indices=(0,), targets=("q", "v")) as profiler:
        x = torch.randn(2, 5, 8)
        y = model.image_encoder(x)
        loss = y.square().mean()
        loss.backward()
        grad = linear.weight.grad.detach().clone()
        expected_q = grad[:8]
        expected_v = grad[16:24]
        profiler.update_after_backward(sample_weight=1.0)
        matrices = profiler.matrices()
        assert torch.allclose(matrices["0:q"], expected_q)
        assert torch.allclose(matrices["0:v"], expected_v)
    assert not linear.weight.requires_grad
    assert linear.weight.grad is None


def test_entropy_effective_rank_behaves_as_gid():
    rank1 = torch.zeros(8, 8)
    rank1[:, 0] = torch.arange(1, 9, dtype=torch.float32)
    identity = torch.eye(8)
    er1 = entropy_effective_rank(rank1)
    eri = entropy_effective_rank(identity)
    assert math.isclose(er1, 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(eri, 8.0, rel_tol=1e-5, abs_tol=1e-5)


def test_target_rank_allocator_prioritizes_high_score_under_same_budget():
    keys = ["4:q", "4:v", "5:q", "5:v"]
    shapes = {key: (16, 16) for key in keys}
    baseline_rank = 8
    target = len(keys) * randlora_trainable_params(16, 16, baseline_rank)
    scores = {"4:q": 0.70, "4:v": 0.10, "5:q": 0.10, "5:v": 0.10}
    pattern = allocate_target_ranks(
        scores,
        target_shapes=shapes,
        baseline_rank=baseline_rank,
        target_total_params=target,
        budget_tolerance=0.20,
        rank_candidates=(4, 8, 16),
    )
    high_cost = randlora_trainable_params(16, 16, pattern["4:q"])
    low_costs = [randlora_trainable_params(16, 16, pattern[key]) for key in keys[1:]]
    assert high_cost >= max(low_costs)
    total = sum(randlora_trainable_params(16, 16, pattern[key]) for key in keys)
    assert abs(total - target) / target <= 0.20


def test_r2_plan_uses_gid_and_specificity_and_stays_budgeted():
    keys = [canonical_target_key(b, t) for b in (4, 5) for t in ("q", "v")]
    hard = {}
    general = {}
    for key in keys:
        hard[key] = torch.eye(16)
        general[key] = torch.eye(16)
    # Make 4:q both high-dimensional and hard-specific.
    hard["4:q"] = 4.0 * torch.eye(16)
    general["4:q"] = 0.25 * torch.eye(16)
    # Make 5:v almost rank-1 and non-specific.
    hard["5:v"] = torch.ones(16, 1) @ torch.ones(1, 16)
    general["5:v"] = hard["5:v"].clone()

    baseline_rank = 8
    target = len(keys) * randlora_trainable_params(16, 16, baseline_rank)
    plan = build_r2_dag_plan(
        hard,
        general_gradients=general,
        baseline_rank=baseline_rank,
        target_total_params=target,
        rank_candidates=(4, 8, 16),
        budget_tolerance=0.20,
    )
    assert plan.stage == "R2-DAG"
    assert plan.scores["4:q"] > plan.scores["5:v"]
    high_cost = randlora_trainable_params(16, 16, plan.target_rank_pattern["4:q"])
    low_cost = randlora_trainable_params(16, 16, plan.target_rank_pattern["5:v"])
    assert high_cost >= low_cost
    assert plan.budget_relative_error <= 0.20


def test_config_target_pattern_roundtrip():
    cfg = RandLoRADamageConfig(
        block_indices=(4, 5),
        qkv_targets=("q", "v"),
        auto_match_budget=False,
        rank=8,
        target_trainable_params=None,
        target_rank_pattern={"4:q": 4, "5:v": 16},
    )
    restored = RandLoRADamageConfig.from_dict(cfg.to_dict())
    assert restored.target_rank_pattern == cfg.target_rank_pattern
    assert restored.rank_for_target(4, "q", 8) == 4
    assert restored.rank_for_target(4, "v", 8) == 8


def test_probe_runner_restores_model_flags_and_mode():
    from randlora_damage import collect_qv_gradient_profile

    torch.manual_seed(0)
    model = MockSAM(dim=8, depth=2)
    model.train(True)
    original = {name: p.requires_grad for name, p in model.named_parameters()}
    batches = [torch.randn(2, 5, 8), torch.randn(2, 5, 8)]

    def loss_fn(m, batch):
        y = m.image_encoder(batch)
        return y.square().mean(), int(batch.shape[0])

    matrices, report = collect_qv_gradient_profile(
        model,
        batches,
        loss_fn,
        block_indices=(0, 1),
        max_instances=4,
        max_batches=4,
    )
    assert report.selected_instances == 4
    assert set(matrices) == {"0:q", "0:v", "1:q", "1:v"}
    assert model.training is True
    assert {name: p.requires_grad for name, p in model.named_parameters()} == original


def test_target_specific_checkpoint_roundtrip(tmp_path):
    from randlora_damage import load_adapter_checkpoint, save_adapter_checkpoint

    torch.manual_seed(3)
    cfg = RandLoRADamageConfig(
        block_indices=(4,),
        qkv_targets=("q", "v"),
        auto_match_budget=False,
        rank=8,
        target_trainable_params=None,
        target_rank_pattern={"4:q": 4, "4:v": 8},
        seed=9,
    )
    source = MockSAM(dim=16, depth=12)
    target = MockSAM(dim=16, depth=12)
    target.load_state_dict(source.state_dict())
    inject_randlora_damage_encoder(source, cfg)
    inject_randlora_damage_encoder(target, cfg)
    wrapper = source.image_encoder.blocks[4].attn.qkv
    with torch.no_grad():
        wrapper.coefficients["q"].randlora_lambda.normal_()
        wrapper.coefficients["v"].randlora_lambda.normal_()
    path = save_adapter_checkpoint(source, tmp_path / "dag.pt")
    info = load_adapter_checkpoint(target, path, strict=True)
    assert info["integrity_verified"] is True
    for (name_a, p_a), (name_b, p_b) in zip(
        source.image_encoder.named_parameters(), target.image_encoder.named_parameters()
    ):
        if "randlora_" in name_a:
            assert name_a == name_b
            assert torch.equal(p_a, p_b)
