from __future__ import annotations

import torch
from torch import nn

from randlora_damage.core import FusedQKVRandLoRA, RandLoRABasisBank, RandLoRACoefficients


def make_coeff(bank, dim, rank, mode="materialized"):
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


def test_paper_formula_matches_manual_sum():
    torch.manual_seed(0)
    dim, rank = 6, 2
    bank = RandLoRABasisBank(rank=rank, max_dim=dim, min_dim=dim, seed=7)
    coeff = make_coeff(bank, dim, rank)
    with torch.no_grad():
        coeff.randlora_lambda.normal_()
        coeff.randlora_gamma.normal_()
    delta = coeff.delta_weight()
    manual = torch.zeros_like(delta)
    a = bank.basis_a[:, 0, :dim]
    b = bank.basis_b[:dim, : coeff.num_bases, :]
    for j in range(coeff.num_bases):
        manual += (
            b[:, j, :]
            @ torch.diag(coeff.randlora_lambda[:, j])
            @ a
            @ torch.diag(coeff.randlora_gamma[j])
        )
    manual *= coeff.scaling
    torch.testing.assert_close(delta, manual, rtol=1e-5, atol=1e-6)


def test_fused_qkv_initial_noop_and_k_never_changes():
    torch.manual_seed(1)
    dim, rank = 8, 2
    base = nn.Linear(dim, 3 * dim)
    bank = RandLoRABasisBank(rank=rank, max_dim=dim, min_dim=dim, seed=3)
    wrapped = FusedQKVRandLoRA(
        base,
        {"q": make_coeff(bank, dim, rank), "v": make_coeff(bank, dim, rank)},
    )
    x = torch.randn(2, 5, dim)
    before = base(x)
    torch.testing.assert_close(wrapped(x), before)

    with torch.no_grad():
        wrapped.coefficients["q"].randlora_lambda.normal_()
        wrapped.coefficients["v"].randlora_lambda.normal_()
    after = wrapped(x)
    bq, bk, bv = before.chunk(3, dim=-1)
    aq, ak, av = after.chunk(3, dim=-1)
    assert not torch.allclose(aq, bq)
    torch.testing.assert_close(ak, bk, rtol=0, atol=0)
    assert not torch.allclose(av, bv)


def test_factorized_and_materialized_agree():
    torch.manual_seed(2)
    dim, rank = 12, 3
    bank = RandLoRABasisBank(rank=rank, max_dim=dim, min_dim=dim, seed=9)
    a = make_coeff(bank, dim, rank, "materialized")
    b = make_coeff(bank, dim, rank, "factorized")
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.randlora_lambda.normal_()
        a.randlora_gamma.normal_()
        b.load_state_dict(a.state_dict())
    x = torch.randn(2, 7, dim)
    torch.testing.assert_close(a(x), b(x), rtol=2e-4, atol=1e-5)


def test_merge_unmerge_roundtrip():
    torch.manual_seed(3)
    dim, rank = 8, 2
    base = nn.Linear(dim, 3 * dim)
    bank = RandLoRABasisBank(rank=rank, max_dim=dim, min_dim=dim, seed=4)
    wrapped = FusedQKVRandLoRA(
        base,
        {"q": make_coeff(bank, dim, rank), "v": make_coeff(bank, dim, rank)},
    )
    with torch.no_grad():
        for coeff in wrapped.coefficients.values():
            coeff.randlora_lambda.normal_()
    x = torch.randn(3, 4, dim)
    expected = wrapped(x)
    original_weight = base.weight.detach().clone()
    wrapped.merge()
    torch.testing.assert_close(wrapped(x), expected, rtol=2e-5, atol=2e-6)
    wrapped.unmerge()
    torch.testing.assert_close(base.weight, original_weight, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(wrapped(x), expected, rtol=2e-5, atol=2e-6)


def test_zero_update_rank_diagnostics():
    from randlora_damage import matrix_rank_diagnostics

    report = matrix_rank_diagnostics(torch.zeros(8, 8))
    assert report.numerical_rank == 0
    assert report.effective_rank == 0.0
    assert report.spectral_energy_90_rank == 0
