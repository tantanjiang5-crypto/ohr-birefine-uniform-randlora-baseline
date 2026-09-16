from __future__ import annotations

from pathlib import Path

import torch

from randlora_damage import (
    RandLoRADamageConfig,
    audit_trainable_image_encoder,
    inject_randlora_damage_encoder,
    load_adapter_checkpoint,
    merge_and_unload,
    save_adapter_checkpoint,
)
from randlora_damage.core import FusedQKVRandLoRA
from tests.mock_sam import MockSAM


def test_budget_match_and_freezing():
    model = MockSAM(dim=16)
    # LoRA-r2 Q/V on all 12 mock blocks: 12*2*(16+16)*2 = 1536.
    config = RandLoRADamageConfig(
        block_indices=tuple(range(4, 12)),
        target_trainable_params=1536,
        rank=None,
        auto_match_budget=True,
    )
    report = inject_randlora_damage_encoder(model, config)
    assert report.actual_adapter_params > 0
    # The small mock dimension has coarse discrete budget steps; rank=5 is the
    # closest feasible value (1,344 params). Real ViT-B resolves to 147,488.
    assert report.default_rank == 5
    assert report.actual_adapter_params == 1344
    audit = audit_trainable_image_encoder(model)
    assert audit["pass"], audit
    assert all(p.requires_grad for p in model.classification_head.parameters())
    assert all(p.requires_grad for p in model.quality_head.parameters())
    for idx in range(4, 12):
        assert isinstance(model.image_encoder.blocks[idx].attn.qkv, FusedQKVRandLoRA)


def test_only_adapter_and_existing_heads_receive_gradients():
    torch.manual_seed(5)
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(
        model,
        RandLoRADamageConfig(
            block_indices=(4, 5), rank=4, target_trainable_params=None, auto_match_budget=False
        ),
    )
    x = torch.randn(2, 6, 16)
    logits, quality = model(x)
    loss = logits.square().mean() + quality.square().mean()
    loss.backward()
    image_grads = {
        n: p.grad for n, p in model.image_encoder.named_parameters() if p.requires_grad
    }
    assert image_grads
    assert all(g is not None and torch.isfinite(g).all() for g in image_grads.values())
    frozen_base_grads = [
        p.grad
        for n, p in model.image_encoder.named_parameters()
        if "base_layer" in n and not p.requires_grad
    ]
    assert frozen_base_grads and all(g is None for g in frozen_base_grads)


def test_adapter_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(6)
    config = RandLoRADamageConfig(
        block_indices=(4, 5), rank=4, target_trainable_params=None, auto_match_budget=False
    )
    source = MockSAM(dim=16)
    inject_randlora_damage_encoder(source, config)
    with torch.no_grad():
        for name, p in source.named_parameters():
            if "randlora_lambda" in name:
                p.normal_()
    checkpoint = save_adapter_checkpoint(source, tmp_path / "adapter.pt", metadata={"epoch": 3})

    target = MockSAM(dim=16)
    target.load_state_dict(
        {
            k: v
            for k, v in source.state_dict().items()
            if "_randlora" not in k and "randlora_" not in k and ".base_layer." not in k
        },
        strict=False,
    )
    # Use the exact same unadapted initialization for a strict output comparison.
    target = MockSAM(dim=16)
    base_state = MockSAM(dim=16).state_dict()
    source2 = MockSAM(dim=16)
    source2.load_state_dict(base_state)
    inject_randlora_damage_encoder(source2, config)
    with torch.no_grad():
        for name, p in source2.named_parameters():
            if "randlora_lambda" in name:
                p.normal_()
    checkpoint = save_adapter_checkpoint(source2, tmp_path / "adapter2.pt")
    target.load_state_dict(base_state)
    inject_randlora_damage_encoder(target, config)
    info = load_adapter_checkpoint(target, checkpoint)
    assert info["loaded_keys"]
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(source2(x)[0], target(x)[0], rtol=1e-5, atol=1e-6)


def test_merge_and_unload_restores_plain_sam_modules():
    torch.manual_seed(7)
    model = MockSAM(dim=16)
    inject_randlora_damage_encoder(
        model,
        RandLoRADamageConfig(
            block_indices=(4,), rank=4, target_trainable_params=None, auto_match_budget=False
        ),
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "randlora_lambda" in name:
                p.normal_()
    x = torch.randn(2, 5, 16)
    expected = model(x)[0]
    merge_and_unload(model)
    assert not any(isinstance(m, FusedQKVRandLoRA) for m in model.modules())
    torch.testing.assert_close(model(x)[0], expected, rtol=2e-5, atol=2e-6)


def test_optional_second_stage_targets_are_supported():
    from randlora_damage.core import RandLoRALinear

    model = MockSAM(dim=16)
    report = inject_randlora_damage_encoder(
        model,
        RandLoRADamageConfig(
            block_indices=(4,),
            qkv_targets=("q", "v"),
            extra_targets=("attn.proj", "mlp.lin1", "mlp.lin2"),
            rank=4,
            target_trainable_params=None,
            auto_match_budget=False,
        ),
    )
    block = model.image_encoder.blocks[4]
    assert isinstance(block.attn.proj, RandLoRALinear)
    assert isinstance(block.mlp.lin1, RandLoRALinear)
    assert isinstance(block.mlp.lin2, RandLoRALinear)
    assert len(report.wrapped_modules) == 4
