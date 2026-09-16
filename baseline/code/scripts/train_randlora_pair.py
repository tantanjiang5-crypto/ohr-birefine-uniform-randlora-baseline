#!/usr/bin/env python3 -u
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from common_runtime.coco_dataset import CocoGTBoxDataset, DeterministicEpochSampler
from common_runtime.evaluation import evaluate_model
from common_runtime.io_utils import (
    atomic_json_dump,
    atomic_torch_save,
    load_json,
    seed_everything,
    sha256_file,
    sha256_jsonable,
    state_dict_max_abs_diff,
)
from common_runtime.losses import build_mqrs_regularizer, compute_joint_losses
from common_runtime.project import build_model, import_tqmotp_loss, optimizer_coverage_audit
from common_runtime.sam_geometry import CorrectSamPreprocessor

RUN_ROOT = PACKAGE_ROOT.parent
RANDLORA_ROOT = RUN_ROOT / "vendor" / "DAG-RandLoRA-v1.0.0"
if str(RANDLORA_ROOT) not in sys.path:
    sys.path.insert(0, str(RANDLORA_ROOT))
from randlora_damage import (
    append_randlora_param_group,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


def collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def worker_seed(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


def scalar_parts(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().float().cpu().item()) for name, value in parts.items()}


def group_gradient_norms(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        squared = 0.0
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum().item())
        result[str(group.get("name", f"group_{index}"))] = math.sqrt(squared)
    return result


def finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.requires_grad
    )


def randlora_gradient_state(model: torch.nn.Module) -> dict[str, float | bool]:
    lambdas, gammas = [], []
    for name, parameter in model.named_parameters():
        if "randlora_lambda" in name:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"invalid RandLoRA lambda gradient: {name}")
            lambdas.append(float(parameter.grad.detach().float().abs().sum().item()))
        elif "randlora_gamma" in name:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"invalid RandLoRA gamma gradient: {name}")
            gammas.append(float(parameter.grad.detach().float().abs().sum().item()))
    if not lambdas or not gammas:
        raise RuntimeError("RandLoRA gradient audit could not find lambda/gamma tensors")
    return {
        "lambda_grad_l1": float(sum(lambdas)),
        "gamma_grad_l1": float(sum(gammas)),
        "lambda_nonzero": bool(sum(lambdas) > 0.0),
        "gamma_nonzero": bool(sum(gammas) > 0.0),
    }


def load_common_initialization(model: torch.nn.Module, common_init: dict, *, adapter_type: str) -> dict[str, object]:
    """Load the shared V0 initialization without ever importing legacy LoRA into V1.

    V1 constructs Meta SAM from its raw checkpoint first, then RandLoRA.  The
    shared checkpoint is applied strictly for all compatible non-encoder state
    (mask decoder, TQ-MOTP, QJ-2 and frozen prompt encoder); encoder keys are
    intentionally excluded so legacy LoRA cannot be reintroduced.
    """
    state = common_init["model_state"]
    if adapter_type == "lora":
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"V0 common-init strict load failed: {result}")
        return {"mode": "strict_full", "loaded_keys": len(state), "skipped_encoder_keys": 0}
    def is_encoder_key(key: str) -> bool:
        # The historical common-init serializes both the direct alias and the
        # SAM-owned alias.  Neither may enter a RandLoRA model.
        return key.startswith("sam_base.sam.image_encoder.") or key.startswith("sam_base.image_encoder.")

    filtered = {key: value for key, value in state.items() if not is_encoder_key(key)}
    result = model.load_state_dict(filtered, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"V1 common-init has unexpected non-encoder keys: {result.unexpected_keys}")
    invalid_missing = [key for key in result.missing_keys if not is_encoder_key(key)]
    if invalid_missing:
        raise RuntimeError(f"V1 common-init missed non-encoder state: {invalid_missing[:8]}")
    return {
        "mode": "strict_non_encoder",
        "loaded_keys": len(filtered),
        "skipped_encoder_keys": len(state) - len(filtered),
        "missing_encoder_keys": len(result.missing_keys),
    }


def measured_baseline_lora_params(config: dict, project_root: Path, model_file: Path) -> int:
    """Instantiate the current copied V0 model and count its actual encoder LoRA."""
    baseline_cfg = copy.deepcopy(config["model"])
    baseline_cfg["adapter_type"] = "lora"
    baseline_cfg.pop("randlora_config", None)
    baseline = build_model(model_file, project_root, baseline_cfg)
    count = sum(
        parameter.numel()
        for name, parameter in baseline.named_parameters()
        if parameter.requires_grad and name.startswith("sam_base.sam.image_encoder") and "lora" in name.lower()
    )
    if count <= 0:
        raise RuntimeError("measured baseline Encoder LoRA parameter count is zero")
    del baseline
    return int(count)


def capture_k_slices(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    for index in range(4, 12):
        qkv = model.sam.image_encoder.blocks[index].attn.qkv
        if not hasattr(qkv, "base_layer"):
            continue
        dim = qkv.base_layer.in_features
        captured[index] = qkv.base_layer.weight.detach()[dim:2 * dim].cpu().clone()
    if not captured:
        raise RuntimeError("could not capture RandLoRA fused-QKV K slices")
    return captured


def assert_k_slices_unchanged(model: torch.nn.Module, initial: dict[int, torch.Tensor]) -> None:
    for index, expected in initial.items():
        qkv = model.sam.image_encoder.blocks[index].attn.qkv
        dim = qkv.base_layer.in_features
        observed = qkv.base_layer.weight.detach()[dim:2 * dim].cpu()
        if not torch.equal(observed, expected):
            raise RuntimeError(f"K slice changed in RandLoRA block {index}")


def save_randlora_adapter_if_needed(
    model: torch.nn.Module, path: Path, *, adapter_type: str, metadata: dict[str, object]
) -> None:
    if adapter_type == "randlora":
        save_adapter_checkpoint(model.sam, path, metadata=metadata)


def verify_roundtrip(
    *, model: torch.nn.Module, model_file: Path, project_root: Path, config: dict,
    common_init: dict, checkpoint_path: Path, adapter_path: Path | None, adapter_type: str,
) -> dict[str, object]:
    """Validate strict full and (for V1) strict adapter checkpoint recovery on CPU."""
    restored = build_model(model_file, project_root, config["model"])
    load_common_initialization(restored, common_init, adapter_type=adapter_type)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model")
    if state is None:
        raise RuntimeError("training checkpoint lacks model state")
    result = restored.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict training checkpoint restore failed: {result}")
    adapter_result: object | None = None
    if adapter_type == "randlora":
        if adapter_path is None or not adapter_path.is_file():
            raise RuntimeError("RandLoRA adapter checkpoint is missing")
        adapter_result = load_adapter_checkpoint(restored.sam, adapter_path, strict=True)
    del restored
    return {
        "status": "PASS", "strict_training_checkpoint": str(checkpoint_path),
        "strict_adapter_checkpoint": str(adapter_path) if adapter_path else None,
        "adapter_result": str(adapter_result) if adapter_result is not None else None,
    }


class ResourceSampler:
    """Lightweight per-run CPU/GPU sampler; never changes device visibility."""

    def __init__(self, physical_gpu: str, interval_seconds: float = 5.0) -> None:
        self.physical_gpu = physical_gpu
        self.interval_seconds = interval_seconds
        self.rows: list[dict[str, float | str | None]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None

    def _sample(self) -> None:
        row: dict[str, float | str | None] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.psutil is not None:
            row["cpu_percent"] = float(self.psutil.cpu_percent(interval=None))
            row["process_rss_mib"] = float(self.psutil.Process().memory_info().rss / (1 << 20))
        else:
            row["cpu_percent"] = None
            row["process_rss_mib"] = None
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--id", self.physical_gpu,
                 "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                check=True, text=True, capture_output=True, timeout=5,
            )
            values = [value.strip() for value in completed.stdout.strip().split(",")]
            row["gpu_util_percent"] = float(values[0])
            row["gpu_memory_used_mib"] = float(values[1])
            row["gpu_memory_total_mib"] = float(values[2])
        except Exception:
            row["gpu_util_percent"] = None
            row["gpu_memory_used_mib"] = None
            row["gpu_memory_total_mib"] = None
        self.rows.append(row)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample()
        self.thread.start()

    def stop(self, output_path: Path) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 2.0)
        self._sample()
        previous_rows: list[dict[str, str]] = []
        if output_path.is_file():
            with output_path.open("r", newline="", encoding="utf-8") as handle:
                previous_rows = list(csv.DictReader(handle))
        all_rows = [*previous_rows, *self.rows]
        fieldnames = sorted({key for row in all_rows for key in row})
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    mode: str,
    mqrs_enabled: bool,
    config: dict[str, Any],
    common_init_sha256: str,
) -> None:
    payload = {
        "format": "tqmotp_qj2_mqrs_training_v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_D_legacy_AP": best_metric,
        "best_epoch": best_epoch,
        "mode": mode,
        "mqrs_enabled": mqrs_enabled,
        "common_init_sha256": common_init_sha256,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    atomic_torch_save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU V0 LoRA / V1 RandLoRA paired trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["V0", "V1"], required=True)
    parser.add_argument("--device", required=True, help="e.g. cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="load --resume strictly and run one complete validation without training or state updates",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if bool(config.get("mqrs", {}).get("enabled", False)):
        raise RuntimeError("MQ-RS must remain disabled in this RandLoRA encoder-only comparison")
    expected_loss = {"tqmotp_weight": 1.0, "mask_bce_weight": 5.0, "mask_dice_weight": 5.0, "qj2_weight": 0.1}
    for key, expected in expected_loss.items():
        actual = float(config.get("loss", {}).get(key, float("nan")))
        if actual != expected:
            raise RuntimeError(f"loss protocol drift: {key}={actual}, expected {expected}")
    mqrs_enabled = False
    adapter_type = str(config["model"].get("adapter_type", "lora"))
    expected_mode = "V1" if adapter_type == "randlora" else "V0"
    if args.mode != expected_mode:
        raise RuntimeError(f"--mode={args.mode} conflicts with adapter_type={adapter_type!r}; expected {expected_mode}")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists (use --resume): {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "code_snapshot").mkdir(exist_ok=True)
    (output_dir / "data_order").mkdir(exist_ok=True)
    for source in [config_path, Path(__file__).resolve(), PACKAGE_ROOT / "common_runtime" / "sam_geometry.py"]:
        target = output_dir / "code_snapshot" / source.name
        target.write_bytes(source.read_bytes())

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    if str(device) != "cuda:0" or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "this runner requires exactly one CUDA-visible GPU and --device cuda:0; "
            "launch it with CUDA_VISIBLE_DEVICES=<one idle physical GPU>"
        )
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    paths = config["paths"]
    project_root = Path(paths["project_root"]).resolve()
    model_file = Path(paths["model_file"]).resolve()
    common_init_path = Path(paths["common_init"]).resolve()
    common_init_sha256 = sha256_file(common_init_path)

    train_dataset = CocoGTBoxDataset(
        paths["train_json"], paths["train_images"],
        category_ids=config["data"].get("category_ids"), filter_empty=True,
    )
    val_dataset = CocoGTBoxDataset(
        paths["val_json"], paths["val_images"],
        category_ids=train_dataset.category_ids, filter_empty=False,
    )
    if train_dataset.class_names != val_dataset.class_names:
        raise RuntimeError("train/val category mapping differs")

    baseline_lora_params = measured_baseline_lora_params(config, project_root, model_file)
    if adapter_type == "randlora":
        rand_cfg = config["model"].get("randlora_config")
        if not isinstance(rand_cfg, dict):
            raise RuntimeError("V1 requires model.randlora_config")
        if rand_cfg.get("target_trainable_params") is not None:
            raise RuntimeError("target_trainable_params must be resolved from a measured baseline, not pre-filled")
        rand_cfg["target_trainable_params"] = baseline_lora_params
        if rand_cfg.get("rank") is not None or not bool(rand_cfg.get("auto_match_budget", False)):
            raise RuntimeError("V1 must use rank=None and auto_match_budget=True")
        if rand_cfg.get("qkv_targets") != ["q", "v"] or rand_cfg.get("extra_targets") != []:
            raise RuntimeError("V1 may target only fused q/v, never K/proj/MLP")
        if rand_cfg.get("block_indices") != list(range(4, 12)):
            raise RuntimeError("V1 must target image encoder blocks 4--11 only")
        if rand_cfg.get("forward_mode") != "auto" or rand_cfg.get("adapter_dtype") != "float32":
            raise RuntimeError("V1 requires forward_mode=auto and adapter_dtype=float32")
        if not bool(rand_cfg.get("freeze_image_encoder", False)):
            raise RuntimeError("V1 requires freeze_image_encoder=True")
    model = build_model(model_file, project_root, config["model"]).to(device)
    common_init = torch.load(common_init_path, map_location="cpu")
    init_load_audit = load_common_initialization(model, common_init, adapter_type=adapter_type)
    initial_max_diff = 0.0 if adapter_type == "lora" else None
    initial_max_name = None
    if adapter_type == "lora":
        initial_max_diff, initial_max_name = state_dict_max_abs_diff(common_init["model_state"], model.state_dict())
        if initial_max_diff != 0.0:
            raise RuntimeError(f"loaded V0 differs from common init: {initial_max_diff} at {initial_max_name}")
    atomic_json_dump({
        "status": "PASS",
        "common_init_sha256": common_init_sha256,
        "max_abs_diff": initial_max_diff,
        "max_abs_diff_parameter": initial_max_name,
        "load_strategy": init_load_audit,
        "source_common_init_model_sha256": common_init.get("source", {}).get("model_file_sha256"),
    }, output_dir / "INITIAL_STATE_AUDIT.json")

    if adapter_type == "randlora":
        report = model.sam_base.randlora_report
        if report is None:
            raise RuntimeError("RandLoRA injection report is missing")
        atomic_json_dump({
            "status": "PASS",
            "baseline_encoder_lora_trainable_params": baseline_lora_params,
            "target_trainable_params": report.requested_budget,
            "resolved_rank": report.default_rank,
            "actual_randlora_trainable_params": report.actual_adapter_params,
            "budget_relative_error": report.budget_relative_error,
            "budget_warning": report.budget_warning,
            "selected_blocks": list(report.selected_blocks),
            "qkv_targets": list(report.qkv_targets),
            "extra_targets": list(report.extra_targets),
            "adapter_dtype": config["model"]["randlora_config"]["adapter_dtype"],
            "forward_mode": config["model"]["randlora_config"]["forward_mode"],
            "freeze_image_encoder": config["model"]["randlora_config"]["freeze_image_encoder"],
            "module_manifest": list(report.module_manifest),
        }, output_dir / "PARAMETER_MANIFEST.json")
    atomic_json_dump(config, output_dir / "RESOLVED_CONFIG.json")

    preprocessor = CorrectSamPreprocessor(model.sam, device)
    optimizer_cfg = config["optimizer"]
    groups = model.get_param_groups(
        lora_lr=float(optimizer_cfg["lora_lr"]),
        mask_decoder_lr=float(optimizer_cfg["mask_decoder_lr"]),
        tq_head_lr=float(optimizer_cfg["tq_head_lr"]),
        prototype_lr=float(optimizer_cfg["prototype_lr"]),
        qj2_head_lr=float(optimizer_cfg["qj2_head_lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    if adapter_type == "randlora":
        append_randlora_param_group(
            groups, model, encoder_lr=float(optimizer_cfg["lora_lr"]),
            encoder_weight_decay=float(optimizer_cfg["weight_decay"]),
            group_name="randlora_encoder",
        )
        # The package uses ``group_name`` for generic optimizers; the existing
        # trainer logs the historical ``name`` field, so mirror it for audit
        # readability without changing any pre-existing group.
        groups[-1]["name"] = "randlora_encoder"
        model.assert_optimizer_parameter_coverage(groups)
    optimizer_audit = optimizer_coverage_audit(model, groups)
    if not optimizer_audit["pass"]:
        raise RuntimeError(f"optimizer audit failed: {optimizer_audit}")
    optimizer = torch.optim.AdamW(groups, betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])))

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    accumulation = int(training_cfg["gradient_accumulation"])
    epochs = int(training_cfg["epochs"])
    steps_per_epoch = math.ceil(math.ceil(len(train_dataset) / batch_size) / accumulation)
    total_steps = max(1, steps_per_epoch * epochs)
    # The validated F0 protocol filters empty/invalid-instance batches before
    # forming an optimizer update.  Consequently its deterministic three-seed
    # runs contain 23,827 successful updates rather than the nominal
    # ceil(loader/accumulation)*20 = 23,840.  F1 must reproduce that protocol;
    # the value is an audit expectation only and does not alter the scheduler.
    expected_successful_steps = int(
        training_cfg.get("expected_successful_optimizer_steps", total_steps)
    )
    scheduler, warmup_steps = build_scheduler(optimizer, total_steps, float(training_cfg.get("warmup_ratio", 0.05)))
    # The verified TQ-MOTP formal path keeps its coupled OT computation in
    # FP32.  Preserve its GradScaler/checkpoint behavior independently of
    # whether CUDA autocast is enabled for the surrounding SAM forward.
    amp_enabled = bool(training_cfg.get("amp", False))
    grad_scaler_enabled = bool(training_cfg.get("grad_scaler", True))
    scaler = torch.cuda.amp.GradScaler(
        enabled=grad_scaler_enabled,
        init_scale=float(training_cfg.get("amp_init_scale", 128.0)),
        growth_interval=int(training_cfg.get("amp_growth_interval", 1000)),
    )

    class_counts = config["data"].get("class_counts") or train_dataset.class_counts()
    if len(class_counts) != len(train_dataset.class_names):
        raise ValueError("class_counts length differs from number of classes")
    TQMOTPLoss = import_tqmotp_loss(project_root)
    tqmotp_loss_fn = TQMOTPLoss(
        num_classes=len(class_counts),
        class_counts=torch.tensor(class_counts, dtype=torch.float32, device=device),
    )
    mqrs_regularizer = build_mqrs_regularizer(config["mqrs"]).to(device)

    sampler = DeterministicEpochSampler(train_dataset, seed=seed)
    workers = int(training_cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": collate,
        "worker_init_fn": worker_seed,
        "persistent_workers": bool(training_cfg.get("persistent_workers", False)) and workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.get("val_batch_size", batch_size)),
        shuffle=False,
        **loader_kwargs,
    )

    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_epoch = 0
    resume_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint["mode"] != args.mode or checkpoint["common_init_sha256"] != common_init_sha256:
            raise RuntimeError("resume checkpoint does not match mode/common initialization")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        resume_epoch = int(checkpoint["epoch"])
        start_epoch = resume_epoch + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_D_legacy_AP"])
        best_epoch = int(checkpoint["best_epoch"])
        rng = checkpoint.get("rng")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    run_manifest = {
        "mode": args.mode,
        "mqrs_enabled": mqrs_enabled,
        "adapter_type": adapter_type,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "seed": seed,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "common_init": str(common_init_path),
        "common_init_sha256": common_init_sha256,
        "model_file": str(model_file),
        "model_file_sha256": sha256_file(model_file),
        "class_names": train_dataset.class_names,
        "class_counts": class_counts,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "expected_successful_optimizer_steps": expected_successful_steps,
        "warmup_steps": warmup_steps,
        "autocast_enabled": amp_enabled,
        "grad_scaler_enabled": grad_scaler_enabled,
        "optimizer_audit": optimizer_audit,
        "resume_checkpoint": str(args.resume) if args.resume else None,
    }
    atomic_json_dump(run_manifest, output_dir / "RUN_MANIFEST.json")
    log_path = output_dir / "train.jsonl"
    metric_rows: list[dict[str, Any]] = []
    existing_metrics_path = output_dir / "metrics.csv"
    if args.resume and existing_metrics_path.is_file():
        with existing_metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
            metric_rows = list(csv.DictReader(handle))
    first_step_audit_written = (output_dir / "FIRST_OPTIMIZER_STEP_AUDIT.json").is_file()
    randlora_first_gradient: dict[str, float | bool] | None = None
    randlora_gamma_seen = False
    initial_k_slices = capture_k_slices(model) if adapter_type == "randlora" else {}
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical_gpu or "," in physical_gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name one physical GPU for resource sampling")
    resource_sampler = ResourceSampler(physical_gpu)
    resource_sampler.start()
    torch.cuda.reset_peak_memory_stats(device)
    successful_optimizer_steps_this_run = 0
    consecutive_nonfinite = 0
    epoch_nonfinite = 0
    recovery_event_path = output_dir / "NUMERICAL_RECOVERY_EVENTS.jsonl"

    if args.evaluate_only:
        if not args.resume:
            raise RuntimeError("--evaluate-only requires --resume so the evaluation state is unambiguous")
        model.eval()
        validation_dir = output_dir / "validation" / f"epoch_{resume_epoch:03d}"
        validation_metrics = evaluate_model(
            model=model,
            preprocessor=preprocessor,
            loader=val_loader,
            output_dir=validation_dir,
            split_name="val",
            score_cfg=config["evaluation"]["score_fusion"],
            calibration_bins=int(config["evaluation"].get("calibration_bins", 15)),
            max_images=args.max_val_images,
        )
        current_metric = float(validation_metrics["coco_segm"]["D_legacy"]["AP"])
        previous_best = float(checkpoint.get("best_D_legacy_AP", float("-inf")))
        if math.isfinite(current_metric) and current_metric > previous_best:
            # Evaluation-only is used after a session interruption.  Promote
            # this checkpoint exactly as the normal epoch loop would have;
            # otherwise a better interrupted epoch could never be selected.
            promoted = copy.deepcopy(checkpoint)
            promoted["best_D_legacy_AP"] = current_metric
            promoted["best_epoch"] = resume_epoch
            atomic_torch_save(promoted, output_dir / "best.pth")
            atomic_torch_save(promoted, output_dir / "last.pth")
            if adapter_type == "randlora" and (output_dir / "randlora_last_adapter.pth").is_file():
                shutil.copy2(output_dir / "randlora_last_adapter.pth", output_dir / "randlora_best_adapter.pth")
        resource_sampler.stop(output_dir / "RESOURCE_SAMPLES.csv")
        print(json.dumps({
            "status": "PASS", "evaluate_only": True, "epoch": resume_epoch,
            "D_legacy_AP": current_metric,
            "D_legacy_AP75": validation_metrics["coco_segm"]["D_legacy"]["AP75"],
        }, ensure_ascii=False), flush=True)
        return

    def recover_nonfinite(*, epoch: int, batch_index: int, phase: str, scale_recorded_by_unscale: bool) -> None:
        nonlocal consecutive_nonfinite, epoch_nonfinite
        scale_before = float(scaler.get_scale())
        if scale_recorded_by_unscale:
            # GradScaler recorded found_inf during unscale_; step is therefore
            # skipped and update halves the dynamic scale in the standard way.
            scaler.step(optimizer)
            scaler.update()
        else:
            # A non-finite value produced by the norm calculation itself is
            # after unscale_.  Do not call optimizer.step(); lower the scaler
            # explicitly and record this exceptional phase.
            scaler.update(new_scale=scale_before / 2.0)
        scale_after = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        consecutive_nonfinite += 1
        epoch_nonfinite += 1
        event = {
            "event": "nonfinite_gradient_skip", "phase": phase, "epoch": epoch,
            "batch_index": batch_index, "global_step": global_step,
            "successful_optimizer_steps_this_run": successful_optimizer_steps_this_run,
            "grad_scaler_before": scale_before, "grad_scaler_after": scale_after,
            "consecutive_nonfinite": consecutive_nonfinite,
            "epoch_nonfinite": epoch_nonfinite,
        }
        with recovery_event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if consecutive_nonfinite >= 3 or epoch_nonfinite >= 5:
            raise RuntimeError(
                f"non-finite recovery limit reached at epoch={epoch}, batch={batch_index}; "
                f"consecutive={consecutive_nonfinite}, epoch_total={epoch_nonfinite}"
            )

    for epoch in range(start_epoch, epochs + 1):
        epoch_nonfinite = 0
        sampler.set_epoch(epoch)
        order_indices = sampler.order()
        order_image_ids = [int(train_dataset.image_ids[index]) for index in order_indices]
        order_record = {
            "epoch": epoch,
            "image_count": len(order_image_ids),
            "image_id_order_sha256": sha256_jsonable(order_image_ids),
            "first_100_image_ids": order_image_ids[:100],
        }
        atomic_json_dump(order_record, output_dir / "data_order" / f"epoch_{epoch:03d}.json")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **loader_kwargs,
        )
        model.train()
        if hasattr(model, "set_lora_dropout"):
            model.set_lora_dropout(True)
        optimizer.zero_grad(set_to_none=True)
        epoch_sums: dict[str, float] = {}
        micro_count = 0
        epoch_started = time.time()
        previous_optimizer_step_time = time.perf_counter()

        for batch_index, batch in enumerate(train_loader, start=1):
            prepared = preprocessor.process(batch)
            labels_count = sum(labels.numel() for labels in prepared.labels_list)
            if labels_count == 0:
                continue
            group_start = ((batch_index - 1) // accumulation) * accumulation + 1
            group_end = min(group_start + accumulation - 1, len(train_loader))
            accumulation_divisor = group_end - group_start + 1
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(prepared.images, prepared.boxes_list)
                bundle = compute_joint_losses(
                    output=output,
                    labels_list=prepared.labels_list,
                    gt_masks_list=prepared.masks_256_list,
                    tqmotp_loss_fn=tqmotp_loss_fn,
                    loss_cfg=config["loss"],
                    mqrs_enabled=mqrs_enabled,
                    mqrs_regularizer=mqrs_regularizer,
                    global_step=global_step,
                )
                scaled_loss = bundle.total / accumulation_divisor
            if not torch.isfinite(bundle.total):
                raise RuntimeError(f"non-finite loss at epoch={epoch}, batch={batch_index}")
            scaler.scale(scaled_loss).backward()
            micro_count += 1
            for name, value in scalar_parts(bundle.parts).items():
                epoch_sums[name] = epoch_sums.get(name, 0.0) + value

            should_step = micro_count % accumulation == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                if not finite_gradients(model):
                    recover_nonfinite(epoch=epoch, batch_index=batch_index, phase="post_unscale", scale_recorded_by_unscale=True)
                    continue
                gradient_norms = group_gradient_norms(optimizer)
                randlora_grads = randlora_gradient_state(model) if adapter_type == "randlora" else None
                if randlora_grads is not None:
                    if global_step == 0 and not bool(randlora_grads["lambda_nonzero"]):
                        raise RuntimeError("first RandLoRA optimizer step has zero lambda gradient")
                    if global_step == 0:
                        randlora_first_gradient = randlora_grads
                    elif bool(randlora_grads["gamma_nonzero"]):
                        randlora_gamma_seen = True
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(training_cfg.get("gradient_clip_norm", 1.0)),
                )
                if not finite_gradients(model):
                    recover_nonfinite(epoch=epoch, batch_index=batch_index, phase="post_clip", scale_recorded_by_unscale=False)
                    continue
                scaler.step(optimizer)
                scaler.update()
                consecutive_nonfinite = 0
                if adapter_type == "randlora":
                    assert_k_slices_unchanged(model, initial_k_slices)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                successful_optimizer_steps_this_run += 1
                optimizer_step_seconds = time.perf_counter() - previous_optimizer_step_time
                previous_optimizer_step_time = time.perf_counter()

                step_record = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "timestamp_unix": time.time(),
                    "optimizer_step_seconds": optimizer_step_seconds,
                    "mode": args.mode,
                    "mqrs_enabled": mqrs_enabled,
                    "instances": int(labels_count),
                    **scalar_parts(bundle.parts),
                    "gradient_norms_pre_clip": gradient_norms,
                    "learning_rates": {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)},
                    "grad_scaler": float(scaler.get_scale()),
                    "gpu_allocated_mib": int(torch.cuda.memory_allocated(device) / (1 << 20)),
                    "gpu_peak_mib": int(torch.cuda.max_memory_allocated(device) / (1 << 20)),
                }
                if randlora_grads is not None:
                    step_record["randlora_gradients"] = randlora_grads
                if bundle.mqrs_output is not None:
                    step_record.update(bundle.mqrs_output.logging_dict(prefix="mqrs"))
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step_record, ensure_ascii=False, allow_nan=True) + "\n")
                if not first_step_audit_written:
                    def marker_norm(*markers: str) -> float:
                        return sum(
                            value for group_name, value in gradient_norms.items()
                            if any(marker in group_name.lower() for marker in markers)
                        )
                    required_gradient_families = {
                        "lora": marker_norm("lora"),
                        "mask_decoder": marker_norm("mask_decoder", "decoder"),
                        "tqmotp": marker_norm("tq", "prototype"),
                        "qj2": marker_norm("qj2"),
                    }
                    zero_required = [name for name, value in required_gradient_families.items() if value <= 0]
                    step_record["required_gradient_families"] = required_gradient_families
                    if zero_required:
                        raise RuntimeError(f"zero gradient in required families: {zero_required}; groups={gradient_norms}")
                    if not mqrs_enabled and abs(step_record["mqrs/loss"]) > 0:
                        raise RuntimeError("MQ-RS loss must be exactly zero")
                    if adapter_type == "randlora":
                        step_record["randlora_first_step"] = {
                            "lambda_must_be_nonzero": bool(randlora_grads and randlora_grads["lambda_nonzero"]),
                            "gamma_may_be_zero": bool(randlora_grads and not randlora_grads["gamma_nonzero"]),
                            "k_slices_bitwise_unchanged": True,
                        }
                    atomic_json_dump(step_record, output_dir / "FIRST_OPTIMIZER_STEP_AUDIT.json")
                    first_step_audit_written = True

                if args.max_optimizer_steps and successful_optimizer_steps_this_run >= args.max_optimizer_steps:
                    break

        # Preserve the completed epoch before running the evaluator.
        save_checkpoint(
            output_dir / "last.pth", model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, global_step=global_step, best_metric=best_metric,
            best_epoch=best_epoch, mode=args.mode, mqrs_enabled=mqrs_enabled, config=config,
            common_init_sha256=common_init_sha256,
        )

        validation_metrics: dict[str, Any] = {}
        if epoch % int(training_cfg.get("validate_every", 1)) == 0:
            validation_metrics = evaluate_model(
                model=model,
                preprocessor=preprocessor,
                loader=val_loader,
                output_dir=output_dir / "validation" / f"epoch_{epoch:03d}",
                split_name="val",
                score_cfg=config["evaluation"]["score_fusion"],
                calibration_bins=int(config["evaluation"].get("calibration_bins", 15)),
                max_images=args.max_val_images,
            )
        current_metric = (
            float(validation_metrics["coco_segm"]["D_legacy"]["AP"])
            if validation_metrics else float("-inf")
        )
        if math.isfinite(current_metric) and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pth", model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch, global_step=global_step, best_metric=best_metric,
                best_epoch=best_epoch, mode=args.mode, mqrs_enabled=mqrs_enabled, config=config,
                common_init_sha256=common_init_sha256,
            )
            save_randlora_adapter_if_needed(
                model, output_dir / "randlora_best_adapter.pth", adapter_type=adapter_type,
                metadata={
                    "epoch": epoch, "best_D_AP": best_metric,
                    "D_AP75": float(validation_metrics["coco_segm"]["D_legacy"]["AP75"]),
                },
            )
        save_checkpoint(
            output_dir / "last.pth", model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, global_step=global_step, best_metric=best_metric,
            best_epoch=best_epoch, mode=args.mode, mqrs_enabled=mqrs_enabled, config=config,
            common_init_sha256=common_init_sha256,
        )
        save_randlora_adapter_if_needed(
            model, output_dir / "randlora_last_adapter.pth", adapter_type=adapter_type,
            metadata={
                "epoch": epoch, "global_step": global_step,
                "D_AP": float(validation_metrics["coco_segm"]["D_legacy"]["AP"]) if validation_metrics else None,
                "D_AP75": float(validation_metrics["coco_segm"]["D_legacy"]["AP75"]) if validation_metrics else None,
            },
        )

        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.time() - epoch_started,
            "peak_gpu_mib": int(torch.cuda.max_memory_allocated(device) / (1 << 20)),
            **{f"train/{name}": value / max(micro_count, 1) for name, value in epoch_sums.items()},
        }
        if validation_metrics:
            epoch_row.update(
                {
                    "val/D_legacy_AP": validation_metrics["coco_segm"]["D_legacy"]["AP"],
                    "val/D_legacy_AP50": validation_metrics["coco_segm"]["D_legacy"]["AP50"],
                    "val/D_legacy_AP75": validation_metrics["coco_segm"]["D_legacy"]["AP75"],
                    "val/B_qj2_AP": validation_metrics["coco_segm"]["B_qj2"]["AP"],
                    "val/classification_macro_f1": validation_metrics["classification"]["macro_f1"],
                    "val/mean_iou": validation_metrics["mask_pixel"]["mean_iou"],
                    "val/mean_dice": validation_metrics["mask_pixel"]["mean_dice"],
                }
            )
        metric_rows.append(epoch_row)
        fieldnames = sorted({key for row in metric_rows for key in row})
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)
        print(json.dumps(epoch_row, ensure_ascii=False, allow_nan=True), flush=True)

        # A segmented invocation represents exactly one data epoch.  Empty or
        # invalid-instance batches can make the successful-step counter fall
        # one or two short of the cap; allowing the outer epoch loop to run in
        # that case silently creates a one-step next epoch and corrupts resume
        # bookkeeping.  Always return to the supervisor after this epoch.
        if args.max_optimizer_steps:
            break

    completed_epoch = int(metric_rows[-1]["epoch"]) if metric_rows else 0
    segmented_run = bool(args.max_optimizer_steps and completed_epoch < epochs)
    # A formal run must account for every planned optimizer attempt.  AMP
    # overflow recovery deliberately skips the optimizer/scheduler update, so
    # successful steps alone are dataset/run dependent and cannot be known in
    # advance.  Count persisted recovery events as attempted steps; this keeps
    # the original 23,827-success/13-skip runs valid and also supports new
    # datasets without baking their eventual overflow count into the config.
    numerical_skip_count = 0
    if recovery_event_path.is_file():
        with recovery_event_path.open("r", encoding="utf-8") as handle:
            numerical_skip_count = sum(1 for line in handle if line.strip())
    accounted_optimizer_attempts = global_step + numerical_skip_count
    step_count_ok = completed_epoch < epochs or accounted_optimizer_attempts == total_steps
    completion_status = (
        "PASS" if completed_epoch >= epochs and step_count_ok
        else "NOT_PASS" if completed_epoch >= epochs and not step_count_ok
        else "IN_PROGRESS"
    )
    completion = {
        "status": completion_status,
        "mode": args.mode,
        "mqrs_enabled": mqrs_enabled,
        "adapter_type": adapter_type,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "expected_total_optimizer_steps": total_steps,
        "expected_successful_optimizer_steps": expected_successful_steps,
        "numerical_skip_count": numerical_skip_count,
        "accounted_optimizer_attempts": accounted_optimizer_attempts,
        "optimizer_step_count_ok": step_count_ok,
        "best_D_legacy_AP": best_metric,
        "best_epoch": best_epoch,
        "common_init_sha256": common_init_sha256,
        "best_checkpoint_exists": (output_dir / "best.pth").exists(),
        "last_checkpoint_exists": (output_dir / "last.pth").exists(),
        "segmented_run": segmented_run,
        "randlora_k_slices_bitwise_unchanged": True if adapter_type == "randlora" else None,
        "randlora_first_lambda_gradient": randlora_first_gradient,
        "randlora_gamma_nonzero_after_first_step": randlora_gamma_seen if adapter_type == "randlora" else None,
    }
    atomic_json_dump(completion, output_dir / "COMPLETION_AUDIT.json")
    roundtrip = verify_roundtrip(
        model=model, model_file=model_file, project_root=project_root, config=config,
        common_init=common_init, checkpoint_path=output_dir / "last.pth",
        adapter_path=(output_dir / "randlora_last_adapter.pth") if adapter_type == "randlora" else None,
        adapter_type=adapter_type,
    )
    atomic_json_dump(roundtrip, output_dir / "CHECKPOINT_ROUNDTRIP_AUDIT.json")
    resource_sampler.stop(output_dir / "RESOURCE_SAMPLES.csv")


if __name__ == "__main__":
    main()
