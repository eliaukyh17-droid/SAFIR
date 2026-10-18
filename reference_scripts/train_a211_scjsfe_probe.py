"""Train the single pre-registered A2.11 SC-JSFE five-epoch probe.

This internal-only runner never opens the official MSRS-361, FMB, or M3FD.
The canonical TIFDFusion path is immutable.  Epoch stages, seed, thresholds,
loss weights, and final-checkpoint selection are fixed before execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MSRSDataset, collate_fn
from models.a211_safe_joint_source_expert import (
    RouteThresholds,
    SafeConditionalJointSourceFusion,
)
from utils.a211_protocol import (
    LOCKED_EPOCH_STAGES,
    stage_for_epoch,
    validate_acceptance,
    validate_forbidden,
    validate_route_thresholds,
    validate_schedule,
)


SEVERITIES = ("mild", "medium", "heavy")
EXPECTED_TRAIN_SAMPLES = 975
EXPECTED_VALIDATION_SAMPLES = 108


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tifdfusion_a211_scjsfe_probe_5ep.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def manifest_stems(path: str | Path) -> tuple[str, ...]:
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                values.append(Path(value).stem.casefold())
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate entries in manifest: {path}")
    return tuple(values)


def validate_manifests(cfg: dict) -> None:
    msrs = cfg["data"]["msrs"]
    train = manifest_stems(msrs["train_manifest"])
    validation = manifest_stems(msrs["val_manifest"])
    if len(train) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(f"A2.11 train count changed: {len(train)}")
    if len(validation) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(f"A2.11 validation count changed: {len(validation)}")
    overlap = set(train).intersection(validation)
    if overlap:
        raise ValueError(f"A2.11 train/validation overlap: {len(overlap)}")


def validate_config(cfg: dict) -> dict:
    model_cfg = cfg["a211_scjsfe"]
    validate_schedule(model_cfg["schedule"])
    validate_route_thresholds(model_cfg["route_thresholds"])
    validate_acceptance(model_cfg["acceptance"])
    validate_forbidden(model_cfg["forbidden_during_probe"])
    if bool(model_cfg["official_test_loaded"]):
        raise RuntimeError("official_test_loaded must remain false")
    if int(cfg["train"]["seed"]) != int(model_cfg["schedule"]["seed"]):
        raise ValueError("top-level and A2.11 training seeds differ")
    if bool(cfg["train"].get("validation_enabled", False)):
        raise ValueError("legacy trainer validation must remain disabled")
    smoke = cfg["data"]["msrs"]["smoke"]
    if not bool(smoke["enabled"]) or float(smoke["probability"]) != 1.0:
        raise ValueError("A2.11 training smoke probability must remain one")
    if tuple(smoke["severities"]) != SEVERITIES:
        raise ValueError("A2.11 training severities changed")
    validate_manifests(cfg)
    return model_cfg


def initialize_cuda(device_name: str) -> tuple[torch.device, int]:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A2.11 five-epoch probe requires CUDA")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise RuntimeError(f"CUDA device {index} outside visible range [0, {count})")
    torch.cuda.set_device(index)
    torch.empty(1, device=device)
    torch.cuda.synchronize(index)
    torch.cuda.reset_peak_memory_stats(index)
    return device, index


def make_train_loader(cfg: dict, model_cfg: dict, epoch: int) -> DataLoader:
    msrs = cfg["data"]["msrs"]
    dataset = MSRSDataset(
        root=msrs["root"],
        split="train",
        img_size=tuple(msrs["img_size"]),
        has_label=True,
        augment=True,
        subset_file=msrs["train_manifest"],
        smoke_config=msrs["smoke"],
    )
    if len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(f"A2.11 train dataset changed: {len(dataset)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(model_cfg["schedule"]["seed"]) + int(epoch))
    return DataLoader(
        dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(model_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def make_validation_loader(cfg: dict, model_cfg: dict, smoke: dict) -> DataLoader:
    msrs = cfg["data"]["msrs"]
    dataset = MSRSDataset(
        root=msrs["root"],
        split="train",
        img_size=tuple(msrs["img_size"]),
        has_label=True,
        augment=False,
        subset_file=msrs["val_manifest"],
        smoke_config=smoke,
    )
    if len(dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.11 validation dataset changed: {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(model_cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_validation_loaders(cfg: dict, model_cfg: dict) -> dict[str, DataLoader]:
    base = dict(cfg["data"]["msrs"]["validation_smoke"])
    if int(base["base_seed"]) != 2320260905 or not bool(base["deterministic"]):
        raise ValueError("A2.11 validation smoke seed/determinism changed")
    loaders = {}
    for index, severity in enumerate(SEVERITIES):
        smoke = dict(base)
        smoke.update(
            enabled=True,
            probability=1.0,
            severities=[severity],
            base_seed=int(base["base_seed"]) + 100000 * index,
        )
        loaders[severity] = make_validation_loader(cfg, model_cfg, smoke)
    clean = dict(base)
    clean.update(enabled=False, probability=0.0)
    loaders["clean"] = make_validation_loader(cfg, model_cfg, clean)
    return loaders


def build_composite(cfg: dict, model_cfg: dict, device: torch.device):
    # Keep protocol/unit-test imports lightweight; the full canonical stack is
    # required only for an actual CUDA training invocation.
    from models.tifd_fusion import build_model

    checkpoint_path = Path(model_cfg["baseline_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = file_sha256(checkpoint_path)
    if actual_sha != str(model_cfg["baseline_sha256"]):
        raise RuntimeError("canonical checkpoint SHA256 mismatch")
    base = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    result = base.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("canonical checkpoint does not load strictly")
    base.set_phase(3)
    model = SafeConditionalJointSourceFusion(
        base,
        channels=model_cfg["channels"],
        router_widths=model_cfg["router_widths"],
        num_heads=int(model_cfg["attention_heads"]),
        thresholds=RouteThresholds(**model_cfg["route_thresholds"]),
    ).to(device)
    return model, checkpoint, actual_sha


def luminance(visible: torch.Tensor) -> torch.Tensor:
    if visible.shape[1] == 1:
        return visible
    weights = visible.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (visible[:, :3] * weights).sum(dim=1, keepdim=True)


def low_pass(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(image, 5, stride=1, padding=2)


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    horizontal = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    vertical = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(horizontal.square() + vertical.square() + 1e-8)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-2:] != value.shape[-2:]:
        mask = F.interpolate(mask, size=value.shape[-2:], mode="area")
    weight = mask.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(
            target, size=prediction.shape[-2:], mode="bilinear", align_corners=False,
        )
    return masked_mean((prediction - target).abs(), mask)


def router_losses(
    smoke_router: dict,
    clean_router: dict,
    smoke_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    smoke_target = torch.ones(
        smoke_mask.shape[0], dtype=torch.long, device=smoke_mask.device,
    )
    clean_target = torch.zeros_like(smoke_target)
    global_loss = 0.5 * (
        F.cross_entropy(smoke_router["class_logits"], smoke_target)
        + F.cross_entropy(clean_router["class_logits"], clean_target)
    )
    local_values = []
    for smoke_logits, clean_logits in zip(
        smoke_router["local_logits"], clean_router["local_logits"],
    ):
        target = F.interpolate(
            smoke_mask, size=smoke_logits.shape[-2:], mode="area",
        )
        local_values.append(
            0.5 * (
                F.binary_cross_entropy_with_logits(smoke_logits, target)
                + F.binary_cross_entropy_with_logits(
                    clean_logits, torch.zeros_like(clean_logits),
                )
            )
        )
    clean_negative = -torch.log1p(
        -clean_router["p_smoke"].clamp(max=1.0 - 1e-6)
    ).mean()
    return {
        "router_global": global_loss,
        "router_local": torch.stack(local_values).mean(),
        "router_clean_negative": clean_negative,
    }


def set_stage_trainability(
    model: SafeConditionalJointSourceFusion,
    stage: str,
) -> None:
    if stage not in LOCKED_EPOCH_STAGES:
        raise ValueError(f"unsupported A2.11 stage: {stage}")
    for parameter in model.router.parameters():
        parameter.requires_grad_(True)
    expert_trainable = stage != "router_warmup"
    for parameter in model.smoke_expert.parameters():
        parameter.requires_grad_(expert_trainable)
    model.freeze_base()


def compute_training_loss(
    model: SafeConditionalJointSourceFusion,
    batch: dict,
    device: torch.device,
    model_cfg: dict,
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visible = batch["vis"].to(device, non_blocking=True)
    clean_visible = batch["vi_clean"].to(device, non_blocking=True)
    infrared = batch["ir"].to(device, non_blocking=True)
    mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)

    smoke_router = model.router(visible, infrared)
    clean_router = model.router(clean_visible, infrared)
    terms = router_losses(smoke_router, clean_router, mask)
    if stage == "router_warmup":
        active = (
            "router_global", "router_local", "router_clean_negative",
        )
    else:
        with torch.no_grad():
            clean_teacher = model.baseline(clean_visible, infrared)
        gate_override = mask if stage == "oracle_joint" else None
        student, auxiliary = model(
            visible,
            infrared,
            route_mode="smoke",
            gate_override=gate_override,
            return_aux=True,
        )
        smoke_canonical = auxiliary["canonical"]
        outside = 1.0 - mask
        visible_target = (
            (1.0 - mask) * luminance(visible)
            + mask * luminance(clean_visible)
        )
        source_gradient = torch.maximum(
            gradient_magnitude(luminance(clean_visible)),
            gradient_magnitude(infrared),
        )
        student_gradient = gradient_magnitude(student)
        terms.update({
            "teacher_smoke_region": masked_l1(student, clean_teacher, mask),
            "teacher_low_frequency": masked_l1(
                low_pass(student), low_pass(clean_teacher), mask,
            ),
            "canonical_clear_region": masked_l1(
                student, smoke_canonical, outside,
            ),
            "visible_reconstruction": masked_l1(
                auxiliary["visible_reconstruction"], visible_target,
                torch.ones_like(mask),
            ),
            "infrared_reconstruction": masked_l1(
                auxiliary["infrared_reconstruction"], infrared,
                torch.ones_like(mask),
            ),
            "source_gradient": masked_l1(
                student_gradient, source_gradient, mask,
            ),
            "unsupported_edge": masked_mean(
                F.relu(student_gradient - source_gradient), mask,
            ),
            "baseline_teacher_mae": (smoke_canonical - clean_teacher).abs().mean(),
            "student_teacher_mae": (student - clean_teacher).abs().mean(),
            "output_delta_abs": (student - smoke_canonical).abs().mean(),
        })
        active = tuple(model_cfg["loss_weights"])

    weights = model_cfg["loss_weights"]
    total = sum(float(weights[name]) * terms[name] for name in active)
    terms["total"] = total
    return total, terms


def add_metrics(target: dict[str, float], values: dict, weight: int) -> None:
    for name, value in values.items():
        number = float(value.detach()) if torch.is_tensor(value) else float(value)
        target[name] += number * weight


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def maximum_state_update(
    before: dict[str, torch.Tensor],
    module: torch.nn.Module,
) -> float:
    after = module.state_dict()
    if before.keys() != after.keys():
        raise RuntimeError("canonical state keys changed")
    maximum = 0.0
    for name, original in before.items():
        difference = (
            after[name].detach().cpu().float() - original.float()
        ).abs().max().item()
        maximum = max(maximum, difference)
    return maximum


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    model_cfg,
    epoch,
    stage,
) -> dict[str, float]:
    model.train()
    set_stage_trainability(model, stage)
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    interaction_gradient_seen = [False, False, False]
    progress = tqdm(
        loader,
        desc=f"A211 {epoch + 1}/5 {stage}",
        ascii=True,
    )
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, terms = compute_training_loss(
            model, batch, device, model_cfg, stage,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite A2.11 loss: {float(loss)}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.base_model.parameters()):
            raise RuntimeError("canonical parameter received a gradient")
        if model.base_model.training or model.smoke_expert.decoder.training:
            raise RuntimeError("canonical or decoder BN statistics entered train mode")
        if stage == "router_warmup":
            if any(
                parameter.grad is not None
                for parameter in model.smoke_expert.parameters()
            ):
                raise RuntimeError("expert received a warm-up gradient")
        else:
            for index, level in enumerate(model.smoke_expert.levels):
                gradient = level.interaction.visible_qkv[0].weight.grad
                if gradient is not None and gradient.abs().max().item() > 0.0:
                    interaction_gradient_seen[index] = True
        trainable = [
            parameter for parameter in model.trainable_parameters()
            if parameter.requires_grad
        ]
        for parameter in trainable:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite A2.11 gradient")
        torch.nn.utils.clip_grad_norm_(
            trainable, float(model_cfg["gradient_clip"]),
        )
        optimizer.step()
        batch_size = int(batch["vis"].shape[0])
        add_metrics(totals, terms, batch_size)
        sample_count += batch_size
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if sample_count != (EXPECTED_TRAIN_SAMPLES // int(model_cfg["batch_size"])) * int(model_cfg["batch_size"]):
        raise RuntimeError(f"unexpected trained sample count: {sample_count}")
    if stage != "router_warmup" and not all(interaction_gradient_seen):
        raise RuntimeError("not all three joint-source interactions received gradients")
    return {
        name: value / sample_count for name, value in totals.items()
    }


@torch.no_grad()
def validate_smoke(model, loader, device, severity: str) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    count = 0
    for batch in tqdm(
        loader, desc=f"validate-{severity}", ascii=True, leave=False,
    ):
        visible = batch["vis"].to(device, non_blocking=True)
        clean_visible = batch["vi_clean"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
        teacher = model.baseline(clean_visible, infrared)
        predicted, predicted_aux = model(
            visible, infrared, route_mode="smoke", return_aux=True,
        )
        oracle = model(
            visible, infrared, route_mode="smoke",
            gate_override=mask,
        )
        canonical = predicted_aux["canonical"]
        decision = model.automatic_decision(predicted_aux)
        automatic = torch.where(
            decision.view(-1, 1, 1, 1), predicted, canonical,
        )
        batch_size = visible.shape[0]
        totals["canonical_mae"] += float(
            (canonical - teacher).abs().mean(dim=(1, 2, 3)).sum()
        )
        totals["predicted_mae"] += float(
            (predicted - teacher).abs().mean(dim=(1, 2, 3)).sum()
        )
        totals["oracle_mae"] += float(
            (oracle - teacher).abs().mean(dim=(1, 2, 3)).sum()
        )
        totals["automatic_mae"] += float(
            (automatic - teacher).abs().mean(dim=(1, 2, 3)).sum()
        )
        totals["auto_admissions"] += int(decision.sum())
        totals["p_smoke"] += float(predicted_aux["p_smoke"].sum())
        totals["uncertainty"] += float(predicted_aux["uncertainty"].sum())
        totals["mask_coverage"] += float(predicted_aux["mask_coverage"].sum())
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"unexpected {severity} validation count: {count}")
    result = {name: value / count for name, value in totals.items()}
    baseline = max(result["canonical_mae"], 1e-12)
    result["predicted_recovery"] = (
        result["canonical_mae"] - result["predicted_mae"]
    ) / baseline
    result["oracle_recovery"] = (
        result["canonical_mae"] - result["oracle_mae"]
    ) / baseline
    result["automatic_recovery"] = (
        result["canonical_mae"] - result["automatic_mae"]
    ) / baseline
    result["auto_admission_rate"] = result.pop("auto_admissions")
    return {"severity": severity, **result}


@torch.no_grad()
def validate_clean(model, loader, device) -> dict[str, float | int]:
    model.eval()
    count = 0
    false_positives = 0
    maximum_error = 0.0
    p_smoke_sum = 0.0
    for batch in tqdm(loader, desc="validate-clean", ascii=True, leave=False):
        visible = batch["vis"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        canonical = model.baseline(visible, infrared)
        automatic, auxiliary = model(
            visible, infrared, route_mode="auto", return_aux=True,
        )
        decision = auxiliary["smoke_decision"]
        false_positives += int(decision.sum())
        maximum_error = max(
            maximum_error,
            float((automatic - canonical).abs().max()),
        )
        p_smoke_sum += float(auxiliary["p_smoke"].sum())
        count += visible.shape[0]
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"unexpected clean validation count: {count}")
    return {
        "samples": count,
        "auto_false_positives": false_positives,
        "auto_identity_max_error": maximum_error,
        "mean_p_smoke": p_smoke_sum / count,
    }


def final_validation(model, loaders, device, acceptance_cfg) -> dict:
    rows = [
        validate_smoke(model, loaders[severity], device, severity)
        for severity in SEVERITIES
    ]
    clean = validate_clean(model, loaders["clean"], device)
    predicted_recovery = float(np.mean([
        row["predicted_recovery"] for row in rows
    ]))
    oracle_recovery = float(np.mean([
        row["oracle_recovery"] for row in rows
    ]))
    automatic_recovery = float(np.mean([
        row["automatic_recovery"] for row in rows
    ]))
    ratio = predicted_recovery / max(oracle_recovery, 1e-12)
    preliminary = {
        "teacher_macro_mae_reduction": (
            predicted_recovery
            >= float(acceptance_cfg["teacher_macro_mae_reduction_min"])
        ),
        "clean_identity_mae": (
            float(clean["auto_identity_max_error"])
            <= float(acceptance_cfg["clean_identity_mae_max"])
        ),
        "auto_clean_false_positives": (
            int(clean["auto_false_positives"])
            <= int(acceptance_cfg["auto_clean_false_positives_max"])
        ),
        "predicted_oracle_recovery_ratio": (
            ratio
            >= float(acceptance_cfg["predicted_oracle_recovery_ratio_min"])
        ),
    }
    preliminary["all_pass"] = all(preliminary.values())
    return {
        "smoke_by_severity": rows,
        "clean": clean,
        "predicted_macro_recovery": predicted_recovery,
        "oracle_macro_recovery": oracle_recovery,
        "automatic_macro_recovery": automatic_recovery,
        "predicted_oracle_recovery_ratio": ratio,
        "preliminary_gates": preliminary,
        "rank_family_evaluation_pending": True,
    }


def csv_fieldnames(rows: list[dict]) -> list[str]:
    """Return a stable union for stage-heterogeneous history records."""
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    return fieldnames


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot save an empty A2.11 CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def random_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    stage,
    history,
    config_path,
    config_sha,
    baseline_sha,
) -> dict:
    return {
        "protocol": "A211_SCJSFE_INTERNAL_975_108_FIXED_5EP",
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "stage": stage,
        "stage_sequence": LOCKED_EPOCH_STAGES,
        "training_seed": 20260911,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "baseline_sha256": baseline_sha,
        "router_state_dict": cpu_state(model.router),
        "expert_state_dict": cpu_state(model.smoke_expert),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "random_state": random_state(),
        "official_test_loaded": False,
    }


def resume_checkpoint(path, model, optimizer, config_sha, baseline_sha):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol"] != "A211_SCJSFE_INTERNAL_975_108_FIXED_5EP":
        raise RuntimeError("A2.11 resume protocol mismatch")
    if checkpoint["config_sha256"] != config_sha:
        raise RuntimeError("A2.11 resume config SHA256 mismatch")
    if checkpoint["baseline_sha256"] != baseline_sha:
        raise RuntimeError("A2.11 resume baseline SHA256 mismatch")
    if checkpoint["official_test_loaded"] is not False:
        raise RuntimeError("A2.11 resume checkpoint accessed official data")
    if tuple(checkpoint["stage_sequence"]) != LOCKED_EPOCH_STAGES:
        raise RuntimeError("A2.11 resume stage sequence mismatch")
    router_result = model.router.load_state_dict(
        checkpoint["router_state_dict"], strict=True,
    )
    expert_result = model.smoke_expert.load_state_dict(
        checkpoint["expert_state_dict"], strict=True,
    )
    if (
        router_result.missing_keys or router_result.unexpected_keys
        or expert_result.missing_keys or expert_result.unexpected_keys
    ):
        raise RuntimeError("A2.11 resume state mismatch")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(checkpoint["random_state"])
    start_epoch = int(checkpoint["next_epoch"])
    history = list(checkpoint["history"])
    if start_epoch != len(history) or not 0 <= start_epoch <= 5:
        raise RuntimeError("A2.11 resume epoch/history mismatch")
    return start_epoch, history


def main() -> None:
    from utils.config import load_config

    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    cfg = load_config(config_path)
    model_cfg = validate_config(cfg)
    device, device_index = initialize_cuda(args.device)
    seed_everything(int(model_cfg["schedule"]["seed"]))

    model, base_checkpoint, baseline_sha = build_composite(
        cfg, model_cfg, device,
    )
    all_trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    canonical_before = cpu_state(model.base_model)
    output = Path(cfg["output_dir"])
    complete_path = output / "training_complete.json"
    checkpoint_path = output / "last_checkpoint.pth"
    if args.resume:
        if not output.is_dir() or not checkpoint_path.is_file():
            raise FileNotFoundError("A2.11 resume checkpoint is absent")
        if complete_path.exists():
            raise FileExistsError("A2.11 training is already complete")
    else:
        if output.exists():
            raise FileExistsError(f"REFUSED: A2.11 output exists: {output}")
        output.mkdir(parents=True)
        (output / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8",
        )

    start_epoch = 0
    history = []
    if args.resume:
        start_epoch, history = resume_checkpoint(
            checkpoint_path, model, optimizer, config_sha, baseline_sha,
        )
    else:
        atomic_torch_save(
            checkpoint_payload(
                model, optimizer, -1, "initialized", history,
                config_path, config_sha, baseline_sha,
            ),
            checkpoint_path,
        )

    print("config:", config_path)
    print("config_sha256:", config_sha)
    print("baseline_sha256:", baseline_sha)
    print("canonical_phase:", base_checkpoint.get("phase"))
    print("canonical_epoch:", base_checkpoint.get("epoch"))
    print("train_samples:", EXPECTED_TRAIN_SAMPLES)
    print("validation_samples:", EXPECTED_VALIDATION_SAMPLES)
    print("stage_sequence:", LOCKED_EPOCH_STAGES)
    print("start_epoch:", start_epoch)
    print("trainable_parameters:", sum(p.numel() for p in all_trainable))
    print("official_test_loaded: False")

    started = time.time()
    for epoch in range(start_epoch, int(model_cfg["schedule"]["epochs"])):
        stage = stage_for_epoch(epoch)
        loader = make_train_loader(cfg, model_cfg, epoch)
        epoch_started = time.time()
        metrics = train_epoch(
            model, loader, optimizer, device, model_cfg, epoch, stage,
        )
        base_update = maximum_state_update(canonical_before, model.base_model)
        if base_update != 0.0:
            raise RuntimeError(f"canonical state changed by {base_update}")
        record = {
            "epoch": epoch,
            "stage": stage,
            "trained_samples": (
                EXPECTED_TRAIN_SAMPLES // int(model_cfg["batch_size"])
            ) * int(model_cfg["batch_size"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            "canonical_state_max_update": base_update,
            **{f"train_{name}": value for name, value in metrics.items()},
        }
        history.append(record)
        payload = checkpoint_payload(
            model, optimizer, epoch, stage, history,
            config_path, config_sha, baseline_sha,
        )
        atomic_torch_save(payload, checkpoint_path)
        print("epoch_summary:", json.dumps(record, sort_keys=True))
        print("checkpoint_saved:", checkpoint_path, file_sha256(checkpoint_path))

    if len(history) != 5:
        raise RuntimeError(f"A2.11 history length is not five: {len(history)}")
    validation_loaders = build_validation_loaders(cfg, model_cfg)
    validation = final_validation(
        model, validation_loaders, device, model_cfg["acceptance"],
    )
    final_path = output / "final_epoch4.pth"
    final_payload = {
        "protocol": "A211_SCJSFE_INTERNAL_975_108_FIXED_5EP",
        "epoch": 4,
        "stage": stage_for_epoch(4),
        "stage_sequence": LOCKED_EPOCH_STAGES,
        "training_seed": int(model_cfg["schedule"]["seed"]),
        "config_sha256": config_sha,
        "baseline_sha256": baseline_sha,
        "router_state_dict": cpu_state(model.router),
        "expert_state_dict": cpu_state(model.smoke_expert),
        "validation": validation,
        "official_test_loaded": False,
        "rank_family_evaluation_pending": True,
    }
    atomic_torch_save(final_payload, final_path)
    save_csv(output / "history.csv", history)
    save_csv(
        output / "final_validation_by_severity.csv",
        validation["smoke_by_severity"],
    )
    completion = {
        "status": "training_complete_evaluation_pending",
        "protocol": "A211_SCJSFE_INTERNAL_975_108_FIXED_5EP",
        "epochs": 5,
        "final_epoch": 4,
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": file_sha256(final_path),
        "baseline_sha256": baseline_sha,
        "config_sha256": config_sha,
        "validation": validation,
        "canonical_state_max_update": maximum_state_update(
            canonical_before, model.base_model,
        ),
        "rank_family_evaluation_pending": True,
        "eligible_candidate": None,
        "official_test_loaded": False,
        "seconds": time.time() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device_index) / 1024**3
        ),
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8",
    )
    print("final_validation:", json.dumps(validation, sort_keys=True))
    print("final_checkpoint:", final_path.resolve())
    print("final_checkpoint_sha256:", completion["final_checkpoint_sha256"])
    print("canonical_state_max_update:", completion["canonical_state_max_update"])
    print("rank_family_evaluation_pending: True")
    print("eligible_candidate: None")
    print("official_test_loaded: False")
    print("peak_cuda_memory_gib:", round(completion["peak_cuda_memory_gib"], 3))
    print("A211_SCJSFE_FIXED_5EP_TRAINING_COMPLETE_EVALUATION_PENDING")


if __name__ == "__main__":
    main()
