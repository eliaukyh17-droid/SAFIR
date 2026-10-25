"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

import random

from collections import defaultdict

from pathlib import Path

import numpy as np

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm

from data.dataset import MSRSDataset, collate_fn

from models.a211_safe_joint_source_expert import SafeConditionalJointSourceFusion

from utils.a211_protocol import LOCKED_EPOCH_STAGES

EXPECTED_TRAIN_SAMPLES = 975

EXPECTED_VALIDATION_SAMPLES = 108

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
