"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

from collections import Counter

import torch

from tqdm import tqdm

from models.a215_source_preserving_refiner import source_preserving_loss

from safir.training.a211_scjsfe_probe import make_train_loader

from utils.a215_training_protocol import stage_for_epoch

EXPECTED_TRAINED_ITEMS = 974

def make_a215_train_loader(cfg, a211, training, epoch: int):
    loader_cfg = dict(a211)
    loader_cfg["batch_size"] = int(training["schedule"]["batch_size"])
    loader_cfg["schedule"] = {
        **dict(loader_cfg["schedule"]),
        "seed": int(training["schedule"]["seed"]),
    }
    return make_train_loader(cfg, loader_cfg, epoch)

def compute_training_loss(model, batch, device, phase0, stage):
    if stage not in {"oracle_support", "frozen_support"}:
        raise ValueError(f"unsupported A2.15 training stage: {stage}")
    visible = batch["vis"].to(device, non_blocking=True)
    clean_visible = batch["vi_clean"].to(device, non_blocking=True)
    infrared = batch["ir"].to(device, non_blocking=True)
    mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
    with torch.no_grad():
        teacher = model.source_model(
            clean_visible,
            infrared,
            route_mode="canonical",
        )
    support_override = mask if stage == "oracle_support" else None
    output, auxiliary = model(
        visible,
        infrared,
        route_mode="smoke",
        support_override=support_override,
        return_aux=True,
    )
    loss, terms = source_preserving_loss(
        output,
        auxiliary["canonical"],
        teacher,
        visible,
        infrared,
        mask,
        auxiliary,
        phase0["loss_weights"],
    )
    diagnostics = {
        "canonical_teacher_mae": (auxiliary["canonical"] - teacher).abs().mean(),
        "output_teacher_mae": (output - teacher).abs().mean(),
        "output_delta_abs": auxiliary["output_delta"].abs().mean(),
        "maximum_residual": auxiliary["bounded_residual"].abs().max(),
    }
    return loss, terms, diagnostics

def train_epoch(model, loader, optimizer, device, phase0, training, epoch):
    stage = stage_for_epoch(epoch)
    model.train()
    totals = Counter()
    sample_count = 0
    deep_gradient_seen = False
    maximum_residual = 0.0
    progress = tqdm(
        loader,
        desc=f"A2.15 {epoch + 1}/5 {stage}",
        ascii=True,
    )
    trainable = tuple(model.trainable_parameters())
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, terms, diagnostics = compute_training_loss(
            model,
            batch,
            device,
            phase0,
            stage,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite A2.15 loss")
        loss.backward()
        if any(
            parameter.grad is not None for parameter in model.source_model.parameters()
        ):
            raise RuntimeError("A2.15 gradient escaped into frozen source")
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        ):
            raise FloatingPointError("A2.15 refiner gradient is absent or invalid")
        if float(model.refiner.head.weight.grad.abs().max()) <= 0.0:
            raise RuntimeError("A2.15 residual-head gradient is zero")
        deep_gradient_seen = deep_gradient_seen or any(
            float(parameter.grad.abs().max()) > 0.0
            for name, parameter in model.refiner.named_parameters()
            if not name.startswith("head.")
        )
        torch.nn.utils.clip_grad_norm_(
            trainable,
            float(training["schedule"]["gradient_clip"]),
        )
        optimizer.step()
        batch_size = int(batch["vis"].shape[0])
        sample_count += batch_size
        for name, value in {**terms, **diagnostics}.items():
            totals[name] += float(value.detach()) * batch_size
        maximum_residual = max(
            maximum_residual,
            float(diagnostics["maximum_residual"].detach()),
        )
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if sample_count != EXPECTED_TRAINED_ITEMS:
        raise RuntimeError(f"A2.15 trained sample count changed: {sample_count}")
    if not deep_gradient_seen:
        raise RuntimeError("A2.15 deep refiner gradient was not observed")
    cap = float(training["acceptance"]["total_residual_cap"])
    if maximum_residual > cap + 1.0e-6:
        raise RuntimeError("A2.15 residual escaped hard bound during training")
    result = {
        name: value / sample_count
        for name, value in totals.items()
        if name != "maximum_residual"
    }
    result["maximum_residual"] = maximum_residual
    result["deep_gradient_seen"] = True
    baseline = max(result["canonical_teacher_mae"], 1.0e-12)
    result["teacher_recovery"] = (
        result["canonical_teacher_mae"] - result["output_teacher_mae"]
    ) / baseline
    return result
