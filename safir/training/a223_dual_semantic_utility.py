"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

from collections import Counter

import torch

from tqdm import tqdm

from models.a223_dual_semantic_utility import a223_phase0_loss, exact_scalar_l1_multiplier

from safir.training.a217_mask_supervised_support import make_a217_train_loader

EXPECTED_TRAINED_ITEMS = 974

def make_train_loader(cfg, a211, training, epoch: int):
    return make_a217_train_loader(
        cfg, a211, {"schedule": dict(training["schedule"])}, epoch
    )

def compute_training_loss(model, batch, device, training):
    visible = batch["vis"].to(device, non_blocking=True)
    clean_visible = batch["vi_clean"].to(device, non_blocking=True)
    infrared = batch["ir"].to(device, non_blocking=True)
    mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
    forced, smoke = model(visible, infrared, route_mode="smoke", return_aux=True)
    clean = model.semantic_prediction(clean_visible, infrared)
    canonical = smoke["canonical"]
    with torch.no_grad():
        teacher = model.source_model(
            clean_visible, infrared, route_mode="canonical"
        )
        task_multiplier, optimum_l1 = exact_scalar_l1_multiplier(
            canonical, smoke["pre_utility_residual"], teacher
        )
        canonical_l1 = (canonical.float() - teacher.float()).abs().mean(dim=1)
        attainable_gain = (canonical_l1 - optimum_l1).clamp_min(0.0)
    loss, terms = a223_phase0_loss(
        smoke,
        clean,
        mask,
        forced,
        canonical,
        teacher,
        task_multiplier,
        attainable_gain,
        training["loss"],
    )
    diagnostics = {
        "task_multiplier_mean": task_multiplier.mean(),
        "attainable_gain_mean": attainable_gain.mean(),
        "smoke_localization_mean": smoke["localization_support"].mean(),
        "smoke_utility_mean": smoke["utility_multiplier"].mean(),
        "clean_localization_mean": clean["localization_support"].mean(),
        "clean_utility_mean": clean["utility_multiplier"].mean(),
        "localization_minimum": smoke["localization_support"].min(),
        "localization_maximum": smoke["localization_support"].max(),
        "utility_minimum": smoke["utility_multiplier"].min(),
        "utility_maximum": smoke["utility_multiplier"].max(),
    }
    return loss, terms, diagnostics

def train_epoch(model, loader, optimizer, device, training, epoch: int) -> dict:
    model.train()
    totals = Counter()
    sample_count = 0
    seen_gradients: set[str] = set()
    minima = {"localization_minimum": 1.0, "utility_minimum": 1.0}
    maxima = {"localization_maximum": 0.0, "utility_maximum": 0.0}
    progress = tqdm(loader, desc=f"A2.23 {epoch + 1}/5 dual-semantic", ascii=True)
    trainable = tuple(model.trainable_parameters())
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, terms, diagnostics = compute_training_loss(
            model, batch, device, training
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite A2.23 loss")
        loss.backward()
        if any(value.grad is not None for value in model.source_model.parameters()):
            raise RuntimeError("A2.23 gradient escaped frozen source")
        for name, value in model.predictor.named_parameters():
            if value.grad is not None and bool(torch.isfinite(value.grad).all()):
                seen_gradients.add(name)
        torch.nn.utils.clip_grad_norm_(
            trainable, float(training["schedule"]["gradient_clip"])
        )
        optimizer.step()
        batch_size = int(batch["vis"].shape[0])
        sample_count += batch_size
        for name, value in {**terms, **diagnostics}.items():
            if name not in {*minima, *maxima}:
                totals[name] += float(value.detach()) * batch_size
        for name in minima:
            minima[name] = min(minima[name], float(diagnostics[name].detach()))
        for name in maxima:
            maxima[name] = max(maxima[name], float(diagnostics[name].detach()))
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if sample_count != EXPECTED_TRAINED_ITEMS:
        raise RuntimeError(f"A2.23 trained sample count changed: {sample_count}")
    expected = {name for name, _ in model.predictor.named_parameters()}
    missing = sorted(expected - seen_gradients)
    if missing:
        raise RuntimeError(f"A2.23 predictor gradients missing: {missing}")
    if not all(0.0 <= value <= 1.0 for value in (*minima.values(), *maxima.values())):
        raise RuntimeError("A2.23 prediction escaped [0,1]")
    return {
        **{name: value / sample_count for name, value in totals.items()},
        **minima,
        **maxima,
        "all_dual_semantic_gradients_seen": True,
    }
