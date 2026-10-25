"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

from collections import Counter

import torch

from tqdm import tqdm

from models.a219_dual_support_benefit_rescue import a219_phase0_loss

from safir.training.preflight_a219_dual_support_benefit_rescue import benefit_target

from safir.training.a217_mask_supervised_support import make_a217_train_loader

EXPECTED_TRAINED_ITEMS = 974

def make_a219_train_loader(cfg, a211, training, epoch: int):
    loader_protocol = {"schedule": dict(training["schedule"])}
    return make_a217_train_loader(cfg, a211, loader_protocol, epoch)

def compute_training_loss(model, batch, device, training):
    visible = batch["vis"].to(device, non_blocking=True)
    clean_visible = batch["vi_clean"].to(device, non_blocking=True)
    infrared = batch["ir"].to(device, non_blocking=True)
    mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
    target = benefit_target(model, visible, clean_visible, infrared)
    smoke = model.controller_prediction(visible, infrared)
    clean = model.controller_prediction(clean_visible, infrared)
    loss, terms = a219_phase0_loss(smoke, clean, mask, target, training["loss_weights"])
    diagnostics = {
        "beneficial_rate": target.mean(),
        "conditioning_support_mean": smoke["conditioning_support"].mean(),
        "multiplier_support_mean": smoke["multiplier_support"].mean(),
        "clean_conditioning_support_mean": clean["conditioning_support"].mean(),
        "clean_multiplier_support_mean": clean["multiplier_support"].mean(),
        "rescue_probability_mean": smoke["rescue_probability"].mean(),
        "conditioning_support_minimum": smoke["conditioning_support"].min(),
        "conditioning_support_maximum": smoke["conditioning_support"].max(),
        "multiplier_support_minimum": smoke["multiplier_support"].min(),
        "multiplier_support_maximum": smoke["multiplier_support"].max(),
    }
    return loss, terms, diagnostics

def train_epoch(model, loader, optimizer, device, training, epoch: int) -> dict:
    model.train()
    totals = Counter()
    sample_count = 0
    seen_gradients: set[str] = set()
    minima = {
        "conditioning_support_minimum": 1.0,
        "multiplier_support_minimum": 1.0,
    }
    maxima = {
        "conditioning_support_maximum": 0.0,
        "multiplier_support_maximum": 0.0,
    }
    progress = tqdm(loader, desc=f"A2.19 {epoch + 1}/5 controller", ascii=True)
    trainable = tuple(model.trainable_parameters())
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, terms, diagnostics = compute_training_loss(model, batch, device, training)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite A2.19 controller loss")
        loss.backward()
        if any(value.grad is not None for value in model.source_model.parameters()):
            raise RuntimeError("A2.19 gradient escaped frozen source")
        for name, value in model.controller.named_parameters():
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
        raise RuntimeError(f"A2.19 trained sample count changed: {sample_count}")
    expected_gradients = {name for name, _ in model.controller.named_parameters()}
    missing = sorted(expected_gradients - seen_gradients)
    if missing:
        raise RuntimeError(f"A2.19 controller gradients missing: {missing}")
    if not all(0.0 <= value <= 1.0 for value in (*minima.values(), *maxima.values())):
        raise RuntimeError("A2.19 support escaped [0,1]")
    return {
        **{name: value / sample_count for name, value in totals.items()},
        **minima,
        **maxima,
        "all_controller_gradients_seen": True,
    }
