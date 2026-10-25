"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

from collections import Counter

import torch

from tqdm import tqdm

from models.a217_mask_supervised_support import mask_supervision_loss

from safir.training.preflight_a217_mask_supervised_support import predictor_inputs

from safir.training.a215_source_preserving_refiner import make_a215_train_loader

EXPECTED_TRAINED_ITEMS = 974

def make_a217_train_loader(cfg, a211, training, epoch: int):
    loader_training = dict(cfg["a215_training"])
    loader_training["schedule"] = {
        **dict(loader_training["schedule"]),
        "seed": int(training["schedule"]["seed"]),
        "batch_size": int(training["schedule"]["batch_size"]),
    }
    return make_a215_train_loader(cfg, a211, loader_training, epoch)

def compute_training_loss(model, batch, device, training):
    visible = batch["vis"].to(device, non_blocking=True)
    clean_visible = batch["vi_clean"].to(device, non_blocking=True)
    infrared = batch["ir"].to(device, non_blocking=True)
    target = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
    smoke_evidence, smoke_features = predictor_inputs(model, visible, infrared)
    clean_evidence, clean_features = predictor_inputs(
        model, clean_visible, infrared,
    )
    smoke_prediction = model.predictor(smoke_evidence, smoke_features)
    clean_prediction = model.predictor(clean_evidence, clean_features)
    loss, terms = mask_supervision_loss(
        smoke_prediction,
        clean_prediction,
        target,
        training["loss_weights"],
    )
    diagnostics = {
        "smoke_support_mean": smoke_prediction["support"].mean(),
        "clean_support_mean": clean_prediction["support"].mean(),
        "target_coverage": target.mean(),
        "support_minimum": smoke_prediction["support"].min(),
        "support_maximum": smoke_prediction["support"].max(),
    }
    return loss, terms, diagnostics

def train_epoch(model, loader, optimizer, device, training, epoch):
    model.train()
    totals = Counter()
    sample_count = 0
    deep_gradient_seen = False
    support_minimum = 1.0
    support_maximum = 0.0
    progress = tqdm(loader, desc=f"A2.17 {epoch + 1}/5 support", ascii=True)
    trainable = tuple(model.trainable_parameters())
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, terms, diagnostics = compute_training_loss(
            model, batch, device, training,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite A2.17 support loss")
        loss.backward()
        if any(
            parameter.grad is not None
            for parameter in model.source_model.parameters()
        ):
            raise RuntimeError("A2.17 gradient escaped into frozen source")
        if not all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        ):
            raise FloatingPointError("A2.17 predictor gradient is absent or invalid")
        deep_gradient_seen = deep_gradient_seen or bool(
            model.predictor.fusion[0].weight.grad.abs().max() > 0
        )
        torch.nn.utils.clip_grad_norm_(
            trainable, float(training["schedule"]["gradient_clip"]),
        )
        optimizer.step()
        batch_size = int(batch["vis"].shape[0])
        sample_count += batch_size
        for name, value in {**terms, **diagnostics}.items():
            totals[name] += float(value.detach()) * batch_size
        support_minimum = min(
            support_minimum, float(diagnostics["support_minimum"].detach()),
        )
        support_maximum = max(
            support_maximum, float(diagnostics["support_maximum"].detach()),
        )
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if sample_count != EXPECTED_TRAINED_ITEMS:
        raise RuntimeError(f"A2.17 trained sample count changed: {sample_count}")
    if not deep_gradient_seen:
        raise RuntimeError("A2.17 deep predictor gradient was not observed")
    if not (0.0 <= support_minimum <= support_maximum <= 1.0):
        raise RuntimeError("A2.17 support escaped [0,1] during training")
    result = {
        name: value / sample_count
        for name, value in totals.items()
        if name not in {"support_minimum", "support_maximum"}
    }
    result.update({
        "support_minimum": support_minimum,
        "support_maximum": support_maximum,
        "deep_gradient_seen": True,
    })
    return result
