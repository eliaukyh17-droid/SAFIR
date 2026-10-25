"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

import hashlib

from collections import Counter

import torch

from torch.utils.data import DataLoader

from tqdm import tqdm

from models.a213_multiscale_standardized_router import multiscale_statistical_features, paired_binary_ranking_loss

from safir.training.a211_scjsfe_probe import make_train_loader

EXPECTED_TRAINED_ITEMS = 968

EXPECTED_STANDARDIZER_VECTORS = 1936

def make_a213_train_loader(cfg: dict, a211: dict, a213: dict, epoch: int) -> DataLoader:
    loader_cfg = dict(a211)
    loader_cfg["batch_size"] = int(a213["schedule"]["batch_size"])
    loader_cfg["schedule"] = dict(a213["schedule"])
    return make_train_loader(cfg, loader_cfg, epoch)

@torch.no_grad()
def fit_training_standardizer(model, loader, device) -> dict:
    model.eval()
    dimension = int(model.router.feature_dimension)
    total = torch.zeros(dimension, dtype=torch.float64)
    total_square = torch.zeros(dimension, dtype=torch.float64)
    paired_items = 0
    vectors = 0
    for batch in tqdm(loader, desc="A2.13 standardizer", ascii=True):
        visible = batch["vis"].to(device, non_blocking=True)
        clean_visible = batch["vi_clean"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        smoke_source = model.router.source_router(visible, infrared)
        clean_source = model.router.source_router(clean_visible, infrared)
        smoke = multiscale_statistical_features(
            model.router.source_router, smoke_source, visible, infrared,
        )
        clean = multiscale_statistical_features(
            model.router.source_router, clean_source, clean_visible, infrared,
        )
        joined = torch.cat((smoke, clean)).double()
        total += joined.sum(dim=0).cpu()
        total_square += joined.square().sum(dim=0).cpu()
        paired_items += int(visible.shape[0])
        vectors += int(joined.shape[0])
    if paired_items != EXPECTED_TRAINED_ITEMS:
        raise RuntimeError(f"A2.13 standardizer paired-item count changed: {paired_items}")
    if vectors != EXPECTED_STANDARDIZER_VECTORS:
        raise RuntimeError(f"A2.13 standardizer vector count changed: {vectors}")
    mean = total / vectors
    variance = (total_square / vectors - mean.square()).clamp_min(0.0)
    scale = variance.sqrt()
    model.router.set_standardizer(mean, scale)
    return {
        "paired_items": paired_items,
        "vectors": vectors,
        "minimum_raw_scale": float(scale.min()),
        "minimum_applied_scale": float(model.router.standardizer_scale.min()),
        "standardizer_sha256": standardizer_sha256(model.router),
    }

def standardizer_state(router) -> dict[str, torch.Tensor]:
    return {
        "mean": router.standardizer_mean.detach().cpu().clone(),
        "scale": router.standardizer_scale.detach().cpu().clone(),
        "fitted": router.standardizer_fitted.detach().cpu().clone(),
    }

def standardizer_sha256(router) -> str:
    digest = hashlib.sha256()
    state = standardizer_state(router)
    for name in ("mean", "scale", "fitted"):
        value = state[name].contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def train_epoch(model, loader, optimizer, device, loss_cfg, epoch: int) -> dict:
    model.eval()
    totals = Counter()
    sample_count = 0
    for batch in tqdm(loader, desc=f"A2.13 {epoch + 1}/5 score", ascii=True):
        visible = batch["vis"].to(device, non_blocking=True)
        clean_visible = batch["vi_clean"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        smoke_score = model.router(visible, infrared)["smoke_score"]
        clean_score = model.router(clean_visible, infrared)["smoke_score"]
        terms = paired_binary_ranking_loss(
            smoke_score,
            clean_score,
            binary_weight=float(loss_cfg["binary_weight"]),
            anchor_weight=float(loss_cfg["anchor_weight"]),
            pair_weight=float(loss_cfg["pair_weight"]),
            anchor_margin=float(loss_cfg["anchor_margin"]),
            pair_margin=float(loss_cfg["pair_margin"]),
        )
        loss = terms["total"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite A2.13 training loss")
        loss.backward()
        head_parameters = tuple(model.router.score_head.parameters())
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in head_parameters
        ):
            raise RuntimeError("A2.13 score-head gradient is absent or invalid")
        if any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if not name.startswith("router.score_head.")
        ):
            raise RuntimeError("A2.13 gradient escaped the score head")
        optimizer.step()
        batch_size = int(visible.shape[0])
        sample_count += batch_size
        for name, value in terms.items():
            totals[name] += float(value.detach()) * batch_size
    if sample_count != EXPECTED_TRAINED_ITEMS:
        raise RuntimeError(f"A2.13 trained sample count changed: {sample_count}")
    return {name: value / sample_count for name, value in totals.items()}
