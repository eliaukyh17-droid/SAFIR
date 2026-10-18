"""Train the single preregistered A2.23 dual-semantic utility model."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.a223_dual_semantic_utility import (
    a223_phase0_loss,
    exact_scalar_l1_multiplier,
)
from tools.preflight_a217_mask_supervised_support import maximum_state_update
from tools.preflight_a223_dual_semantic_utility import load_bound_model
from train_a211_scjsfe_probe import (
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_VALIDATION_SAMPLES,
    SEVERITIES,
    seed_everything,
    validate_manifests,
)
from train_a212_router_head_calibration import (
    atomic_save,
    cpu_state,
    file_sha256,
    initialize_cuda,
    random_state,
    restore_random_state,
    save_csv,
)
from train_a213_multiscale_standardized_router import build_a213_validation_loaders
from train_a217_mask_supervised_support import make_a217_train_loader
from utils.a223_protocol import validate_a223_protocol
from utils.a223_training_protocol import (
    EVIDENCE_FILES,
    PHASE0_FILES,
    PROTOCOL,
    validate_a223_training_protocol,
)


EXPECTED_TRAINED_ITEMS = 974
TRAINING_SEED = 20260924
STAGE = "dual_semantic_gain_weighted_utility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tifdfusion_a223_dual_semantic_utility_5ep.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_config(cfg: dict) -> tuple[dict, dict]:
    phase0 = validate_a223_protocol(cfg["a223_utility"])
    training = validate_a223_training_protocol(cfg["a223_training"])
    schedule = training["schedule"]
    if int(cfg["train"]["seed"]) != int(schedule["seed"]):
        raise ValueError("A2.23 top-level and schedule seeds differ")
    smoke = cfg["data"]["msrs"]["smoke"]
    if int(smoke["base_seed"]) != int(schedule["seed"]):
        raise ValueError("A2.23 smoke and schedule seeds differ")
    if not bool(smoke["enabled"]) or float(smoke["probability"]) != 1.0:
        raise ValueError("A2.23 training smoke contract changed")
    if tuple(smoke["severities"]) != SEVERITIES:
        raise ValueError("A2.23 training severities changed")
    if int(schedule["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS:
        raise ValueError("A2.23 trained-item count changed")
    if dict(training["loss"]) != dict(phase0["loss"]):
        raise ValueError("A2.23 loss changed from Phase-0")
    if int(training["trainable_parameters"]) != int(
        phase0["architecture"]["trainable_parameters"]
    ):
        raise ValueError("A2.23 trainable count changed from Phase-0")
    if float(training["acceptance"]["total_residual_cap"]) != float(
        phase0["architecture"]["total_residual_cap"]
    ):
        raise ValueError("A2.23 residual cap changed from Phase-0")
    inherited = cfg["a219_training"]["acceptance"]
    acceptance = training["acceptance"]
    for name in (
        "teacher_macro_mae_reduction_min",
        "forced_teacher_macro_mae_reduction_min",
        "clean_identity_mae_max",
        "auto_clean_false_positives_max",
        "total_residual_cap",
        "old_route_drops_max",
        "rejected_identity_mae_max",
        "smoky_source_average_rank_max",
        "clean_source_average_rank_max",
        "smoky_source_top3_min",
        "clean_source_top3_min",
        "family_average_rank_max",
    ):
        if acceptance[name] != inherited[name]:
            raise ValueError(f"A2.23 inherited acceptance changed: {name}")
    if (
        acceptance["localization_macro_soft_dice_min"]
        != inherited["conditioning_macro_soft_dice_min"]
        or acceptance["localization_macro_coverage_error_max"]
        != inherited["conditioning_macro_coverage_error_max"]
        or acceptance["clean_localization_mean_max"]
        != inherited["clean_conditioning_support_mean_max"]
    ):
        raise ValueError("A2.23 localization gates changed from inherited gates")
    validate_manifests(cfg)
    if int(training["train_manifest_samples"]) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("A2.23 train manifest declaration changed")
    if int(training["validation_manifest_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("A2.23 validation manifest declaration changed")
    if bool(training["official_test_loaded"]):
        raise RuntimeError("A2.23 official_test_loaded must remain false")
    return phase0, training


def verify_bound_files() -> None:
    for name, expected in {**PHASE0_FILES, **EVIDENCE_FILES}.items():
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"A2.23 bound file mismatch: {path}")


def load_training_model(cfg: dict, phase0: dict, device: torch.device):
    source_cfg = deepcopy(cfg)
    source_seed = int(source_cfg["a219_training"]["schedule"]["seed"])
    source_cfg["train"]["seed"] = source_seed
    source_cfg["data"]["msrs"]["smoke"]["base_seed"] = source_seed
    return load_bound_model(source_cfg, phase0, device)


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


def _support_statistics(support, target) -> tuple[torch.Tensor, torch.Tensor]:
    intersection = (support * target).sum(dim=(1, 2, 3))
    denominator = support.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    coverage = (support.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs()
    return dice, coverage


def _maximum_selected_error(left, right, selected) -> float:
    if not bool(selected.any()):
        return 0.0
    selected4 = selected.view(-1, 1, 1, 1).expand_as(left)
    return float((left - right)[selected4].abs().max())


@torch.no_grad()
def validate_smoke(model, loader, device, severity: str, epsilon: float) -> dict:
    model.eval()
    totals = Counter()
    maxima = Counter()
    count = 0
    for batch in tqdm(
        loader, desc=f"A2.23-validate-{severity}", ascii=True, leave=False
    ):
        visible = batch["vis"].to(device, non_blocking=True)
        clean_visible = batch["vi_clean"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
        teacher = model.source_model(clean_visible, infrared, route_mode="canonical")
        canonical = model(visible, infrared, route_mode="canonical")
        forced, forced_aux = model(
            visible, infrared, route_mode="smoke", return_aux=True
        )
        automatic, automatic_aux = model(
            visible, infrared, route_mode="auto", return_aux=True
        )
        task_multiplier, optimum_l1 = exact_scalar_l1_multiplier(
            canonical, forced_aux["pre_utility_residual"], teacher
        )
        task_forced = (
            canonical + task_multiplier * forced_aux["pre_utility_residual"]
        ).clamp(0.0, 1.0)
        decision = automatic_aux["smoke_decision"]
        task_automatic = torch.where(
            decision.view(-1, 1, 1, 1), task_forced, canonical
        )
        localization = forced_aux["localization_support"]
        utility = forced_aux["utility_multiplier"]
        localization_dice, localization_coverage = _support_statistics(
            localization, mask
        )
        canonical_pixel_l1 = (canonical - teacher).abs().mean(dim=1)
        gain = (canonical_pixel_l1 - optimum_l1).clamp_min(0.0)
        utility_l1 = (utility - task_multiplier).abs().squeeze(1)
        per_sample_weighted_l1 = (utility_l1 * gain).sum(dim=(1, 2)) / (
            gain.sum(dim=(1, 2)) + epsilon
        )
        batch_size = int(visible.shape[0])
        for name, value in (
            ("canonical_mae", canonical),
            ("forced_mae", forced),
            ("automatic_mae", automatic),
            ("task_forced_mae", task_forced),
            ("task_automatic_mae", task_automatic),
        ):
            totals[name] += float((value - teacher).abs().mean(dim=(1, 2, 3)).sum())
        totals["localization_soft_dice"] += float(localization_dice.sum())
        totals["localization_coverage_error"] += float(localization_coverage.sum())
        totals["gain_weighted_utility_l1"] += float(per_sample_weighted_l1.sum())
        totals["localization_mean"] += float(
            localization.mean(dim=(1, 2, 3)).sum()
        )
        totals["utility_mean"] += float(utility.mean(dim=(1, 2, 3)).sum())
        old_decision = automatic_aux["old_smoke_decision"]
        totals["old_admissions"] += int(old_decision.sum())
        totals["auto_admissions"] += int(decision.sum())
        totals["old_route_drops"] += int((old_decision & ~decision).sum())
        maxima["maximum_forced_residual"] = max(
            maxima["maximum_forced_residual"],
            float(forced_aux["bounded_residual"].abs().max()),
        )
        maxima["maximum_automatic_residual"] = max(
            maxima["maximum_automatic_residual"],
            float((automatic - canonical).abs().max()),
        )
        maxima["maximum_rejected_identity_error"] = max(
            maxima["maximum_rejected_identity_error"],
            _maximum_selected_error(automatic, canonical, ~decision),
        )
        if bool((optimum_l1 > canonical_pixel_l1 + 1.0e-6).any()):
            raise RuntimeError("A2.23 final task oracle worse than canonical")
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.23 {severity} validation count changed: {count}")
    averaged = {
        "canonical_mae",
        "forced_mae",
        "automatic_mae",
        "task_forced_mae",
        "task_automatic_mae",
        "localization_soft_dice",
        "localization_coverage_error",
        "gain_weighted_utility_l1",
        "localization_mean",
        "utility_mean",
    }
    result = {
        name: (value / count if name in averaged else int(value))
        for name, value in totals.items()
    }
    canonical_mae = max(float(result["canonical_mae"]), 1.0e-12)
    for route in ("forced", "automatic", "task_forced", "task_automatic"):
        result[f"{route}_recovery"] = (
            float(result["canonical_mae"]) - float(result[f"{route}_mae"])
        ) / canonical_mae
    task_gain = float(result["canonical_mae"]) - float(result["task_forced_mae"])
    predicted_gain = float(result["canonical_mae"]) - float(result["forced_mae"])
    result["utility_attainable_gain_capture"] = predicted_gain / max(
        task_gain, 1.0e-12
    )
    result["auto_admission_rate"] = int(result["auto_admissions"]) / count
    result.update({name: float(value) for name, value in maxima.items()})
    return {"severity": severity, **result}


@torch.no_grad()
def validate_clean(model, loader, device) -> dict:
    model.eval()
    count = 0
    false_positives = 0
    rescue_false_positives = 0
    maximum_error = 0.0
    localization_sum = 0.0
    utility_sum = 0.0
    for batch in tqdm(loader, desc="A2.23-validate-clean", ascii=True, leave=False):
        visible = batch["vis"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        canonical = model(visible, infrared, route_mode="canonical")
        automatic, auxiliary = model(
            visible, infrared, route_mode="auto", return_aux=True
        )
        batch_size = int(visible.shape[0])
        false_positives += int(auxiliary["smoke_decision"].sum())
        rescue_false_positives += int(auxiliary["rescue_decision"].sum())
        maximum_error = max(maximum_error, float((automatic - canonical).abs().max()))
        localization_sum += float(
            auxiliary["localization_support"].mean(dim=(1, 2, 3)).sum()
        )
        utility_sum += float(
            auxiliary["utility_multiplier"].mean(dim=(1, 2, 3)).sum()
        )
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.23 clean validation count changed: {count}")
    return {
        "samples": count,
        "auto_false_positives": false_positives,
        "rescue_false_positives": rescue_false_positives,
        "auto_identity_max_error": maximum_error,
        "localization_mean": localization_sum / count,
        "utility_mean": utility_sum / count,
    }


def final_validation(model, loaders, device, training) -> dict:
    epsilon = float(training["loss"]["gain_denominator_epsilon"])
    rows = [
        validate_smoke(model, loaders[severity], device, severity, epsilon)
        for severity in SEVERITIES
    ]
    clean = validate_clean(model, loaders["clean"], device)

    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in rows]))

    automatic_recovery = mean("automatic_recovery")
    forced_recovery = mean("forced_recovery")
    task_automatic_recovery = mean("task_automatic_recovery")
    task_forced_recovery = mean("task_forced_recovery")
    localization_dice = mean("localization_soft_dice")
    localization_coverage = mean("localization_coverage_error")
    utility_l1 = mean("gain_weighted_utility_l1")
    gain_capture = mean("utility_attainable_gain_capture")
    maximum_residual = max(
        max(row["maximum_forced_residual"] for row in rows),
        max(row["maximum_automatic_residual"] for row in rows),
    )
    maximum_rejected_error = max(
        row["maximum_rejected_identity_error"] for row in rows
    )
    old_route_drops = sum(int(row["old_route_drops"]) for row in rows)
    acceptance = training["acceptance"]
    gates = {
        "automatic_teacher_recovery": automatic_recovery
        >= float(acceptance["teacher_macro_mae_reduction_min"]),
        "forced_teacher_recovery": forced_recovery
        >= float(acceptance["forced_teacher_macro_mae_reduction_min"]),
        "clean_identity": float(clean["auto_identity_max_error"])
        <= float(acceptance["clean_identity_mae_max"]),
        "clean_false_positives": int(clean["auto_false_positives"])
        <= int(acceptance["auto_clean_false_positives_max"]),
        "residual_bound": maximum_residual
        <= float(acceptance["total_residual_cap"]) + 1.0e-6,
        "localization_soft_dice": localization_dice
        >= float(acceptance["localization_macro_soft_dice_min"]),
        "localization_coverage": localization_coverage
        <= float(acceptance["localization_macro_coverage_error_max"]),
        "clean_localization_suppression": float(clean["localization_mean"])
        <= float(acceptance["clean_localization_mean_max"]),
        "utility_attainable_gain_capture": gain_capture
        >= float(acceptance["utility_attainable_gain_capture_min"]),
        "utility_gain_weighted_l1": utility_l1
        <= float(acceptance["utility_gain_weighted_l1_max"]),
        "clean_utility_suppression": float(clean["utility_mean"])
        <= float(acceptance["clean_utility_mean_max"]),
        "old_route_retention": old_route_drops
        <= int(acceptance["old_route_drops_max"]),
        "rejected_identity": maximum_rejected_error
        <= float(acceptance["rejected_identity_mae_max"]),
        "task_oracle_dominance": task_forced_recovery + 1.0e-7
        >= forced_recovery,
    }
    gates["all_pass"] = all(gates.values())
    rank_pending = bool(gates["all_pass"])
    return {
        "smoke_by_severity": rows,
        "clean": clean,
        "automatic_macro_recovery": automatic_recovery,
        "forced_macro_recovery": forced_recovery,
        "task_automatic_macro_recovery": task_automatic_recovery,
        "task_forced_macro_recovery": task_forced_recovery,
        "localization_macro_soft_dice": localization_dice,
        "localization_macro_coverage_error": localization_coverage,
        "utility_gain_weighted_l1": utility_l1,
        "utility_attainable_gain_capture": gain_capture,
        "maximum_residual": maximum_residual,
        "maximum_rejected_identity_error": maximum_rejected_error,
        "old_route_drops": old_route_drops,
        "internal_gates": gates,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": rank_pending,
    }


def checkpoint_payload(
    model, optimizer, epoch, history, config_sha, evidence_hashes, rng_state=None
):
    return {
        "protocol": PROTOCOL,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "stage": STAGE,
        "training_seed": TRAINING_SEED,
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "phase0_files": PHASE0_FILES,
        "evidence_files": evidence_hashes,
        "predictor_state_dict": cpu_state(model.predictor),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "random_state": random_state() if rng_state is None else rng_state,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": False,
        "official_test_loaded": False,
    }


def resume_checkpoint(path, model, optimizer, config_sha, evidence_hashes):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol"] != PROTOCOL or checkpoint["stage"] != STAGE:
        raise RuntimeError("A2.23 resume protocol/stage mismatch")
    if (
        checkpoint["config_sha256"] != config_sha
        or checkpoint["phase0_files"] != PHASE0_FILES
        or checkpoint["evidence_files"] != evidence_hashes
    ):
        raise RuntimeError("A2.23 resume evidence binding mismatch")
    if (
        int(checkpoint["training_seed"]) != TRAINING_SEED
        or int(checkpoint["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS
        or checkpoint["eligible_candidate"] is not None
        or checkpoint["rank_family_evaluation_pending"] is not False
        or checkpoint["official_test_loaded"] is not False
    ):
        raise RuntimeError("A2.23 resume boundary mismatch")
    result = model.predictor.load_state_dict(
        checkpoint["predictor_state_dict"], strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("A2.23 predictor resume state mismatch")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(checkpoint["random_state"])
    start_epoch = int(checkpoint["next_epoch"])
    history = list(checkpoint["history"])
    if start_epoch != len(history) or not 0 <= start_epoch <= 5:
        raise RuntimeError("A2.23 resume epoch/history mismatch")
    if any(
        int(record.get("epoch", -1)) != index
        or record.get("stage") != STAGE
        or int(record.get("trained_samples", -1)) != EXPECTED_TRAINED_ITEMS
        for index, record in enumerate(history)
    ):
        raise RuntimeError("A2.23 resume history binding mismatch")
    return start_epoch, history


def main() -> None:
    from utils.config import load_config

    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    cfg = load_config(config_path)
    phase0, training = validate_config(cfg)
    verify_bound_files()
    device, device_index = initialize_cuda(args.device)
    torch.cuda.reset_peak_memory_stats(device_index)
    seed_everything(int(training["schedule"]["seed"]))
    model, trainable, a211, a213 = load_training_model(cfg, phase0, device)
    frozen_before = cpu_state(model.source_model)
    evidence_hashes = dict(EVIDENCE_FILES)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["schedule"]["learning_rate"]),
        weight_decay=float(training["schedule"]["weight_decay"]),
    )
    output = Path(cfg["output_dir"])
    checkpoint_path = output / "last_checkpoint.pth"
    complete_path = output / "training_complete.json"
    if args.resume:
        if not output.is_dir() or not checkpoint_path.is_file():
            raise FileNotFoundError("A2.23 resume checkpoint is absent")
        if complete_path.exists():
            raise FileExistsError("A2.23 training is already complete")
        start_epoch, history = resume_checkpoint(
            checkpoint_path, model, optimizer, config_sha, evidence_hashes
        )
    else:
        if output.exists():
            raise FileExistsError(f"REFUSED: A2.23 output exists: {output}")
        output.mkdir(parents=True)
        (output / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
        start_epoch = 0
        history = []
        atomic_save(
            checkpoint_payload(
                model, optimizer, -1, history, config_sha, evidence_hashes
            ),
            checkpoint_path,
        )
    if maximum_state_update(frozen_before, model.source_model) != 0.0:
        raise RuntimeError("A2.23 source changed during initialization")
    print("protocol:", PROTOCOL)
    print("config:", config_path)
    print("config_sha256:", config_sha)
    print("start_epoch:", start_epoch)
    print("trained_items_per_epoch:", EXPECTED_TRAINED_ITEMS)
    print("trainable_parameters:", sum(value.numel() for value in trainable))
    print("eligible_candidate: None")
    print("official_test_loaded: False")

    started = time.time()
    for epoch in range(start_epoch, int(training["schedule"]["epochs"])):
        epoch_started = time.time()
        metrics = train_epoch(
            model,
            make_train_loader(cfg, a211, training, epoch),
            optimizer,
            device,
            training,
            epoch,
        )
        frozen_update = maximum_state_update(frozen_before, model.source_model)
        if frozen_update != 0.0:
            raise RuntimeError(f"A2.23 frozen source changed by {frozen_update}")
        record = {
            "epoch": epoch,
            "stage": STAGE,
            "trained_samples": EXPECTED_TRAINED_ITEMS,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            "frozen_state_max_update": frozen_update,
            **{f"train_{name}": value for name, value in metrics.items()},
        }
        history.append(record)
        atomic_save(
            checkpoint_payload(
                model, optimizer, epoch, history, config_sha, evidence_hashes
            ),
            checkpoint_path,
        )
        print("epoch_summary:", json.dumps(record, sort_keys=True))
        print("checkpoint_saved:", checkpoint_path, file_sha256(checkpoint_path))
    if len(history) != 5:
        raise RuntimeError(f"A2.23 history length is not five: {len(history)}")
    validation = final_validation(
        model, build_a213_validation_loaders(cfg, a211, a213), device, training
    )
    frozen_update = maximum_state_update(frozen_before, model.source_model)
    if frozen_update != 0.0:
        raise RuntimeError("A2.23 source changed during final validation")
    rank_pending = bool(validation["rank_family_evaluation_pending"])
    final_path = output / "final_epoch4.pth"
    final_payload = {
        "protocol": PROTOCOL,
        "epoch": 4,
        "stage": STAGE,
        "training_seed": int(training["schedule"]["seed"]),
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "phase0_files": PHASE0_FILES,
        "evidence_files": evidence_hashes,
        "predictor_state_dict": cpu_state(model.predictor),
        "validation": validation,
        "internal_gates_all_pass": bool(validation["internal_gates"]["all_pass"]),
        "eligible_candidate": None,
        "rank_family_evaluation_pending": rank_pending,
        "official_test_loaded": False,
    }
    atomic_save(final_payload, final_path)
    save_csv(output / "history.csv", history)
    save_csv(
        output / "final_validation_by_severity.csv", validation["smoke_by_severity"]
    )
    status = (
        "training_complete_internal_gates_pass_rank_pending"
        if rank_pending
        else "training_complete_internal_gates_failed"
    )
    completion = {
        "status": status,
        "protocol": PROTOCOL,
        "epochs": 5,
        "final_epoch": 4,
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": file_sha256(final_path),
        "config_sha256": config_sha,
        "phase0_files": PHASE0_FILES,
        "evidence_files": evidence_hashes,
        "validation": validation,
        "internal_gates_all_pass": bool(validation["internal_gates"]["all_pass"]),
        "frozen_state_max_update": frozen_update,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": rank_pending,
        "official_test_loaded": False,
        "seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device_index) / 1024**3,
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not math.isfinite(completion["peak_cuda_memory_gib"]):
        raise RuntimeError("A2.23 peak CUDA memory is non-finite")
    print("final_validation:", json.dumps(validation, sort_keys=True))
    print("final_checkpoint:", final_path.resolve())
    print("final_checkpoint_sha256:", completion["final_checkpoint_sha256"])
    print("internal_gates_all_pass:", validation["internal_gates"]["all_pass"])
    print("frozen_state_max_update:", frozen_update)
    print("eligible_candidate: None")
    print("rank_family_evaluation_pending:", rank_pending)
    print("official_test_loaded: False")
    print("peak_cuda_memory_gib:", round(completion["peak_cuda_memory_gib"], 3))
    print("A223_DUAL_SEMANTIC_GAIN_WEIGHTED_UTILITY_FIXED_5EP_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
