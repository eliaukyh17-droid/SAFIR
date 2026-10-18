"""Train the single preregistered A2.19 controller."""

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

from models.a219_dual_support_benefit_rescue import a219_phase0_loss
from tools.preflight_a217_mask_supervised_support import maximum_state_update
from tools.preflight_a219_dual_support_benefit_rescue import (
    benefit_target,
    load_bound_model,
)
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
from train_a213_multiscale_standardized_router import (
    build_a213_validation_loaders,
)
from train_a217_mask_supervised_support import make_a217_train_loader
from utils.a219_protocol import validate_a219_protocol
from utils.a219_training_protocol import (
    EVIDENCE_FILES,
    PHASE0_FILES,
    PROTOCOL,
    validate_a219_training_protocol,
)


EXPECTED_TRAINED_ITEMS = 974


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tifdfusion_a219_dual_support_benefit_rescue_5ep.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_config(cfg: dict) -> tuple[dict, dict, dict, dict]:
    phase0 = validate_a219_protocol(cfg["a219_controller"])
    training = validate_a219_training_protocol(cfg["a219_training"])
    a211 = cfg["a211_scjsfe"]
    a213 = cfg["a213_router"]
    schedule = training["schedule"]
    if int(cfg["train"]["seed"]) != int(schedule["seed"]):
        raise ValueError("A2.19 top-level and schedule seeds differ")
    smoke = cfg["data"]["msrs"]["smoke"]
    if int(smoke["base_seed"]) != int(schedule["seed"]):
        raise ValueError("A2.19 smoke and schedule seeds differ")
    if not bool(smoke["enabled"]) or float(smoke["probability"]) != 1.0:
        raise ValueError("A2.19 training smoke contract changed")
    if tuple(smoke["severities"]) != SEVERITIES:
        raise ValueError("A2.19 training severities changed")
    if int(schedule["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS:
        raise ValueError("A2.19 trained-item count changed")
    if int(training["validation_smoke_seed"]) != int(a213["validation_smoke_seed"]):
        raise ValueError("A2.19 validation seed changed")
    if dict(training["loss_weights"]) != dict(phase0["loss_weights"]):
        raise ValueError("A2.19 loss changed from Phase-0")
    if int(training["trainable_parameters"]) != int(
        phase0["architecture"]["trainable_parameters"]
    ):
        raise ValueError("A2.19 trainable count changed from Phase-0")
    inherited = cfg["a217_training"]["acceptance"]
    acceptance = training["acceptance"]
    for name in (
        "teacher_macro_mae_reduction_min",
        "forced_teacher_macro_mae_reduction_min",
        "clean_identity_mae_max",
        "auto_clean_false_positives_max",
        "total_residual_cap",
        "smoky_source_average_rank_max",
        "clean_source_average_rank_max",
        "smoky_source_top3_min",
        "clean_source_top3_min",
        "family_average_rank_max",
    ):
        if acceptance[name] != inherited[name]:
            raise ValueError(f"A2.19 inherited acceptance changed: {name}")
    for prefix in ("conditioning", "multiplier"):
        if (
            acceptance[f"{prefix}_macro_soft_dice_min"]
            != inherited["smoke_macro_soft_dice_min"]
            or acceptance[f"{prefix}_macro_coverage_error_max"]
            != inherited["smoke_macro_coverage_error_max"]
            or acceptance[f"clean_{prefix}_support_mean_max"]
            != inherited["clean_support_mean_max"]
        ):
            raise ValueError(f"A2.19 inherited {prefix} support gates changed")
    if float(acceptance["total_residual_cap"]) != float(
        phase0["architecture"]["total_residual_cap"]
    ):
        raise ValueError("A2.19 residual cap changed from Phase-0")
    if int(acceptance["old_route_drops_max"]) != 0:
        raise ValueError("A2.19 old route retention gate changed")
    if float(acceptance["rejected_identity_mae_max"]) != 0.0:
        raise ValueError("A2.19 reject identity gate changed")
    validate_manifests(cfg)
    if int(training["train_manifest_samples"]) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("A2.19 train manifest declaration changed")
    if int(training["validation_manifest_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("A2.19 validation manifest declaration changed")
    if bool(training["official_test_loaded"]):
        raise RuntimeError("A2.19 official_test_loaded must remain false")
    return phase0, training, a211, a213


def verify_bound_files(training: dict) -> None:
    for name, expected in {**PHASE0_FILES, **EVIDENCE_FILES}.items():
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"A2.19 bound file mismatch: {path}")


def load_training_model(cfg: dict, phase0: dict, device: torch.device):
    source_cfg = deepcopy(cfg)
    source_seed = int(source_cfg["a217_training"]["schedule"]["seed"])
    source_cfg["train"]["seed"] = source_seed
    source_cfg["data"]["msrs"]["smoke"]["base_seed"] = source_seed
    return load_bound_model(source_cfg, phase0, device)


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


def _support_statistics(support, target) -> tuple[torch.Tensor, torch.Tensor]:
    intersection = (support * target).sum(dim=(1, 2, 3))
    denominator = support.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    coverage = (support.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs()
    return dice, coverage


def _maximum_selected_error(
    left: torch.Tensor, right: torch.Tensor, selected: torch.Tensor
) -> float:
    if not bool(selected.any()):
        return 0.0
    selected4 = selected.view(-1, 1, 1, 1).expand_as(left)
    return float((left - right)[selected4].abs().max())


@torch.no_grad()
def validate_smoke(model, loader, device, severity: str) -> dict:
    model.eval()
    totals = Counter()
    count = 0
    maximum_forced_residual = 0.0
    maximum_automatic_residual = 0.0
    maximum_rejected_identity_error = 0.0
    maximum_oracle_zero_support_delta = 0.0
    for batch in tqdm(
        loader, desc=f"A2.19-validate-{severity}", ascii=True, leave=False
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
        oracle = model.source_model.source_model(
            visible, infrared, route_mode="smoke", support_override=mask
        )
        conditioning = forced_aux["conditioning_support"]
        multiplier = forced_aux["multiplier_support"]
        conditioning_dice, conditioning_coverage = _support_statistics(
            conditioning, mask
        )
        multiplier_dice, multiplier_coverage = _support_statistics(multiplier, mask)
        benefit = benefit_target(model, visible, clean_visible, infrared).bool()
        old_decision = automatic_aux["old_smoke_decision"]
        rescue_decision = automatic_aux["rescue_decision"]
        decision = automatic_aux["smoke_decision"]
        batch_size = int(visible.shape[0])
        for name, value in (
            ("canonical_mae", canonical),
            ("forced_mae", forced),
            ("oracle_mae", oracle),
            ("automatic_mae", automatic),
        ):
            totals[name] += float((value - teacher).abs().mean(dim=(1, 2, 3)).sum())
        totals["old_admissions"] += int(old_decision.sum())
        totals["rescue_admissions"] += int((rescue_decision & ~old_decision).sum())
        totals["auto_admissions"] += int(decision.sum())
        totals["old_route_drops"] += int((old_decision & ~decision).sum())
        totals["beneficial_samples"] += int(benefit.sum())
        totals["rescue_true_positives"] += int(
            (rescue_decision & ~old_decision & benefit).sum()
        )
        totals["rescue_false_positives"] += int(
            (rescue_decision & ~old_decision & ~benefit).sum()
        )
        totals["conditioning_support_soft_dice"] += float(conditioning_dice.sum())
        totals["conditioning_support_coverage_error"] += float(
            conditioning_coverage.sum()
        )
        totals["multiplier_support_soft_dice"] += float(multiplier_dice.sum())
        totals["multiplier_support_coverage_error"] += float(multiplier_coverage.sum())
        totals["conditioning_support_mean"] += float(
            conditioning.mean(dim=(1, 2, 3)).sum()
        )
        totals["multiplier_support_mean"] += float(multiplier.mean(dim=(1, 2, 3)).sum())
        maximum_forced_residual = max(
            maximum_forced_residual,
            float(forced_aux["bounded_residual"].abs().max()),
        )
        maximum_automatic_residual = max(
            maximum_automatic_residual, float((automatic - canonical).abs().max())
        )
        maximum_rejected_identity_error = max(
            maximum_rejected_identity_error,
            _maximum_selected_error(automatic, canonical, ~decision),
        )
        outside = mask <= 0.0
        if bool(outside.any()):
            maximum_oracle_zero_support_delta = max(
                maximum_oracle_zero_support_delta,
                float((oracle - canonical)[outside].abs().max()),
            )
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.19 {severity} validation count changed: {count}")
    averaged_names = {
        "canonical_mae",
        "forced_mae",
        "oracle_mae",
        "automatic_mae",
        "conditioning_support_soft_dice",
        "conditioning_support_coverage_error",
        "multiplier_support_soft_dice",
        "multiplier_support_coverage_error",
        "conditioning_support_mean",
        "multiplier_support_mean",
    }
    result = {
        name: (value / count if name in averaged_names else int(value))
        for name, value in totals.items()
    }
    canonical_mae = max(float(result["canonical_mae"]), 1.0e-12)
    for route in ("forced", "oracle", "automatic"):
        result[f"{route}_recovery"] = (
            float(result["canonical_mae"]) - float(result[f"{route}_mae"])
        ) / canonical_mae
    result["auto_admission_rate"] = int(result["auto_admissions"]) / count
    result["maximum_forced_residual"] = maximum_forced_residual
    result["maximum_automatic_residual"] = maximum_automatic_residual
    result["maximum_rejected_identity_error"] = maximum_rejected_identity_error
    result["maximum_oracle_zero_support_delta"] = maximum_oracle_zero_support_delta
    return {"severity": severity, **result}


@torch.no_grad()
def validate_clean(model, loader, device) -> dict:
    model.eval()
    count = 0
    false_positives = 0
    rescue_false_positives = 0
    maximum_error = 0.0
    conditioning_sum = 0.0
    multiplier_sum = 0.0
    for batch in tqdm(loader, desc="A2.19-validate-clean", ascii=True, leave=False):
        visible = batch["vis"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        canonical = model(visible, infrared, route_mode="canonical")
        automatic, auxiliary = model(
            visible, infrared, route_mode="auto", return_aux=True
        )
        prediction = model.controller_prediction(visible, infrared)
        batch_size = int(visible.shape[0])
        false_positives += int(auxiliary["smoke_decision"].sum())
        rescue_false_positives += int(auxiliary["rescue_decision"].sum())
        maximum_error = max(maximum_error, float((automatic - canonical).abs().max()))
        conditioning_sum += float(
            prediction["conditioning_support"].mean(dim=(1, 2, 3)).sum()
        )
        multiplier_sum += float(
            prediction["multiplier_support"].mean(dim=(1, 2, 3)).sum()
        )
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.19 clean validation count changed: {count}")
    return {
        "samples": count,
        "auto_false_positives": false_positives,
        "rescue_false_positives": rescue_false_positives,
        "auto_identity_max_error": maximum_error,
        "conditioning_support_mean": conditioning_sum / count,
        "multiplier_support_mean": multiplier_sum / count,
    }


def final_validation(model, loaders, device, training, source_report) -> dict:
    rows = [
        validate_smoke(model, loaders[severity], device, severity)
        for severity in SEVERITIES
    ]
    clean = validate_clean(model, loaders["clean"], device)
    automatic_recovery = float(np.mean([row["automatic_recovery"] for row in rows]))
    forced_recovery = float(np.mean([row["forced_recovery"] for row in rows]))
    oracle_recovery = float(np.mean([row["oracle_recovery"] for row in rows]))
    conditioning_dice = float(
        np.mean([row["conditioning_support_soft_dice"] for row in rows])
    )
    multiplier_dice = float(
        np.mean([row["multiplier_support_soft_dice"] for row in rows])
    )
    conditioning_coverage = float(
        np.mean([row["conditioning_support_coverage_error"] for row in rows])
    )
    multiplier_coverage = float(
        np.mean([row["multiplier_support_coverage_error"] for row in rows])
    )
    maximum_residual = max(
        max(row["maximum_forced_residual"] for row in rows),
        max(row["maximum_automatic_residual"] for row in rows),
    )
    maximum_rejected_error = max(row["maximum_rejected_identity_error"] for row in rows)
    maximum_zero_support_delta = max(
        row["maximum_oracle_zero_support_delta"] for row in rows
    )
    old_route_drops = sum(int(row["old_route_drops"]) for row in rows)
    source_oracle = float(source_report["validation"]["oracle_macro_recovery"])
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
        "conditioning_support_soft_dice": conditioning_dice
        >= float(acceptance["conditioning_macro_soft_dice_min"]),
        "multiplier_support_soft_dice": multiplier_dice
        >= float(acceptance["multiplier_macro_soft_dice_min"]),
        "conditioning_support_coverage": conditioning_coverage
        <= float(acceptance["conditioning_macro_coverage_error_max"]),
        "multiplier_support_coverage": multiplier_coverage
        <= float(acceptance["multiplier_macro_coverage_error_max"]),
        "clean_conditioning_support_suppression": float(
            clean["conditioning_support_mean"]
        )
        <= float(acceptance["clean_conditioning_support_mean_max"]),
        "clean_multiplier_support_suppression": float(clean["multiplier_support_mean"])
        <= float(acceptance["clean_multiplier_support_mean_max"]),
        "old_route_retention": old_route_drops
        <= int(acceptance["old_route_drops_max"]),
        "rejected_identity": maximum_rejected_error
        <= float(acceptance["rejected_identity_mae_max"]),
        "oracle_zero_support": maximum_zero_support_delta == 0.0,
        "oracle_reference_binding": abs(oracle_recovery - source_oracle) <= 1.0e-10,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "smoke_by_severity": rows,
        "clean": clean,
        "automatic_macro_recovery": automatic_recovery,
        "forced_macro_recovery": forced_recovery,
        "oracle_macro_recovery": oracle_recovery,
        "conditioning_macro_soft_dice": conditioning_dice,
        "multiplier_macro_soft_dice": multiplier_dice,
        "conditioning_macro_coverage_error": conditioning_coverage,
        "multiplier_macro_coverage_error": multiplier_coverage,
        "maximum_residual": maximum_residual,
        "maximum_rejected_identity_error": maximum_rejected_error,
        "maximum_oracle_zero_support_delta": maximum_zero_support_delta,
        "old_route_drops": old_route_drops,
        "internal_gates": gates,
        "rank_family_evaluation_pending": bool(gates["all_pass"]),
    }


def checkpoint_payload(
    model, optimizer, epoch, history, config_sha, evidence_hashes, rng_state=None
):
    return {
        "protocol": PROTOCOL,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "training_seed": 20260920,
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "phase0_files": PHASE0_FILES,
        "evidence_files": evidence_hashes,
        "controller_state_dict": cpu_state(model.controller),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "random_state": random_state() if rng_state is None else rng_state,
        "official_test_loaded": False,
    }


def resume_checkpoint(
    path, model, optimizer, config_sha, evidence_hashes
) -> tuple[int, list[dict]]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol"] != PROTOCOL:
        raise RuntimeError("A2.19 resume protocol mismatch")
    if (
        checkpoint["config_sha256"] != config_sha
        or checkpoint["phase0_files"] != PHASE0_FILES
        or checkpoint["evidence_files"] != evidence_hashes
    ):
        raise RuntimeError("A2.19 resume evidence binding mismatch")
    if (
        int(checkpoint["training_seed"]) != 20260920
        or int(checkpoint["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS
        or checkpoint["official_test_loaded"] is not False
    ):
        raise RuntimeError("A2.19 resume schedule or data binding mismatch")
    result = model.controller.load_state_dict(
        checkpoint["controller_state_dict"], strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("A2.19 controller resume state mismatch")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(checkpoint["random_state"])
    epoch = int(checkpoint["epoch"])
    start_epoch = int(checkpoint["next_epoch"])
    history = list(checkpoint["history"])
    if (
        start_epoch != epoch + 1
        or start_epoch != len(history)
        or not 0 <= start_epoch <= 5
    ):
        raise RuntimeError("A2.19 resume epoch/history mismatch")
    if any(
        int(record.get("epoch", -1)) != index
        or record.get("stage") != "dual_support_benefit_rescue"
        or int(record.get("trained_samples", -1)) != EXPECTED_TRAINED_ITEMS
        for index, record in enumerate(history)
    ):
        raise RuntimeError("A2.19 resume history binding mismatch")
    return start_epoch, history


def main() -> None:
    from utils.config import load_config

    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    cfg = load_config(config_path)
    phase0, training, a211, a213 = validate_config(cfg)
    verify_bound_files(training)
    device, device_index = initialize_cuda(args.device)
    seed_everything(int(training["schedule"]["seed"]))
    model, trainable, loaded_a211, loaded_a213 = load_training_model(
        cfg, phase0, device
    )
    if loaded_a211 != a211 or loaded_a213 != a213:
        raise RuntimeError("A2.19 source configuration changed")
    frozen_before = cpu_state(model.source_model)
    evidence_hashes = dict(EVIDENCE_FILES)
    source_report = json.loads(
        Path(
            "experiments/tifdfusion_a217_mask_supervised_support_5ep/"
            "training_complete.json"
        ).read_text(encoding="utf-8")
    )
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
            raise FileNotFoundError("A2.19 resume checkpoint is absent")
        if complete_path.exists():
            raise FileExistsError("A2.19 training is already complete")
        start_epoch, history = resume_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config_sha,
            evidence_hashes,
        )
    else:
        if output.exists():
            raise FileExistsError(f"REFUSED: A2.19 output exists: {output}")
        output.mkdir(parents=True)
        (output / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
        start_epoch = 0
        history = []
        atomic_save(
            checkpoint_payload(
                model,
                optimizer,
                -1,
                history,
                config_sha,
                evidence_hashes,
            ),
            checkpoint_path,
        )
    if maximum_state_update(frozen_before, model.source_model) != 0.0:
        raise RuntimeError("A2.19 source changed during initialization")
    print("config:", config_path)
    print("config_sha256:", config_sha)
    print("start_epoch:", start_epoch)
    print("trained_items_per_epoch:", EXPECTED_TRAINED_ITEMS)
    print("trainable_parameters:", sum(value.numel() for value in trainable))
    print("official_test_loaded: False")

    started = time.time()
    for epoch in range(start_epoch, int(training["schedule"]["epochs"])):
        epoch_started = time.time()
        metrics = train_epoch(
            model,
            make_a219_train_loader(cfg, a211, training, epoch),
            optimizer,
            device,
            training,
            epoch,
        )
        frozen_update = maximum_state_update(frozen_before, model.source_model)
        if frozen_update != 0.0:
            raise RuntimeError(f"A2.19 frozen source changed by {frozen_update}")
        record = {
            "epoch": epoch,
            "stage": "dual_support_benefit_rescue",
            "trained_samples": EXPECTED_TRAINED_ITEMS,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            "frozen_state_max_update": frozen_update,
            **{f"train_{name}": value for name, value in metrics.items()},
        }
        history.append(record)
        atomic_save(
            checkpoint_payload(
                model,
                optimizer,
                epoch,
                history,
                config_sha,
                evidence_hashes,
            ),
            checkpoint_path,
        )
        print("epoch_summary:", json.dumps(record, sort_keys=True))
        print("checkpoint_saved:", checkpoint_path, file_sha256(checkpoint_path))
    if len(history) != 5:
        raise RuntimeError(f"A2.19 history length is not five: {len(history)}")
    validation = final_validation(
        model,
        build_a213_validation_loaders(cfg, a211, a213),
        device,
        training,
        source_report,
    )
    frozen_update = maximum_state_update(frozen_before, model.source_model)
    if frozen_update != 0.0:
        raise RuntimeError("A2.19 source changed during final validation")
    final_path = output / "final_epoch4.pth"
    final_payload = {
        "protocol": PROTOCOL,
        "epoch": 4,
        "stage": "dual_support_benefit_rescue",
        "training_seed": int(training["schedule"]["seed"]),
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "phase0_files": PHASE0_FILES,
        "evidence_files": evidence_hashes,
        "controller_state_dict": cpu_state(model.controller),
        "validation": validation,
        "official_test_loaded": False,
        "rank_family_evaluation_pending": validation["rank_family_evaluation_pending"],
        "eligible_candidate": None,
    }
    atomic_save(final_payload, final_path)
    save_csv(output / "history.csv", history)
    save_csv(
        output / "final_validation_by_severity.csv", validation["smoke_by_severity"]
    )
    status = (
        "training_complete_rank_evaluation_pending"
        if validation["internal_gates"]["all_pass"]
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
        "frozen_state_max_update": frozen_update,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": validation["rank_family_evaluation_pending"],
        "official_test_loaded": False,
        "seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device_index) / 1024**3,
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not math.isfinite(completion["peak_cuda_memory_gib"]):
        raise RuntimeError("A2.19 peak CUDA memory is non-finite")
    print("final_validation:", json.dumps(validation, sort_keys=True))
    print("final_checkpoint:", final_path.resolve())
    print("final_checkpoint_sha256:", completion["final_checkpoint_sha256"])
    print("frozen_state_max_update:", frozen_update)
    print("eligible_candidate: None")
    print(
        "rank_family_evaluation_pending:", validation["rank_family_evaluation_pending"]
    )
    print("official_test_loaded: False")
    print("peak_cuda_memory_gib:", round(completion["peak_cuda_memory_gib"], 3))
    print("A219_DUAL_SUPPORT_BENEFIT_RESCUE_FIXED_5EP_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
