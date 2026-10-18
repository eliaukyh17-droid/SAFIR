"""Train the single preregistered A2.17 local-support predictor."""

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

from models.a217_mask_supervised_support import mask_supervision_loss
from tools.preflight_a217_mask_supervised_support import (
    load_bound_model,
    maximum_state_update,
    predictor_inputs,
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
from train_a215_source_preserving_refiner import make_a215_train_loader
from utils.a217_protocol import validate_a217_protocol
from utils.a217_training_protocol import (
    PHASE0_FILES,
    PROTOCOL,
    validate_a217_training_protocol,
)


EXPECTED_TRAINED_ITEMS = 974


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tifdfusion_a217_mask_supervised_support_5ep.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_config(cfg: dict) -> tuple[dict, dict, dict, dict]:
    phase0 = validate_a217_protocol(cfg["a217_support"])
    training = validate_a217_training_protocol(cfg["a217_training"])
    a211 = cfg["a211_scjsfe"]
    a213 = cfg["a213_router"]
    schedule = training["schedule"]
    if int(cfg["train"]["seed"]) != int(schedule["seed"]):
        raise ValueError("A2.17 top-level and schedule seeds differ")
    smoke = cfg["data"]["msrs"]["smoke"]
    if int(smoke["base_seed"]) != int(schedule["seed"]):
        raise ValueError("A2.17 smoke and schedule seeds differ")
    if not bool(smoke["enabled"]) or float(smoke["probability"]) != 1.0:
        raise ValueError("A2.17 training smoke contract changed")
    if tuple(smoke["severities"]) != SEVERITIES:
        raise ValueError("A2.17 training severities changed")
    if int(schedule["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS:
        raise ValueError("A2.17 trained-item count changed")
    if int(training["validation_smoke_seed"]) != int(
        a213["validation_smoke_seed"]
    ):
        raise ValueError("A2.17 validation seed changed from A2.13")
    if dict(training["loss_weights"]) != dict(phase0["loss_weights"]):
        raise ValueError("A2.17 support loss changed from Phase-0")
    inherited = cfg["a215_training"]["acceptance"]
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
            raise ValueError(f"A2.17 inherited acceptance changed: {name}")
    if float(acceptance["total_residual_cap"]) != float(
        phase0["architecture"]["total_residual_cap"]
    ):
        raise ValueError("A2.17 residual cap changed from Phase-0")
    validate_manifests(cfg)
    if int(training["train_manifest_samples"]) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("A2.17 train manifest declaration changed")
    if int(training["validation_manifest_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("A2.17 validation manifest declaration changed")
    if bool(training["official_test_loaded"]):
        raise RuntimeError("A2.17 official_test_loaded must remain false")
    return phase0, training, a211, a213


def verify_phase0_files(training: dict) -> None:
    if dict(training["phase0_files"]) != PHASE0_FILES:
        raise RuntimeError("A2.17 Phase-0 file contract changed")
    for name, expected in PHASE0_FILES.items():
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"A2.17 Phase-0 file mismatch: {path}")


def load_training_model(cfg, phase0, device):
    source_cfg = deepcopy(cfg)
    source_seed = int(source_cfg["a215_training"]["schedule"]["seed"])
    source_cfg["train"]["seed"] = source_seed
    source_cfg["data"]["msrs"]["smoke"]["base_seed"] = source_seed
    return load_bound_model(source_cfg, phase0, device)


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


def _support_statistics(support, target) -> tuple[torch.Tensor, torch.Tensor]:
    intersection = (support * target).sum(dim=(1, 2, 3))
    denominator = support.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    coverage = (
        support.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))
    ).abs()
    return dice, coverage


@torch.no_grad()
def validate_smoke(model, loader, device, severity: str) -> dict:
    model.eval()
    totals = Counter()
    count = 0
    maximum_forced_residual = 0.0
    maximum_automatic_residual = 0.0
    maximum_oracle_zero_support_delta = 0.0
    for batch in tqdm(
        loader,
        desc=f"A2.17-validate-{severity}",
        ascii=True,
        leave=False,
    ):
        visible = batch["vis"].to(device, non_blocking=True)
        clean_visible = batch["vi_clean"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
        teacher = model.source_model(
            clean_visible, infrared, route_mode="canonical",
        )
        canonical = model(visible, infrared, route_mode="canonical")
        forced, forced_aux = model(
            visible, infrared, route_mode="smoke", return_aux=True,
        )
        oracle = model.source_model(
            visible,
            infrared,
            route_mode="smoke",
            support_override=mask,
        )
        automatic, automatic_aux = model(
            visible, infrared, route_mode="auto", return_aux=True,
        )
        evidence, features = predictor_inputs(model, visible, infrared)
        support = model.predictor(evidence, features)["support"]
        dice, coverage = _support_statistics(support, mask)
        batch_size = int(visible.shape[0])
        for name, value in (
            ("canonical_mae", canonical),
            ("forced_mae", forced),
            ("oracle_mae", oracle),
            ("automatic_mae", automatic),
        ):
            totals[name] += float(
                (value - teacher).abs().mean(dim=(1, 2, 3)).sum()
            )
        totals["auto_admissions"] += int(
            automatic_aux["smoke_decision"].sum()
        )
        totals["support_soft_dice"] += float(dice.sum())
        totals["support_coverage_error"] += float(coverage.sum())
        totals["support_mean"] += float(
            support.mean(dim=(1, 2, 3)).sum()
        )
        maximum_forced_residual = max(
            maximum_forced_residual,
            float(forced_aux["bounded_residual"].abs().max()),
        )
        maximum_automatic_residual = max(
            maximum_automatic_residual,
            float((automatic - canonical).abs().max()),
        )
        outside = mask <= 0.0
        if bool(outside.any()):
            maximum_oracle_zero_support_delta = max(
                maximum_oracle_zero_support_delta,
                float((oracle - canonical)[outside].abs().max()),
            )
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.17 {severity} validation count changed: {count}")
    result = {name: value / count for name, value in totals.items()}
    canonical_mae = max(result["canonical_mae"], 1.0e-12)
    for route in ("forced", "oracle", "automatic"):
        result[f"{route}_recovery"] = (
            result["canonical_mae"] - result[f"{route}_mae"]
        ) / canonical_mae
    result["auto_admissions"] = int(totals["auto_admissions"])
    result["auto_admission_rate"] = result["auto_admissions"] / count
    result["maximum_forced_residual"] = maximum_forced_residual
    result["maximum_automatic_residual"] = maximum_automatic_residual
    result["maximum_oracle_zero_support_delta"] = (
        maximum_oracle_zero_support_delta
    )
    return {"severity": severity, **result}


@torch.no_grad()
def validate_clean(model, loader, device) -> dict:
    model.eval()
    count = 0
    false_positives = 0
    maximum_error = 0.0
    support_sum = 0.0
    for batch in tqdm(
        loader, desc="A2.17-validate-clean", ascii=True, leave=False,
    ):
        visible = batch["vis"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        canonical = model(visible, infrared, route_mode="canonical")
        automatic, auxiliary = model(
            visible, infrared, route_mode="auto", return_aux=True,
        )
        evidence, features = predictor_inputs(model, visible, infrared)
        support = model.predictor(evidence, features)["support"]
        batch_size = int(visible.shape[0])
        false_positives += int(auxiliary["smoke_decision"].sum())
        maximum_error = max(
            maximum_error, float((automatic - canonical).abs().max()),
        )
        support_sum += float(support.mean(dim=(1, 2, 3)).sum())
        count += batch_size
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.17 clean validation count changed: {count}")
    return {
        "samples": count,
        "auto_false_positives": false_positives,
        "auto_identity_max_error": maximum_error,
        "support_mean": support_sum / count,
    }


def expected_admission_counts(source_report: dict) -> dict[str, int]:
    counts = {
        str(name): int(value)
        for name, value in source_report["validation"][
            "source_admission_counts"
        ].items()
    }
    if set(counts) != set(SEVERITIES):
        raise RuntimeError("A2.17 source admission report changed")
    return counts


def final_validation(model, loaders, device, training, source_report) -> dict:
    rows = [
        validate_smoke(model, loaders[severity], device, severity)
        for severity in SEVERITIES
    ]
    clean = validate_clean(model, loaders["clean"], device)
    expected = expected_admission_counts(source_report)
    observed = {row["severity"]: int(row["auto_admissions"]) for row in rows}
    if observed != expected:
        raise RuntimeError(
            ("A2.17 frozen admission counts changed", observed, expected)
        )
    automatic_recovery = float(np.mean([
        row["automatic_recovery"] for row in rows
    ]))
    forced_recovery = float(np.mean([row["forced_recovery"] for row in rows]))
    oracle_recovery = float(np.mean([row["oracle_recovery"] for row in rows]))
    smoke_soft_dice = float(np.mean([
        row["support_soft_dice"] for row in rows
    ]))
    smoke_coverage_error = float(np.mean([
        row["support_coverage_error"] for row in rows
    ]))
    maximum_residual = max(
        max(row["maximum_forced_residual"] for row in rows),
        max(row["maximum_automatic_residual"] for row in rows),
    )
    maximum_zero_support_delta = max(
        row["maximum_oracle_zero_support_delta"] for row in rows
    )
    source_oracle = float(
        source_report["validation"]["oracle_macro_recovery"]
    )
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
        "oracle_zero_support": maximum_zero_support_delta == 0.0,
        "source_admission_binding": observed == expected,
        "oracle_reference_binding": abs(oracle_recovery - source_oracle) <= 1.0e-10,
        "smoke_support_soft_dice": smoke_soft_dice
        >= float(acceptance["smoke_macro_soft_dice_min"]),
        "smoke_support_coverage": smoke_coverage_error
        <= float(acceptance["smoke_macro_coverage_error_max"]),
        "clean_support_suppression": float(clean["support_mean"])
        <= float(acceptance["clean_support_mean_max"]),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "smoke_by_severity": rows,
        "clean": clean,
        "automatic_macro_recovery": automatic_recovery,
        "forced_macro_recovery": forced_recovery,
        "oracle_macro_recovery": oracle_recovery,
        "smoke_macro_soft_dice": smoke_soft_dice,
        "smoke_macro_coverage_error": smoke_coverage_error,
        "maximum_residual": maximum_residual,
        "maximum_oracle_zero_support_delta": maximum_zero_support_delta,
        "source_admission_counts": observed,
        "internal_gates": gates,
        "rank_family_evaluation_pending": bool(gates["all_pass"]),
    }


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    history,
    config_sha,
    source_sha,
    report_sha,
    geometry_sha,
    rng_state=None,
):
    return {
        "protocol": PROTOCOL,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "training_seed": 20260918,
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "a215_final_checkpoint_sha256": source_sha,
        "a215_training_report_sha256": report_sha,
        "a216_geometry_report_sha256": geometry_sha,
        "phase0_files": PHASE0_FILES,
        "predictor_state_dict": cpu_state(model.predictor),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "random_state": random_state() if rng_state is None else rng_state,
        "official_test_loaded": False,
    }


def resume_checkpoint(
    path,
    model,
    optimizer,
    config_sha,
    source_sha,
    report_sha,
    geometry_sha,
):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol"] != PROTOCOL:
        raise RuntimeError("A2.17 resume protocol mismatch")
    if (
        checkpoint["config_sha256"] != config_sha
        or checkpoint["a215_final_checkpoint_sha256"] != source_sha
        or checkpoint["a215_training_report_sha256"] != report_sha
        or checkpoint["a216_geometry_report_sha256"] != geometry_sha
        or checkpoint["phase0_files"] != PHASE0_FILES
    ):
        raise RuntimeError("A2.17 resume evidence binding mismatch")
    if (
        int(checkpoint["training_seed"]) != 20260918
        or int(checkpoint["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS
    ):
        raise RuntimeError("A2.17 resume schedule mismatch")
    if checkpoint["official_test_loaded"] is not False:
        raise RuntimeError("A2.17 resume accessed official data")
    result = model.predictor.load_state_dict(
        checkpoint["predictor_state_dict"], strict=True,
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("A2.17 predictor resume state mismatch")
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
        raise RuntimeError("A2.17 resume epoch/history mismatch")
    if any(
        int(record.get("epoch", -1)) != index
        or record.get("stage") != "mask_supervision"
        or int(record.get("trained_samples", -1)) != EXPECTED_TRAINED_ITEMS
        for index, record in enumerate(history)
    ):
        raise RuntimeError("A2.17 resume history binding mismatch")
    return start_epoch, history


def main() -> None:
    from utils.config import load_config

    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    cfg = load_config(config_path)
    phase0, training, a211, a213 = validate_config(cfg)
    verify_phase0_files(training)
    device, device_index = initialize_cuda(args.device)
    seed_everything(int(training["schedule"]["seed"]))
    model, trainable, a211, a213 = load_training_model(cfg, phase0, device)
    frozen_before = cpu_state(model.source_model)
    source = phase0["source"]
    source_sha = file_sha256(Path(source["a215_final_checkpoint"]))
    report_sha = file_sha256(Path(source["a215_training_report"]))
    geometry_sha = file_sha256(Path(source["a216_geometry_report"]))
    source_report = json.loads(
        Path(source["a215_training_report"]).read_text(encoding="utf-8")
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
            raise FileNotFoundError("A2.17 resume checkpoint is absent")
        if complete_path.exists():
            raise FileExistsError("A2.17 training is already complete")
        start_epoch, history = resume_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config_sha,
            source_sha,
            report_sha,
            geometry_sha,
        )
    else:
        if output.exists():
            raise FileExistsError(f"REFUSED: A2.17 output exists: {output}")
        output.mkdir(parents=True)
        (output / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8",
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
                source_sha,
                report_sha,
                geometry_sha,
            ),
            checkpoint_path,
        )

    if maximum_state_update(frozen_before, model.source_model) != 0.0:
        raise RuntimeError("A2.17 source changed during initialization")
    print("config:", config_path)
    print("config_sha256:", config_sha)
    print("a215_final_checkpoint_sha256:", source_sha)
    print("a215_training_report_sha256:", report_sha)
    print("a216_geometry_report_sha256:", geometry_sha)
    print("start_epoch:", start_epoch)
    print("trained_items_per_epoch:", EXPECTED_TRAINED_ITEMS)
    print("trainable_parameters:", sum(value.numel() for value in trainable))
    print("official_test_loaded: False")

    started = time.time()
    for epoch in range(start_epoch, int(training["schedule"]["epochs"])):
        epoch_started = time.time()
        metrics = train_epoch(
            model,
            make_a217_train_loader(cfg, a211, training, epoch),
            optimizer,
            device,
            training,
            epoch,
        )
        frozen_update = maximum_state_update(frozen_before, model.source_model)
        if frozen_update != 0.0:
            raise RuntimeError(f"A2.17 frozen source changed by {frozen_update}")
        record = {
            "epoch": epoch,
            "stage": "mask_supervision",
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
                source_sha,
                report_sha,
                geometry_sha,
            ),
            checkpoint_path,
        )
        print("epoch_summary:", json.dumps(record, sort_keys=True))
        print("checkpoint_saved:", checkpoint_path, file_sha256(checkpoint_path))

    if len(history) != 5:
        raise RuntimeError(f"A2.17 history length is not five: {len(history)}")
    validation = final_validation(
        model,
        build_a213_validation_loaders(cfg, a211, a213),
        device,
        training,
        source_report,
    )
    frozen_update = maximum_state_update(frozen_before, model.source_model)
    if frozen_update != 0.0:
        raise RuntimeError("A2.17 source changed during final validation")

    final_path = output / "final_epoch4.pth"
    final_payload = {
        "protocol": PROTOCOL,
        "epoch": 4,
        "stage": "mask_supervision",
        "training_seed": int(training["schedule"]["seed"]),
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "a215_final_checkpoint_sha256": source_sha,
        "a215_training_report_sha256": report_sha,
        "a216_geometry_report_sha256": geometry_sha,
        "phase0_files": PHASE0_FILES,
        "predictor_state_dict": cpu_state(model.predictor),
        "validation": validation,
        "official_test_loaded": False,
        "rank_family_evaluation_pending": validation[
            "rank_family_evaluation_pending"
        ],
        "eligible_candidate": None,
    }
    atomic_save(final_payload, final_path)
    save_csv(output / "history.csv", history)
    save_csv(
        output / "final_validation_by_severity.csv",
        validation["smoke_by_severity"],
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
        "a215_final_checkpoint_sha256": source_sha,
        "a215_training_report_sha256": report_sha,
        "a216_geometry_report_sha256": geometry_sha,
        "config_sha256": config_sha,
        "validation": validation,
        "frozen_state_max_update": frozen_update,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": validation[
            "rank_family_evaluation_pending"
        ],
        "official_test_loaded": False,
        "seconds": time.time() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device_index) / 1024**3
        ),
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8",
    )
    if not math.isfinite(completion["peak_cuda_memory_gib"]):
        raise RuntimeError("A2.17 peak CUDA memory is non-finite")
    print("final_validation:", json.dumps(validation, sort_keys=True))
    print("final_checkpoint:", final_path.resolve())
    print("final_checkpoint_sha256:", completion["final_checkpoint_sha256"])
    print("frozen_state_max_update:", frozen_update)
    print("eligible_candidate: None")
    print(
        "rank_family_evaluation_pending:",
        validation["rank_family_evaluation_pending"],
    )
    print("official_test_loaded: False")
    print("peak_cuda_memory_gib:", round(completion["peak_cuda_memory_gib"], 3))
    print("A217_MASK_SUPERVISED_SUPPORT_FIXED_5EP_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
