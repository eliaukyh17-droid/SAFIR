"""Train the single preregistered A2.15 bounded residual candidate."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.a215_source_preserving_refiner import (
    A215CanonicalAnchoredFusion,
    CanonicalAnchoredResidualRefiner,
    source_preserving_loss,
)
from tools.prepare_a213_internal_rank_panel import load_locked_candidate
from train_a211_scjsfe_probe import (
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_VALIDATION_SAMPLES,
    SEVERITIES,
    make_train_loader,
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
from utils.a215_protocol import validate_a215_protocol
from utils.a215_training_protocol import (
    PHASE0_FILES,
    PROTOCOL,
    STAGE_SEQUENCE,
    stage_for_epoch,
    validate_a215_training_protocol,
)


EXPECTED_TRAINED_ITEMS = 974


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tifdfusion_a215_source_preserving_refiner_5ep.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_config(cfg: dict) -> tuple[dict, dict, dict, dict]:
    phase0 = validate_a215_protocol(cfg["a215_refiner"])
    training = validate_a215_training_protocol(cfg["a215_training"])
    a211 = cfg["a211_scjsfe"]
    a213 = cfg["a213_router"]
    schedule = training["schedule"]
    if int(cfg["train"]["seed"]) != int(schedule["seed"]):
        raise ValueError("A2.15 top-level and schedule seeds differ")
    smoke = cfg["data"]["msrs"]["smoke"]
    if int(smoke["base_seed"]) != int(schedule["seed"]):
        raise ValueError("A2.15 smoke and schedule seeds differ")
    if not bool(smoke["enabled"]) or float(smoke["probability"]) != 1.0:
        raise ValueError("A2.15 training smoke contract changed")
    if tuple(smoke["severities"]) != SEVERITIES:
        raise ValueError("A2.15 training severities changed")
    if int(schedule["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS:
        raise ValueError("A2.15 trained-item count changed")
    if int(training["validation_smoke_seed"]) != int(a213["validation_smoke_seed"]):
        raise ValueError("A2.15 validation seed changed from A2.13")
    phase_acceptance = phase0["acceptance"]
    training_acceptance = training["acceptance"]
    for name in (
        "teacher_macro_mae_reduction_min",
        "clean_identity_mae_max",
        "smoky_source_average_rank_max",
        "clean_source_average_rank_max",
        "smoky_source_top3_min",
        "clean_source_top3_min",
        "family_average_rank_max",
    ):
        if training_acceptance[name] != phase_acceptance[name]:
            raise ValueError(f"A2.15 acceptance changed from Phase-0: {name}")
    if float(training_acceptance["total_residual_cap"]) != float(
        phase0["architecture"]["total_residual_cap"]
    ):
        raise ValueError("A2.15 residual cap changed from Phase-0")
    validate_manifests(cfg)
    if int(training["train_manifest_samples"]) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("A2.15 train manifest declaration changed")
    if int(training["validation_manifest_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("A2.15 validation manifest declaration changed")
    if bool(training["official_test_loaded"]):
        raise RuntimeError("A2.15 official_test_loaded must remain false")
    return phase0, training, a211, a213


def verify_phase0_files(training: dict) -> None:
    if dict(training["phase0_files"]) != PHASE0_FILES:
        raise RuntimeError("A2.15 Phase-0 file contract changed")
    for name, expected in PHASE0_FILES.items():
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"A2.15 Phase-0 file mismatch: {path}")


def load_a215_model(cfg, phase0, training, device):
    a214_path = Path(phase0["a214_report"])
    if (
        not a214_path.is_file()
        or file_sha256(a214_path) != phase0["a214_report_sha256"]
    ):
        raise RuntimeError("A2.15 A2.14 evidence mismatch")
    a214 = json.loads(a214_path.read_text(encoding="utf-8"))
    if not (
        a214.get("conclusion") == "proven_infeasible"
        and a214.get("solver_proved_infeasible") is True
        and a214.get("official_test_loaded") is False
    ):
        raise RuntimeError("A2.15 A2.14 conclusion changed")
    source, a211, a213, source_report = load_locked_candidate(
        cfg,
        cfg["a213_internal_rank"],
        device,
    )
    architecture = phase0["architecture"]
    refiner = CanonicalAnchoredResidualRefiner(
        evidence_channels=int(architecture["evidence_channels"]),
        width=int(architecture["width"]),
        dilations=architecture["dilations"],
        lowpass_kernel=int(architecture["lowpass_kernel"]),
        highpass_mean_kernel=int(architecture["highpass_mean_kernel"]),
        low_residual_cap=float(architecture["low_residual_cap"]),
        high_residual_cap=float(architecture["high_residual_cap"]),
    ).to(device)
    model = A215CanonicalAnchoredFusion(source, refiner).to(device)
    trainable = tuple(model.trainable_parameters())
    count = sum(parameter.numel() for parameter in trainable)
    if count != int(training["trainable_parameters"]):
        raise RuntimeError(f"A2.15 trainable parameter count changed: {count}")
    if any(parameter.requires_grad for parameter in model.source_model.parameters()):
        raise RuntimeError("A2.15 source parameter remained trainable")
    source_sha = file_sha256(Path(phase0["source_checkpoint"]))
    return model, trainable, a211, a213, source_report, source_sha


def maximum_state_update(before: dict[str, torch.Tensor], module) -> float:
    after = module.state_dict()
    if before.keys() != after.keys():
        raise RuntimeError("A2.15 frozen source keys changed")
    maximum = 0.0
    for name, original in before.items():
        difference = float(
            (after[name].detach().cpu().float() - original.float()).abs().max()
        )
        maximum = max(maximum, difference)
    return maximum


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
        desc=f"A2.15-validate-{severity}",
        ascii=True,
        leave=False,
    ):
        visible = batch["vis"].to(device, non_blocking=True)
        clean_visible = batch["vi_clean"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        mask = batch["smoke_mask"].to(device, non_blocking=True).clamp(0.0, 1.0)
        teacher = model.source_model(
            clean_visible,
            infrared,
            route_mode="canonical",
        )
        canonical = model.source_model(
            visible,
            infrared,
            route_mode="canonical",
        )
        forced, forced_aux = model(
            visible,
            infrared,
            route_mode="smoke",
            return_aux=True,
        )
        oracle = model(
            visible,
            infrared,
            route_mode="smoke",
            support_override=mask,
        )
        automatic, automatic_aux = model(
            visible,
            infrared,
            route_mode="auto",
            return_aux=True,
        )
        batch_size = int(visible.shape[0])
        for name, value in (
            ("canonical_mae", canonical),
            ("forced_mae", forced),
            ("oracle_mae", oracle),
            ("automatic_mae", automatic),
        ):
            totals[name] += float((value - teacher).abs().mean(dim=(1, 2, 3)).sum())
        totals["auto_admissions"] += int(automatic_aux["smoke_decision"].sum())
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
        raise RuntimeError(f"A2.15 {severity} validation count changed: {count}")
    result = {name: value / count for name, value in totals.items()}
    canonical_mae = max(result["canonical_mae"], 1.0e-12)
    for route in ("forced", "oracle", "automatic"):
        result[f"{route}_recovery"] = (
            result["canonical_mae"] - result[f"{route}_mae"]
        ) / canonical_mae
    result["auto_admission_rate"] = result.pop("auto_admissions")
    result["maximum_forced_residual"] = maximum_forced_residual
    result["maximum_automatic_residual"] = maximum_automatic_residual
    result["maximum_oracle_zero_support_delta"] = maximum_oracle_zero_support_delta
    return {"severity": severity, **result}


@torch.no_grad()
def validate_clean(model, loader, device) -> dict:
    model.eval()
    count = 0
    false_positives = 0
    maximum_error = 0.0
    for batch in tqdm(
        loader,
        desc="A2.15-validate-clean",
        ascii=True,
        leave=False,
    ):
        visible = batch["vis"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        canonical = model.source_model(
            visible,
            infrared,
            route_mode="canonical",
        )
        automatic, auxiliary = model(
            visible,
            infrared,
            route_mode="auto",
            return_aux=True,
        )
        false_positives += int(auxiliary["smoke_decision"].sum())
        maximum_error = max(
            maximum_error,
            float((automatic - canonical).abs().max()),
        )
        count += int(visible.shape[0])
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"A2.15 clean validation count changed: {count}")
    return {
        "samples": count,
        "auto_false_positives": false_positives,
        "auto_identity_max_error": maximum_error,
    }


def expected_admission_counts(source_report: dict) -> dict[str, int]:
    rows = source_report["validation"]["smoke_by_severity"]
    counts = {
        row["severity"]: round(
            float(row["auto_admission_rate"]) * EXPECTED_VALIDATION_SAMPLES
        )
        for row in rows
    }
    if set(counts) != set(SEVERITIES):
        raise RuntimeError("A2.15 source admission report changed")
    return counts


def final_validation(model, loaders, device, training, source_report) -> dict:
    rows = [
        validate_smoke(model, loaders[severity], device, severity)
        for severity in SEVERITIES
    ]
    clean = validate_clean(model, loaders["clean"], device)
    expected = expected_admission_counts(source_report)
    observed = {
        row["severity"]: round(
            float(row["auto_admission_rate"]) * EXPECTED_VALIDATION_SAMPLES
        )
        for row in rows
    }
    if observed != expected:
        raise RuntimeError(
            ("A2.15 frozen admission counts changed", observed, expected)
        )
    automatic_recovery = float(np.mean([row["automatic_recovery"] for row in rows]))
    forced_recovery = float(np.mean([row["forced_recovery"] for row in rows]))
    oracle_recovery = float(np.mean([row["oracle_recovery"] for row in rows]))
    maximum_residual = max(
        max(row["maximum_forced_residual"] for row in rows),
        max(row["maximum_automatic_residual"] for row in rows),
    )
    maximum_zero_support_delta = max(
        row["maximum_oracle_zero_support_delta"] for row in rows
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
    }
    gates["all_pass"] = all(gates.values())
    return {
        "smoke_by_severity": rows,
        "clean": clean,
        "automatic_macro_recovery": automatic_recovery,
        "forced_macro_recovery": forced_recovery,
        "oracle_macro_recovery": oracle_recovery,
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
    stage,
    history,
    config_sha,
    source_sha,
    a214_sha,
    rng_state=None,
):
    return {
        "protocol": PROTOCOL,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "stage": stage,
        "stage_sequence": STAGE_SEQUENCE,
        "training_seed": 20260916,
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "a214_report_sha256": a214_sha,
        "phase0_files": PHASE0_FILES,
        "refiner_state_dict": cpu_state(model.refiner),
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
    a214_sha,
):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol"] != PROTOCOL:
        raise RuntimeError("A2.15 resume protocol mismatch")
    if (
        checkpoint["config_sha256"] != config_sha
        or checkpoint["source_sha256"] != source_sha
        or checkpoint["a214_report_sha256"] != a214_sha
        or checkpoint["phase0_files"] != PHASE0_FILES
    ):
        raise RuntimeError("A2.15 resume evidence binding mismatch")
    if (
        checkpoint["stage_sequence"] != STAGE_SEQUENCE
        or int(checkpoint["training_seed"]) != 20260916
        or int(checkpoint["trained_items_per_epoch"]) != EXPECTED_TRAINED_ITEMS
    ):
        raise RuntimeError("A2.15 resume schedule mismatch")
    if checkpoint["official_test_loaded"] is not False:
        raise RuntimeError("A2.15 resume accessed official data")
    epoch = int(checkpoint["epoch"])
    expected_stage = "initialized" if epoch == -1 else stage_for_epoch(epoch)
    if checkpoint["stage"] != expected_stage:
        raise RuntimeError("A2.15 resume stage mismatch")
    result = model.refiner.load_state_dict(
        checkpoint["refiner_state_dict"],
        strict=True,
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("A2.15 refiner resume state mismatch")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(checkpoint["random_state"])
    start_epoch = int(checkpoint["next_epoch"])
    history = list(checkpoint["history"])
    if start_epoch != len(history) or not 0 <= start_epoch <= 5:
        raise RuntimeError("A2.15 resume epoch/history mismatch")
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
    model, trainable, a211, a213, source_report, source_sha = load_a215_model(
        cfg,
        phase0,
        training,
        device,
    )
    frozen_before = cpu_state(model.source_model)
    a214_sha = file_sha256(Path(phase0["a214_report"]))
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
            raise FileNotFoundError("A2.15 resume checkpoint is absent")
        if complete_path.exists():
            raise FileExistsError("A2.15 training is already complete")
        start_epoch, history = resume_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config_sha,
            source_sha,
            a214_sha,
        )
    else:
        if output.exists():
            raise FileExistsError(f"REFUSED: A2.15 output exists: {output}")
        output.mkdir(parents=True)
        (output / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        start_epoch = 0
        history = []
        atomic_save(
            checkpoint_payload(
                model,
                optimizer,
                -1,
                "initialized",
                history,
                config_sha,
                source_sha,
                a214_sha,
            ),
            checkpoint_path,
        )

    if maximum_state_update(frozen_before, model.source_model) != 0.0:
        raise RuntimeError("A2.15 source changed during initialization")
    print("config:", config_path)
    print("config_sha256:", config_sha)
    print("source_sha256:", source_sha)
    print("a214_report_sha256:", a214_sha)
    print("stage_sequence:", STAGE_SEQUENCE)
    print("start_epoch:", start_epoch)
    print("trained_items_per_epoch:", EXPECTED_TRAINED_ITEMS)
    print("trainable_parameters:", sum(p.numel() for p in trainable))
    print("official_test_loaded: False")

    started = time.time()
    for epoch in range(start_epoch, int(training["schedule"]["epochs"])):
        stage = stage_for_epoch(epoch)
        epoch_started = time.time()
        metrics = train_epoch(
            model,
            make_a215_train_loader(cfg, a211, training, epoch),
            optimizer,
            device,
            phase0,
            training,
            epoch,
        )
        frozen_update = maximum_state_update(frozen_before, model.source_model)
        if frozen_update != 0.0:
            raise RuntimeError(f"A2.15 frozen source changed by {frozen_update}")
        record = {
            "epoch": epoch,
            "stage": stage,
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
                stage,
                history,
                config_sha,
                source_sha,
                a214_sha,
            ),
            checkpoint_path,
        )
        print("epoch_summary:", json.dumps(record, sort_keys=True))
        print("checkpoint_saved:", checkpoint_path, file_sha256(checkpoint_path))

    if len(history) != 5:
        raise RuntimeError(f"A2.15 history length is not five: {len(history)}")
    validation = final_validation(
        model,
        build_a213_validation_loaders(cfg, a211, a213),
        device,
        training,
        source_report,
    )
    frozen_update = maximum_state_update(frozen_before, model.source_model)
    if frozen_update != 0.0:
        raise RuntimeError("A2.15 source changed during final validation")

    final_path = output / "final_epoch4.pth"
    final_payload = {
        "protocol": PROTOCOL,
        "epoch": 4,
        "stage": stage_for_epoch(4),
        "stage_sequence": STAGE_SEQUENCE,
        "training_seed": int(training["schedule"]["seed"]),
        "trained_items_per_epoch": EXPECTED_TRAINED_ITEMS,
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "a214_report_sha256": a214_sha,
        "phase0_files": PHASE0_FILES,
        "refiner_state_dict": cpu_state(model.refiner),
        "validation": validation,
        "official_test_loaded": False,
        "rank_family_evaluation_pending": validation["rank_family_evaluation_pending"],
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
        "source_sha256": source_sha,
        "a214_report_sha256": a214_sha,
        "config_sha256": config_sha,
        "validation": validation,
        "frozen_state_max_update": frozen_update,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": validation["rank_family_evaluation_pending"],
        "official_test_loaded": False,
        "seconds": time.time() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device_index) / 1024**3
        ),
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not math.isfinite(completion["peak_cuda_memory_gib"]):
        raise RuntimeError("A2.15 peak CUDA memory is non-finite")
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
    print("A215_SOURCE_PRESERVING_REFINER_FIXED_5EP_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
