"""Train the single pre-registered A2.13 statistical route candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.a213_multiscale_standardized_router import (
    freeze_except_score_head,
    install_a213_router,
    multiscale_statistical_features,
    paired_binary_ranking_loss,
)
from train_a211_scjsfe_probe import (
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_VALIDATION_SAMPLES,
    SEVERITIES,
    build_composite,
    make_train_loader,
    make_validation_loader,
    seed_everything,
    validate_clean,
    validate_smoke,
)
from train_a212_router_head_calibration import (
    atomic_save,
    cpu_state,
    file_sha256,
    initialize_cuda,
    random_state,
    restore_random_state,
    roc_auc,
    save_csv,
)
from utils.a213_protocol import (
    validate_acceptance,
    validate_features,
    validate_forbidden,
    validate_loss,
    validate_route_thresholds,
    validate_schedule,
    validate_source,
    validate_trainable,
)


PROTOCOL = "A213_FROZEN_MULTISCALE_STANDARDIZED_PAIRED_ROUTER_FIXED_5EP"
STAGE = "standardized_score_head_only"
TRAINING_SEED = 20260913
STANDARDIZER_EPOCH = 0
EXPECTED_TRAINED_ITEMS = 968
EXPECTED_STANDARDIZER_VECTORS = 1936


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tifdfusion_a213_multiscale_standardized_router_5ep.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_config(cfg: dict) -> tuple[dict, dict]:
    a211 = cfg["a211_scjsfe"]
    a213 = cfg["a213_router"]
    validate_source(a213["source"])
    validate_features(a213["features"])
    validate_trainable(a213["trainable"])
    validate_schedule(a213["schedule"])
    validate_loss(a213["loss"])
    validate_route_thresholds(a213["route_thresholds"])
    validate_acceptance(a213["acceptance"])
    validate_forbidden(a213["forbidden"])
    if dict(a213["route_thresholds"]) != dict(a211["route_thresholds"]):
        raise RuntimeError("A2.13 changed an A2.11 route threshold")
    if int(a213["validation_smoke_seed"]) != 2520260906:
        raise RuntimeError("A2.13 validation smoke seed changed")
    if bool(a213["official_test_loaded"]):
        raise RuntimeError("A2.13 official_test_loaded must remain false")
    return a211, a213


def load_source_model(cfg, a211, a213, device):
    source_path = Path(a213["source_checkpoint"])
    source_sha = file_sha256(source_path)
    if source_sha != str(a213["source"]["checkpoint_sha256"]):
        raise RuntimeError("A2.13 source checkpoint SHA256 mismatch")
    source = torch.load(source_path, map_location="cpu")
    if source["protocol"] != a213["source"]["protocol"]:
        raise RuntimeError("A2.13 source protocol mismatch")
    if int(source["epoch"]) != int(a213["source"]["epoch"]):
        raise RuntimeError("A2.13 source epoch mismatch")
    if source["config_sha256"] != a213["source"]["config_sha256"]:
        raise RuntimeError("A2.13 source config mismatch")
    if source["official_test_loaded"] is not False:
        raise RuntimeError("A2.13 source accessed official data")

    model, _, baseline_sha = build_composite(cfg, a211, device)
    router_result = model.router.load_state_dict(source["router_state_dict"], strict=True)
    expert_result = model.smoke_expert.load_state_dict(source["expert_state_dict"], strict=True)
    if (
        router_result.missing_keys or router_result.unexpected_keys
        or expert_result.missing_keys or expert_result.unexpected_keys
    ):
        raise RuntimeError("A2.13 source state does not load strictly")
    wrapper = install_a213_router(model)
    trainable = freeze_except_score_head(model)
    if wrapper.feature_dimension != 316:
        raise RuntimeError("A2.13 statistical feature dimension changed")
    if sum(parameter.numel() for parameter in trainable) != 317:
        raise RuntimeError("A2.13 trainable parameter count changed")
    return model, trainable, source_sha, baseline_sha


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


def frozen_source_state(model) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "base": cpu_state(model.base_model),
        "router": cpu_state(model.router.source_router),
        "expert": cpu_state(model.smoke_expert),
    }


def maximum_module_update(before: dict[str, torch.Tensor], module) -> float:
    after = {name: value.detach().cpu() for name, value in module.state_dict().items()}
    if before.keys() != after.keys():
        raise RuntimeError("A2.13 frozen source keys changed")
    return max(
        float((after[name].float() - before[name].float()).abs().max())
        for name in before
    )


def frozen_maximum_update(before: dict, model) -> float:
    return max(
        maximum_module_update(before["base"], model.base_model),
        maximum_module_update(before["router"], model.router.source_router),
        maximum_module_update(before["expert"], model.smoke_expert),
    )


def assert_standardizer_unchanged(before: dict, router) -> None:
    after = standardizer_state(router)
    if before.keys() != after.keys() or any(
        not torch.equal(before[name], after[name]) for name in before
    ):
        raise RuntimeError("A2.13 standardizer changed after materialization")


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


def build_a213_validation_loaders(cfg: dict, a211: dict, a213: dict) -> dict:
    base = dict(cfg["data"]["msrs"]["validation_smoke"])
    base_seed = int(a213["validation_smoke_seed"])
    loaders = {}
    for index, severity in enumerate(SEVERITIES):
        smoke = dict(base)
        smoke.update(
            enabled=True,
            probability=1.0,
            severities=[severity],
            deterministic=True,
            base_seed=base_seed + 100000 * index,
        )
        loaders[severity] = make_validation_loader(cfg, a211, smoke)
    clean = dict(base)
    clean.update(enabled=False, probability=0.0, deterministic=True, base_seed=base_seed)
    loaders["clean"] = make_validation_loader(cfg, a211, clean)
    return loaders


@torch.no_grad()
def smoke_score_auc(model, loaders, device) -> float:
    positive = []
    negative = []
    for severity in SEVERITIES:
        count = 0
        for batch in loaders[severity]:
            output = model.router(
                batch["vis"].to(device, non_blocking=True),
                batch["ir"].to(device, non_blocking=True),
            )
            positive.extend(float(value) for value in output["smoke_score"].cpu())
            count += int(output["smoke_score"].shape[0])
        if count != EXPECTED_VALIDATION_SAMPLES:
            raise RuntimeError("A2.13 smoky AUC count changed")
    count = 0
    for batch in loaders["clean"]:
        output = model.router(
            batch["vis"].to(device, non_blocking=True),
            batch["ir"].to(device, non_blocking=True),
        )
        negative.extend(float(value) for value in output["smoke_score"].cpu())
        count += int(output["smoke_score"].shape[0])
    if count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("A2.13 clean AUC count changed")
    return roc_auc(positive, negative)


def final_validation(model, loaders, device, acceptance) -> dict:
    rows = [validate_smoke(model, loaders[name], device, name) for name in SEVERITIES]
    clean = validate_clean(model, loaders["clean"], device)
    admissions = {row["severity"]: float(row["auto_admission_rate"]) for row in rows}
    automatic_recovery = float(np.mean([row["automatic_recovery"] for row in rows]))
    predicted_recovery = float(np.mean([row["predicted_recovery"] for row in rows]))
    macro_admission = float(np.mean(list(admissions.values())))
    auc = smoke_score_auc(model, loaders, device)
    gates = {
        "clean_identity": float(clean["auto_identity_max_error"]) <= float(acceptance["clean_identity_mae_max"]),
        "clean_false_positives": int(clean["auto_false_positives"]) <= int(acceptance["auto_clean_false_positives_max"]),
        "smoke_auc": auc >= float(acceptance["smoke_auc_min"]),
        "mild_auto_admission": admissions["mild"] >= float(acceptance["mild_auto_admission_min"]),
        "medium_auto_admission": admissions["medium"] >= float(acceptance["medium_auto_admission_min"]),
        "heavy_auto_admission": admissions["heavy"] >= float(acceptance["heavy_auto_admission_min"]),
        "macro_auto_admission": macro_admission >= float(acceptance["macro_auto_admission_min"]),
        "automatic_teacher_recovery": automatic_recovery >= float(acceptance["automatic_teacher_macro_mae_reduction_min"]),
        "predicted_teacher_recovery": predicted_recovery >= float(acceptance["predicted_teacher_macro_mae_reduction_min"]),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "smoke_by_severity": rows,
        "clean": clean,
        "smoke_auc": auc,
        "macro_auto_admission": macro_admission,
        "automatic_macro_recovery": automatic_recovery,
        "predicted_macro_recovery": predicted_recovery,
        "internal_route_gates": gates,
        "rank_family_evaluation_pending": bool(gates["all_pass"]),
    }


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    history,
    config_sha,
    source_sha,
    standardizer_report,
    rng_state=None,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "stage": STAGE,
        "training_seed": TRAINING_SEED,
        "standardizer_epoch": STANDARDIZER_EPOCH,
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "standardizer_state_dict": standardizer_state(model.router),
        "standardizer_report": dict(standardizer_report),
        "score_head_state_dict": cpu_state(model.router.score_head),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "random_state": random_state() if rng_state is None else rng_state,
        "official_test_loaded": False,
    }


def resume_checkpoint(path, model, optimizer, config_sha, source_sha):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol"] != PROTOCOL:
        raise RuntimeError("A2.13 resume protocol mismatch")
    if checkpoint["config_sha256"] != config_sha or checkpoint["source_sha256"] != source_sha:
        raise RuntimeError("A2.13 resume hash binding mismatch")
    if (
        checkpoint["stage"] != STAGE
        or checkpoint["training_seed"] != TRAINING_SEED
        or checkpoint["standardizer_epoch"] != STANDARDIZER_EPOCH
    ):
        raise RuntimeError("A2.13 resume schedule mismatch")
    if checkpoint["official_test_loaded"] is not False:
        raise RuntimeError("A2.13 resume accessed official data")
    standardizer = checkpoint["standardizer_state_dict"]
    model.router.set_standardizer(standardizer["mean"], standardizer["scale"])
    if not bool(standardizer["fitted"]):
        raise RuntimeError("A2.13 resume standardizer was not fitted")
    if standardizer_sha256(model.router) != checkpoint["standardizer_report"]["standardizer_sha256"]:
        raise RuntimeError("A2.13 resume standardizer hash mismatch")
    result = model.router.score_head.load_state_dict(
        checkpoint["score_head_state_dict"], strict=True,
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("A2.13 score-head resume state mismatch")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(checkpoint["random_state"])
    start_epoch = int(checkpoint["next_epoch"])
    history = list(checkpoint["history"])
    if start_epoch != len(history) or not 0 <= start_epoch <= 5:
        raise RuntimeError("A2.13 resume epoch/history mismatch")
    return start_epoch, history, dict(checkpoint["standardizer_report"])


def main() -> None:
    from utils.config import load_config

    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    cfg = load_config(config_path)
    a211, a213 = validate_config(cfg)
    device, device_index = initialize_cuda(args.device)
    seed_everything(int(a213["schedule"]["seed"]))
    model, trainable, source_sha, baseline_sha = load_source_model(
        cfg, a211, a213, device,
    )
    frozen_before = frozen_source_state(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(a213["schedule"]["learning_rate"]),
        weight_decay=float(a213["schedule"]["weight_decay"]),
    )

    output = Path(cfg["output_dir"])
    checkpoint_path = output / "last_checkpoint.pth"
    complete_path = output / "training_complete.json"
    if args.resume:
        if not output.is_dir() or not checkpoint_path.is_file():
            raise FileNotFoundError("A2.13 resume checkpoint is absent")
        if complete_path.exists():
            raise FileExistsError("A2.13 training is already complete")
        start_epoch, history, standardizer_report = resume_checkpoint(
            checkpoint_path, model, optimizer, config_sha, source_sha,
        )
    else:
        if output.exists():
            raise FileExistsError(f"REFUSED: A2.13 output exists: {output}")
        standardizer_report = fit_training_standardizer(
            model,
            make_a213_train_loader(cfg, a211, a213, STANDARDIZER_EPOCH),
            device,
        )
        if standardizer_report["paired_items"] != int(a213["schedule"]["standardizer_pairs"]):
            raise RuntimeError("A2.13 standardizer protocol count mismatch")
        output.mkdir(parents=True)
        (output / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8",
        )
        start_epoch = 0
        history = []
        atomic_save(
            checkpoint_payload(
                model, optimizer, -1, history, config_sha, source_sha,
                standardizer_report,
            ),
            checkpoint_path,
        )

    standardizer_before = standardizer_state(model.router)
    if not bool(model.router.standardizer_fitted):
        raise RuntimeError("A2.13 standardizer is not fitted")
    if frozen_maximum_update(frozen_before, model) != 0.0:
        raise RuntimeError("A2.13 source changed during standardizer setup")

    print("config:", config_path)
    print("config_sha256:", config_sha)
    print("source_sha256:", source_sha)
    print("baseline_sha256:", baseline_sha)
    print("start_epoch:", start_epoch)
    print("standardizer_report:", json.dumps(standardizer_report, sort_keys=True))
    print("trainable_parameters:", sum(item.numel() for item in trainable))
    print("official_test_loaded: False")

    started = time.time()
    for epoch in range(start_epoch, int(a213["schedule"]["epochs"])):
        loader = make_a213_train_loader(cfg, a211, a213, epoch)
        epoch_started = time.time()
        metrics = train_epoch(model, loader, optimizer, device, a213["loss"], epoch)
        frozen_update = frozen_maximum_update(frozen_before, model)
        assert_standardizer_unchanged(standardizer_before, model.router)
        if frozen_update != 0.0:
            raise RuntimeError(f"A2.13 frozen state changed by {frozen_update}")
        record = {
            "epoch": epoch,
            "stage": STAGE,
            "trained_samples": EXPECTED_TRAINED_ITEMS,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            "frozen_state_max_update": frozen_update,
            "standardizer_sha256": standardizer_report["standardizer_sha256"],
            **{f"train_{name}": value for name, value in metrics.items()},
        }
        history.append(record)
        atomic_save(
            checkpoint_payload(
                model, optimizer, epoch, history, config_sha, source_sha,
                standardizer_report,
            ),
            checkpoint_path,
        )
        print("epoch_summary:", json.dumps(record, sort_keys=True))
        print("checkpoint_saved:", checkpoint_path, file_sha256(checkpoint_path))

    if len(history) != 5:
        raise RuntimeError("A2.13 history length is not five")
    validation = final_validation(
        model,
        build_a213_validation_loaders(cfg, a211, a213),
        device,
        a213["acceptance"],
    )
    assert_standardizer_unchanged(standardizer_before, model.router)
    frozen_update = frozen_maximum_update(frozen_before, model)
    if frozen_update != 0.0:
        raise RuntimeError("A2.13 frozen state changed during final validation")

    final_path = output / "final_epoch4.pth"
    final_payload = {
        "protocol": PROTOCOL,
        "epoch": 4,
        "stage": STAGE,
        "training_seed": TRAINING_SEED,
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "standardizer_state_dict": standardizer_state(model.router),
        "standardizer_report": standardizer_report,
        "score_head_state_dict": cpu_state(model.router.score_head),
        "validation": validation,
        "official_test_loaded": False,
        "rank_family_evaluation_pending": validation["rank_family_evaluation_pending"],
    }
    atomic_save(final_payload, final_path)
    save_csv(output / "history.csv", history)
    save_csv(output / "final_validation_by_severity.csv", validation["smoke_by_severity"])
    status = (
        "training_complete_rank_evaluation_pending"
        if validation["internal_route_gates"]["all_pass"]
        else "training_complete_internal_route_gates_failed"
    )
    completion = {
        "status": status,
        "protocol": PROTOCOL,
        "epochs": 5,
        "final_epoch": 4,
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": file_sha256(final_path),
        "source_sha256": source_sha,
        "baseline_sha256": baseline_sha,
        "config_sha256": config_sha,
        "standardizer_report": standardizer_report,
        "validation": validation,
        "frozen_state_max_update": frozen_update,
        "eligible_candidate": None,
        "official_test_loaded": False,
        "seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device_index) / 1024**3,
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8",
    )
    print("final_validation:", json.dumps(validation, sort_keys=True))
    print("final_checkpoint:", final_path.resolve())
    print("final_checkpoint_sha256:", completion["final_checkpoint_sha256"])
    print("standardizer_sha256:", standardizer_report["standardizer_sha256"])
    print("frozen_state_max_update:", frozen_update)
    print("eligible_candidate: None")
    print("official_test_loaded: False")
    print("peak_cuda_memory_gib:", round(completion["peak_cuda_memory_gib"], 3))
    print("A213_FROZEN_MULTISCALE_STANDARDIZED_ROUTER_FIXED_5EP_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
