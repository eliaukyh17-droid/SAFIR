"""Machine-checkable locks for the single A2.15 five-epoch run."""

from __future__ import annotations


PROTOCOL = "A215_CANONICAL_ANCHORED_SOURCE_PRESERVING_REFINER_FIXED_5EP"
STAGE_SEQUENCE = [
    "oracle_support",
    "oracle_support",
    "frozen_support",
    "frozen_support",
    "frozen_support",
]
PHASE0_FILES = {
    "A215_CANONICAL_ANCHORED_SOURCE_PRESERVING_REFINER_PREREG.txt": (
        "0aecd19b082668b6649baf01371a026c55220c33b281abe176c2809b049e7266"
    ),
    "models/a215_source_preserving_refiner.py": (
        "404c3c9b81183168d72cc1d3c37b67676b2e6bd7f4f40ae3ce4382ccd14a3c56"
    ),
    "utils/a215_protocol.py": (
        "d958a7e04cc827bc6e148dce59e3ef63f63435370f162a14ccc9916a9af53aa3"
    ),
    "configs/tifdfusion_a215_source_preserving_refiner_phase0.yaml": (
        "de908d42dce30fa5462403d83bd3c3fd78aad508a1cbf682729d96cf746decf5"
    ),
}
SCHEDULE = {
    "seed": 20260916,
    "epochs": 5,
    "batch_size": 2,
    "trained_items_per_epoch": 974,
    "learning_rate": 1.0e-4,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "stage_sequence": STAGE_SEQUENCE,
}
ACCEPTANCE = {
    "teacher_macro_mae_reduction_min": 0.10,
    "forced_teacher_macro_mae_reduction_min": 0.10,
    "clean_identity_mae_max": 0.0,
    "auto_clean_false_positives_max": 0,
    "total_residual_cap": 0.15,
    "smoky_source_average_rank_max": 6.0,
    "clean_source_average_rank_max": 4.0,
    "smoky_source_top3_min": 4,
    "clean_source_top3_min": 6,
    "family_average_rank_max": 7.0,
}
FORBIDDEN = [
    "official_msrs_361",
    "fmb",
    "m3fd",
    "multi_seed",
    "parameter_grid",
    "loss_scan",
    "threshold_scan",
    "stage_scan",
    "best_epoch_selection",
    "early_stopping",
    "canonical_training",
    "a211_expert_training",
    "a213_router_training",
    "a213_rerun",
    "a214_rerun",
]


def stage_for_epoch(epoch: int) -> str:
    epoch = int(epoch)
    if not 0 <= epoch < len(STAGE_SEQUENCE):
        raise ValueError(f"A2.15 epoch outside fixed schedule: {epoch}")
    return STAGE_SEQUENCE[epoch]


def validate_a215_training_protocol(value: dict) -> dict:
    expected = {
        "protocol": PROTOCOL,
        "phase0_files": PHASE0_FILES,
        "schedule": SCHEDULE,
        "acceptance": ACCEPTANCE,
        "validation_smoke_seed": 2520260906,
        "train_manifest_samples": 975,
        "validation_manifest_samples": 108,
        "trainable_parameters": 7714,
        "final_epoch": 4,
        "candidate_selection": "final_epoch_only",
        "training_authorized": False,
        "eligible_candidate": None,
        "rank_family_evaluation_pending": True,
        "official_test_loaded": False,
        "forbidden": FORBIDDEN,
    }
    for name, wanted in expected.items():
        if value.get(name) != wanted:
            raise ValueError(f"A2.15 fixed-training lock changed: {name}")
    return value
