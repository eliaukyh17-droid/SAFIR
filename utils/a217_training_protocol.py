"""Machine-checkable locks for the single A2.17 five-epoch run."""

from __future__ import annotations


PROTOCOL = "A217_MASK_SUPERVISED_MULTISCALE_SUPPORT_PREDICTOR_FIXED_5EP"
PHASE0_FILES = {
    "A217_MASK_SUPERVISED_MULTISCALE_SUPPORT_PREDICTOR_PREREG.txt": (
        "7693d69d7d4a4ddc2c63ad602a0b03c229ca8efc1a9bb19479e97d6a111178bc"
    ),
    "models/a217_mask_supervised_support.py": (
        "4ff923df82e8d8c8aaa3ee15cb9ed5f383c1d486c55de2ce01b14a979229e68f"
    ),
    "utils/a217_protocol.py": (
        "78ea4ce59f83c8c1c8257dff5a51e278339fc101c60c5049017d9907724489b0"
    ),
    "configs/tifdfusion_a217_mask_supervised_support_phase0.yaml": (
        "d8951c8786b7e73e3ac6687953023d37455888cfc7926e843e6ddc68c3ae02b4"
    ),
    "tools/preflight_a217_mask_supervised_support.py": (
        "eac75c04f2398a9bd4135d9816688d750245f01aa4041ba004b5b85a0381ad8b"
    ),
}
SCHEDULE = {
    "seed": 20260918,
    "epochs": 5,
    "batch_size": 2,
    "trained_items_per_epoch": 974,
    "learning_rate": 1.0e-4,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
}
ACCEPTANCE = {
    "teacher_macro_mae_reduction_min": 0.10,
    "forced_teacher_macro_mae_reduction_min": 0.10,
    "clean_identity_mae_max": 0.0,
    "auto_clean_false_positives_max": 0,
    "total_residual_cap": 0.15,
    "smoke_macro_soft_dice_min": 0.50,
    "smoke_macro_coverage_error_max": 0.10,
    "clean_support_mean_max": 0.10,
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
    "support_threshold_scan",
    "stage_scan",
    "best_epoch_selection",
    "early_stopping",
    "canonical_training",
    "a211_expert_training",
    "a213_router_training",
    "a215_refiner_training",
    "a215_rerun",
]


def validate_a217_training_protocol(value: dict) -> dict:
    expected = {
        "protocol": PROTOCOL,
        "phase0_files": PHASE0_FILES,
        "schedule": SCHEDULE,
        "loss_weights": {
            "smoke_bce": 1.0,
            "soft_dice": 1.0,
            "coverage": 1.0,
            "clean_bce": 2.0,
            "total_variation": 0.1,
        },
        "acceptance": ACCEPTANCE,
        "validation_smoke_seed": 2520260906,
        "train_manifest_samples": 975,
        "validation_manifest_samples": 108,
        "trainable_parameters": 5545,
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
            raise ValueError(f"A2.17 fixed-training lock changed: {name}")
    return value
