"""Machine-checkable locks for A2.17 Phase-0."""

from __future__ import annotations


PROTOCOL = "A217_MASK_SUPERVISED_MULTISCALE_SUPPORT_PREDICTOR_PHASE0"
SOURCE = {
    "a215_final_checkpoint": (
        "./experiments/tifdfusion_a215_source_preserving_refiner_5ep/"
        "final_epoch4.pth"
    ),
    "a215_final_checkpoint_sha256": (
        "48278cf7c7ea1fecc6c5a36a0462b3ac53f950f7be854e07c0463cd7983e7893"
    ),
    "a215_training_report": (
        "./experiments/tifdfusion_a215_source_preserving_refiner_5ep/"
        "training_complete.json"
    ),
    "a215_training_report_sha256": (
        "b8266fc0cdc57325b784234b05e055f09b1133c34cbc97ea586d191701e4ab07"
    ),
    "a216_geometry_report": (
        "./results/external_baseline_comparison_20260901/"
        "a216_local_support_geometry/geometry_report.json"
    ),
    "a216_geometry_report_sha256": (
        "60b1ccb2ac1953598dc809b0b64efdc6152ab23ca24ac87b027e9259c8f7b2d9"
    ),
}
ARCHITECTURE = {
    "router_feature_channels": [16, 24, 32],
    "lateral_channels": 8,
    "evidence_channels": 7,
    "fusion_channels": 16,
    "support_activation": "sigmoid",
    "support_roles": ["refiner_conditioning", "residual_multiplier"],
    "trainable_parameters": 5545,
    "low_residual_cap": 0.12,
    "high_residual_cap": 0.03,
    "total_residual_cap": 0.15,
    "clean_reject_exact_canonical": True,
}
LOSS_WEIGHTS = {
    "smoke_bce": 1.0,
    "soft_dice": 1.0,
    "coverage": 1.0,
    "clean_bce": 2.0,
    "total_variation": 0.1,
}
PREFLIGHT = {
    "seed": 20260917,
    "batch_size": 2,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "steps": 2,
    "clean_validation_samples": 108,
    "checkpoint_saved": False,
}
FORBIDDEN = [
    "official_msrs_361",
    "fmb",
    "m3fd",
    "formal_training",
    "multi_seed",
    "parameter_scan",
    "loss_scan",
    "threshold_scan",
    "stage_scan",
    "best_epoch_selection",
    "early_stopping",
    "canonical_training",
    "a211_training",
    "a213_training",
    "a215_refiner_training",
    "a215_rerun",
    "rank_family_evaluation",
]


def expected_protocol() -> dict:
    return {
        "protocol": PROTOCOL,
        "source": SOURCE,
        "architecture": ARCHITECTURE,
        "loss_weights": LOSS_WEIGHTS,
        "phase0": PREFLIGHT,
        "training_authorized": False,
        "eligible_candidate": None,
        "official_test_loaded": False,
        "forbidden": FORBIDDEN,
    }


def validate_a217_protocol(value: dict) -> dict:
    expected = expected_protocol()
    for name, wanted in expected.items():
        if value.get(name) != wanted:
            raise ValueError(f"A2.17 Phase-0 lock changed: {name}")
    return value
