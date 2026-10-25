"""Immutable A2.15 Phase-0 structural and safety locks."""

from __future__ import annotations


PROTOCOL = "A215_CANONICAL_ANCHORED_SOURCE_PRESERVING_REFINER_PHASE0"
SOURCE_CHECKPOINT_SHA256 = (
    "48a325067d9425bb9133f843aa8ba6a41f13199c469c30abdcc69e1c863b66b1"
)
SOURCE_REPORT_SHA256 = (
    "17b82addb06e3fc99eeee6ea22a12939c719b3e41d305a7eca909fe155b42837"
)
A214_REPORT_SHA256 = (
    "2c97eefdc2d6ae4beba37b15a34f2d9376112d0b638e62f2a78f0dd4f0c222dc"
)
ARCHITECTURE = {
    "evidence_channels": 12,
    "width": 32,
    "dilations": [1, 2, 4],
    "normalization": "group_norm_8",
    "activation": "silu",
    "head_channels": 2,
    "head_initialization": "all_zero",
    "trainable_parameters": 7714,
    "lowpass_kernel": 7,
    "highpass_mean_kernel": 3,
    "low_residual_cap": 0.12,
    "high_residual_cap": 0.03,
    "total_residual_cap": 0.15,
    "full_decoder_trainable": False,
    "clean_hard_bypass": True,
    "rejected_hard_bypass": True,
}
LOSS_WEIGHTS = {
    "teacher_smoke_region": 1.0,
    "teacher_low_frequency": 1.0,
    "canonical_clear_region": 2.0,
    "source_signed_gradient": 2.0,
    "unsupported_edge": 2.0,
    "canonical_gradient_floor": 2.0,
    "local_variance_floor": 1.0,
    "canonical_correlation": 0.5,
    "residual_budget": 0.25,
    "residual_high_frequency": 1.0,
}
ACCEPTANCE = {
    "teacher_macro_mae_reduction_min": 0.10,
    "clean_identity_mae_max": 0.0,
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
    "a213_rerun",
    "a214_rerun",
    "canonical_training",
    "a211_expert_training",
    "a213_router_training",
    "phase0_checkpoint",
]


def validate_a215_protocol(value: dict) -> dict:
    expected = {
        "protocol": PROTOCOL,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "a214_report_sha256": A214_REPORT_SHA256,
        "a214_required_conclusion": "proven_infeasible",
        "architecture": ARCHITECTURE,
        "loss_weights": LOSS_WEIGHTS,
        "acceptance": ACCEPTANCE,
        "phase0_seed": 20260915,
        "phase0_batch_size": 2,
        "phase0_optimizer_steps": 2,
        "phase0_learning_rate": 1.0e-4,
        "phase0_weight_decay": 0.0,
        "forbidden": FORBIDDEN,
        "training_authorized": False,
        "checkpoint_saved": False,
        "official_test_loaded": False,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"A2.15 Phase-0 lock changed: {key}")
    return value
