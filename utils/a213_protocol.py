"""Machine-checkable locks for A2.13 frozen statistical paired routing."""

from __future__ import annotations


LOCKED_SOURCE = {
    "protocol": "A211_SCJSFE_INTERNAL_975_108_FIXED_5EP",
    "epoch": 4,
    "checkpoint_sha256": "2b5f997e14a6b5ee03853dbfb765eaa0632906fcde72054eff9cae9cdcb098a3",
    "config_sha256": "0cd9d1521dcc6aa4bc7a7da100d85296edf2f2520c6c9d09b46c875020118de0",
}

LOCKED_FEATURES = {
    "router_stages": [0, 1, 2],
    "router_widths": [16, 24, 32],
    "evidence_channels": 7,
    "statistics": ["mean", "std", "maximum", "top_tail_mean"],
    "top_tail_divisor": 16,
    "dimension": 316,
    "standardizer_source": "paired_training_manifest_only",
    "standard_deviation_floor": 1e-6,
    "standardized_clip": 8.0,
}

LOCKED_TRAINABLE = {
    "module": "router.score_head",
    "in_features": 316,
    "out_features": 1,
    "parameters": 317,
    "initialization": "all_zero",
    "three_way_logits": ["negative_score", "positive_score", "zero_unknown"],
}

LOCKED_SCHEDULE = {
    "seed": 20260913,
    "epochs": 5,
    "batch_size": 8,
    "standardizer_pairs": 968,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
}

LOCKED_LOSS = {
    "binary_weight": 1.0,
    "anchor_weight": 1.0,
    "pair_weight": 1.0,
    "anchor_margin": 4.0,
    "pair_margin": 8.0,
}

LOCKED_ROUTE_THRESHOLDS = {
    "smoke_probability_min": 0.95,
    "smoke_clean_margin_min": 0.80,
    "uncertainty_max": 0.20,
    "mask_coverage_min": 0.01,
}

LOCKED_ACCEPTANCE = {
    "clean_identity_mae_max": 0.0,
    "auto_clean_false_positives_max": 0,
    "smoke_auc_min": 0.90,
    "mild_auto_admission_min": 0.10,
    "medium_auto_admission_min": 0.25,
    "heavy_auto_admission_min": 0.40,
    "macro_auto_admission_min": 0.25,
    "automatic_teacher_macro_mae_reduction_min": 0.10,
    "predicted_teacher_macro_mae_reduction_min": 0.10,
    "smoky_source_average_rank_max": 6.0,
    "clean_source_average_rank_max": 4.0,
    "smoky_source_top3_min": 4,
    "clean_source_top3_min": 6,
    "family_average_rank_max": 7.0,
}

FORBIDDEN = (
    "official_msrs_361",
    "fmb",
    "m3fd",
    "multi_seed",
    "parameter_grid",
    "threshold_scan",
    "a212_validation_reuse",
    "expert_training",
    "router_backbone_training",
    "local_gate_training",
    "canonical_training",
)


def _exact(observed, expected, name):
    if observed != expected:
        raise ValueError(f"A2.13 {name} lock changed")
    return observed.copy() if isinstance(observed, dict) else tuple(observed)


def validate_source(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_SOURCE, "source")


def validate_features(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_FEATURES, "features")


def validate_trainable(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_TRAINABLE, "trainable")


def validate_schedule(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_SCHEDULE, "schedule")


def validate_loss(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_LOSS, "loss")


def validate_route_thresholds(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_ROUTE_THRESHOLDS, "route thresholds")


def validate_acceptance(observed: dict) -> dict:
    return _exact(dict(observed), LOCKED_ACCEPTANCE, "acceptance")


def validate_forbidden(observed) -> tuple[str, ...]:
    return _exact(tuple(observed), FORBIDDEN, "forbidden actions")
