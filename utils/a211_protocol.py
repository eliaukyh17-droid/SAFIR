"""Machine-checkable pre-registration locks for A2.11 SC-JSFE."""

from __future__ import annotations


LOCKED_SCHEDULE = {
    "seed": 20260911,
    "epochs": 5,
    "router_warmup_epochs": 1,
    "oracle_expert_epochs": 2,
    "predicted_joint_epochs": 2,
}

LOCKED_ROUTE_THRESHOLDS = {
    "smoke_probability_min": 0.95,
    "smoke_clean_margin_min": 0.80,
    "uncertainty_max": 0.20,
    "mask_coverage_min": 0.01,
}

# These are copied without relaxation from A2.10.  New A2.11 route-specific
# requirements are additive and may not replace any inherited gate.
LOCKED_ACCEPTANCE = {
    "teacher_macro_mae_reduction_min": 0.10,
    "clean_identity_mae_max": 0.0,
    "smoky_source_average_rank_max": 6.0,
    "clean_source_average_rank_max": 4.0,
    "smoky_source_top3_min": 4,
    "clean_source_top3_min": 6,
    "family_average_rank_max": 7.0,
    "auto_clean_false_positives_max": 0,
    "predicted_oracle_recovery_ratio_min": 0.75,
}

FORBIDDEN_DURING_PROBE = (
    "official_msrs_361",
    "fmb",
    "m3fd",
    "multi_seed",
    "parameter_grid",
)

LOCKED_EPOCH_STAGES = (
    "router_warmup",
    "oracle_joint",
    "oracle_joint",
    "predicted_joint",
    "predicted_joint",
)


def _exact_mapping(observed: dict, expected: dict, name: str) -> dict:
    if dict(observed) != expected:
        raise ValueError(f"A2.11 {name} lock changed")
    return dict(observed)


def validate_schedule(observed: dict) -> dict:
    return _exact_mapping(observed, LOCKED_SCHEDULE, "schedule")


def validate_route_thresholds(observed: dict) -> dict:
    return _exact_mapping(
        observed, LOCKED_ROUTE_THRESHOLDS, "route-threshold",
    )


def validate_acceptance(observed: dict) -> dict:
    return _exact_mapping(observed, LOCKED_ACCEPTANCE, "acceptance")


def validate_forbidden(observed: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    observed = tuple(str(value) for value in observed)
    if observed != FORBIDDEN_DURING_PROBE:
        raise ValueError("A2.11 forbidden-action lock changed")
    return observed


def stage_for_epoch(epoch: int) -> str:
    """Return the only allowed stage for one zero-based probe epoch."""
    epoch = int(epoch)
    if not 0 <= epoch < len(LOCKED_EPOCH_STAGES):
        raise ValueError("A2.11 epoch is outside the locked five-epoch probe")
    return LOCKED_EPOCH_STAGES[epoch]
