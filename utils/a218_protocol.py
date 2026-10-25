"""Machine-checkable A2.18 read-only decomposition contract."""

from __future__ import annotations


PROTOCOL = "A218_READ_ONLY_ROUTE_SUPPORT_DECOMPOSITION"
SEVERITIES = ("mild", "medium", "heavy")
FACTORIAL_CELLS = (
    "predicted_conditioning_predicted_multiplier",
    "predicted_conditioning_oracle_multiplier",
    "oracle_conditioning_predicted_multiplier",
    "oracle_conditioning_oracle_multiplier",
)
INPUTS = {
    "a217_final_checkpoint": (
        "./experiments/tifdfusion_a217_mask_supervised_support_5ep/final_epoch4.pth"
    ),
    "a217_final_checkpoint_sha256": (
        "57e5a420a55d06db1d5c2565dff01ce7c11ca4fcf8f11122577a087fcce36311"
    ),
    "a217_training_report": (
        "./experiments/tifdfusion_a217_mask_supervised_support_5ep/"
        "training_complete.json"
    ),
    "a217_training_report_sha256": (
        "e916000a463e1758839d4d0d248d830c86d6ccc8dfad048250df417a3dbb2fad"
    ),
    "a215_final_checkpoint_sha256": (
        "48278cf7c7ea1fecc6c5a36a0462b3ac53f950f7be854e07c0463cd7983e7893"
    ),
    "a216_geometry_report_sha256": (
        "60b1ccb2ac1953598dc809b0b64efdc6152ab23ca24ac87b027e9259c8f7b2d9"
    ),
}
OUTPUTS = {
    "root": (
        "./results/external_baseline_comparison_20260901/"
        "a218_route_support_decomposition"
    ),
    "smoke_per_image": "smoke_per_image.csv",
    "by_severity": "by_severity.csv",
    "clean_per_image": "clean_support_per_image.csv",
    "report": "decomposition_report.json",
}
FORBIDDEN = (
    "official_msrs_361",
    "fmb",
    "m3fd",
    "training",
    "optimizer",
    "backward",
    "parameter_update",
    "buffer_update",
    "multi_seed",
    "threshold_scan",
    "parameter_scan",
    "loss_scan",
    "stage_scan",
    "image_generation",
    "rank_family_evaluation",
    "candidate_selection",
    "a215_rerun",
    "a216_rerun",
    "a217_rerun",
)


def expected_protocol() -> dict:
    return {
        "protocol": PROTOCOL,
        "inputs": INPUTS,
        "validation_smoke_seed": 2520260906,
        "severities": list(SEVERITIES),
        "samples_per_severity": 108,
        "expected_smoke_samples": 324,
        "expected_clean_samples": 108,
        "factorial_cells": list(FACTORIAL_CELLS),
        "reproduction_tolerance": 1.0e-7,
        "threshold_free_geometry": True,
        "fixed_route_only": True,
        "oracle_only_counterfactuals": True,
        "deployable": False,
        "eligible_candidate": None,
        "training_performed": False,
        "images_generated": False,
        "official_test_loaded": False,
        "outputs": OUTPUTS,
        "forbidden": list(FORBIDDEN),
    }


def validate_a218_protocol(value: dict) -> dict:
    for name, wanted in expected_protocol().items():
        if value.get(name) != wanted:
            raise ValueError(f"A2.18 read-only lock changed: {name}")
    return value
