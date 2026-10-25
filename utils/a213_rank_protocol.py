"""Pure locks for A2.13 internal fixed-reference qualification."""

from __future__ import annotations


CANDIDATE_NAME = "A2.13-auto-epoch4"
FINAL_CHECKPOINT_SHA256 = (
    "48a325067d9425bb9133f843aa8ba6a41f13199c469c30abdcc69e1c863b66b1"
)
TRAINING_REPORT_SHA256 = (
    "17b82addb06e3fc99eeee6ea22a12939c719b3e41d305a7eca909fe155b42837"
)
TRAINING_CONFIG_SHA256 = (
    "debe23194824c21aa21bf6a91528787daed0369028c4b3b744dd2297e49beba8"
)
SOURCE_CHECKPOINT_SHA256 = (
    "2b5f997e14a6b5ee03853dbfb765eaa0632906fcde72054eff9cae9cdcb098a3"
)
STANDARDIZER_SHA256 = (
    "e2914ffc2dc34509e2c2a03c4df09d390232b78b6a635565671624b5f94df150"
)
DEVELOPMENT_SEED = 2520260906
EXTERNAL_METHODS = (
    "SHIP", "ContiFuse", "MaeFuse", "RPFNet", "SFMFusion",
    "DWSFusion", "ISFM", "SBNF", "UniFusion", "FNet",
)
CONDITIONS = ("clean", "mild", "medium", "heavy")
SMOKE_SEVERITIES = ("mild", "medium", "heavy")
RANK_GATES = {
    "teacher_macro_mae_reduction_min": 0.10,
    "clean_identity_mae_max": 0.0,
    "smoky_source_average_rank_max": 6.0,
    "clean_source_average_rank_max": 4.0,
    "smoky_source_top3_min": 4,
    "clean_source_top3_min": 6,
    "family_average_rank_max": 7.0,
}


def validate_rank_protocol(protocol: dict) -> dict:
    expected = {
        "candidate_name": CANDIDATE_NAME,
        "candidate_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "training_report_sha256": TRAINING_REPORT_SHA256,
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "standardizer_sha256": STANDARDIZER_SHA256,
        "development_seed": DEVELOPMENT_SEED,
        "samples_per_condition": 108,
        "conditions": list(CONDITIONS),
        "smoke_severities": list(SMOKE_SEVERITIES),
        "external_methods": list(EXTERNAL_METHODS),
        "bootstrap_resamples": 1000,
        "confidence": 0.95,
        "candidate_selected_on_this_panel": False,
        "holdout_used_for_selection": False,
        "internal_route_gates_required": True,
        "official_test_loaded": False,
        **RANK_GATES,
    }
    for name, value in expected.items():
        observed = protocol[name]
        if observed != value:
            raise ValueError(f"A2.13 rank lock changed: {name}")
    return protocol


def require_internal_route_pass(report: dict) -> dict:
    if report.get("status") != "training_complete_rank_evaluation_pending":
        raise ValueError("A2.13 training report is not rank-evaluation pending")
    if report.get("official_test_loaded") is not False:
        raise ValueError("A2.13 training report accessed official data")
    if report.get("eligible_candidate") is not None:
        raise ValueError("A2.13 was declared eligible before rank evaluation")
    validation = report.get("validation", {})
    gates = validation.get("internal_route_gates", {})
    if gates.get("all_pass") is not True or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("A2.13 internal route gates did not all pass")
    if validation.get("rank_family_evaluation_pending") is not True:
        raise ValueError("A2.13 rank/family evaluation is not pending")
    return validation


def finalize_scorecard(row: dict, internal_route_pass: bool) -> dict:
    result = dict(row)
    result["pass_internal_route"] = bool(internal_route_pass)
    result["eligible"] = bool(internal_route_pass) and all(
        bool(result[f"pass_{name}"])
        for name in ("clean_identity", "teacher", "rank", "family")
    )
    return result
