"""Load the released SAFIR model without historical stage checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

FINAL_SHA256 = "de83cc6408c9ab4800cf018c04b36ead50411da8894f7ba03e79f35517a2c1db"
SEGFORMER_SHA256 = "584d72ace8c17c5683c198e1a4c3eda2e1026bafc02711ac5e5a4dc097bd05fe"
YOLO_SHA256 = "268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d"
BACKBONE_ROOT = "source_model." * 5 + "base_model."
PRIOR_PREFIXES = tuple(
    BACKBONE_ROOT + "prior_extractor." + name + ".backbone."
    for name in ("seg_extractor", "det_extractor")
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(path, expected=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected and sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}; use the documented paper weights")
    return path


def build_model(*, segformer_path, yolo_path, scan_backend="auto", verify_priors=True, through="a32"):
    """Construct the original module hierarchy; only pretrained priors are read.

    ``segformer_path`` is a local directory containing config.json and the
    original pytorch_model.bin. ``yolo_path`` is a trusted local YOLOv8s.pt.
    Prior checksums are enforced by default. Disabling that check permits
    experiments with other priors, which can change the published outputs.
    """
    if scan_backend not in {"auto", "cuda", "reference"}:
        raise ValueError("scan_backend must be auto, cuda, or reference")
    segformer_path = Path(segformer_path).expanduser().resolve()
    _check_file(segformer_path / "config.json")
    if verify_priors:
        _check_file(segformer_path / "pytorch_model.bin", SEGFORMER_SHA256)
    yolo_path = _check_file(yolo_path, YOLO_SHA256 if verify_priors else None)

    from models import tifm
    if not tifm.HAS_WAVELETS:
        raise RuntimeError("pytorch_wavelets is required; approximate wavelet fallback is disabled")
    from models.tifd_fusion import TIFDFusion
    from models.encoder import SS2D
    from models.a211_safe_joint_source_expert import SafeConditionalJointSourceFusion
    from models.a213_multiscale_standardized_router import install_a213_router
    from models.a215_source_preserving_refiner import A215CanonicalAnchoredFusion
    from models.a217_mask_supervised_support import A217MaskSupervisedSupportFusion
    from models.a219_dual_support_benefit_rescue import A219DualSupportBenefitRescueFusion
    from models.a223_dual_semantic_utility import A223DualSemanticUtilityFusion
    from models.a32_prompt_selected_safe_residual import A32PromptSelectedSafeResidualFusion

    base = TIFDFusion(
        decoder_norm="in",
        seg_model_name=str(segformer_path),
        det_model_path=str(yolo_path),
    )
    for name in ("seg_extractor", "det_extractor"):
        extractor = getattr(base.prior_extractor, name)
        if not extractor.available or extractor.backbone is None:
            raise RuntimeError(f"Failed to load {name}; zero-prior fallback is disabled")
    base.set_phase(3)
    for module in base.modules():
        if isinstance(module, SS2D):
            module.scan_backend = scan_backend
    if through == "base":
        return base.eval()
    source = SafeConditionalJointSourceFusion(base)
    if through == "a211":
        return source.eval()
    install_a213_router(source)
    if through == "a213":
        return source.eval()
    source = A215CanonicalAnchoredFusion(source)
    if through == "a215":
        return source.eval()
    source = A217MaskSupervisedSupportFusion(source)
    if through == "a217":
        return source.eval()
    source = A219DualSupportBenefitRescueFusion(source)
    if through == "a219":
        return source.eval()
    source = A223DualSemanticUtilityFusion(source)
    if through == "a223":
        return source.eval()
    if through != "a32":
        raise ValueError(f"Unknown construction stage: {through}")
    model = A32PromptSelectedSafeResidualFusion(source)
    for module in model.modules():
        if isinstance(module, SS2D):
            module.scan_backend = scan_backend
    return model.eval()


def _validate_state(model, state):
    """Validate all retained keys/shapes and the exact allowed exclusion set."""
    expected = model.state_dict()
    excluded = {key for key in expected if key.startswith(PRIOR_PREFIXES)}
    retained = set(expected) - excluded
    if len(excluded) != 719:
        raise RuntimeError(
            f"Expected 719 frozen prior tensors, found {len(excluded)}; "
            "check third-party library versions and pretrained model variants"
        )
    if len(state) != 807 or len(retained) != 807:
        raise RuntimeError(f"Expected 807 retained tensors, found checkpoint={len(state)}, model={len(retained)}")
    missing = retained - state.keys()
    unexpected = state.keys() - retained
    if missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise RuntimeError(f"Tensor shape/dtype mismatch for {key}: {value.shape}/{value.dtype}")
    return excluded


def load_model(
    checkpoint, *, segformer_path, yolo_path, device="cpu", scan_backend="auto",
    verify_checkpoint=True, verify_priors=True,
):
    """Return a frozen evaluation model restored from SAFIR_final.safetensors.

    Invoke with ``model(visible_rgb, infrared_gray,
    degradation_condition='normal' or 'smoke', return_aux=True)``. Inputs are
    BCHW floats in [0,1], spatially aligned and padded to a multiple of 16.
    The returned model's ``source_model`` is the A223 smoke-adaptation model.
    """
    from safetensors.torch import load_file

    checkpoint = _check_file(checkpoint, FINAL_SHA256 if verify_checkpoint else None)
    exported = load_file(str(checkpoint), device="cpu")
    if any(not key.startswith("state_dict.") for key in exported):
        raise ValueError("Expected safir_final_state_v1 keys with the state_dict. prefix")
    state = {key[len("state_dict."):]: value for key, value in exported.items()}
    model = build_model(
        segformer_path=segformer_path, yolo_path=yolo_path,
        scan_backend=scan_backend, verify_priors=verify_priors,
    )
    excluded = _validate_state(model, state)
    # Supply prior buffers too: BatchNorm otherwise synthesizes missing counters
    # when loading a plain state dict without module-version metadata.
    prior_state = model.state_dict()
    state.update({key: prior_state[key] for key in excluded})
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    model.load_report = {
        "format": "safir_final_state_v1",
        "retained_tensors_loaded": len(state) - len(excluded),
        "prior_tensors_loaded_externally": len(excluded),
        "checkpoint_sha256_verified": bool(verify_checkpoint),
        "prior_file_sha256_verified": bool(verify_priors),
        "scan_backend": scan_backend,
        "device": str(device),
    }
    return model
