"""Frozen SAFIR-A32-Controlled paired-image inference; no metrics or training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.a32_protocol import validate_a32_protocol
from utils.a33_final_protocol import validate_a33_final_protocol
from utils.config import load_config


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def tensor_to_image(tensor):
    """Use the canonical test.py uint8 serialization exactly."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().numpy()
    if image.min() < -0.5:
        image = (np.clip(image, -1, 1) + 1) / 2.0 * 255
    else:
        image = np.clip(image, 0, 1) * 255
    image = image.astype(np.uint8)
    if image.shape[0] == 1:
        return image[0]
    if image.shape[0] == 3:
        return np.transpose(image, (1, 2, 0))
    return image


def pad_to_multiple(tensor, multiple=16):
    """Use the canonical test.py right/bottom reflection padding exactly."""
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, pad_h, pad_w


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_grayscale_pair(
    output: torch.Tensor,
    canonical: torch.Tensor,
    output_path: Path,
    canonical_path: Path,
    *,
    require_identity: bool,
) -> tuple[str, str]:
    output_image = tensor_to_image(output)
    canonical_image = tensor_to_image(canonical)
    if require_identity and not np.array_equal(output_image, canonical_image):
        raise RuntimeError("A3.3 clean encoded identity changed")
    Image.fromarray(output_image, mode="L").save(output_path)
    Image.fromarray(canonical_image, mode="L").save(canonical_path)
    return file_sha256(output_path), file_sha256(canonical_path)


def load_manifest(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid or duplicate manifest: {path}")
    return names


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError(f"duplicate image stem in {directory}: {path.stem}")
        result[key] = path
    return result


def resolve_pairs(visible_dir: Path, infrared_dir: Path, manifest: Path):
    visible = index_images(visible_dir)
    infrared = index_images(infrared_dir)
    pairs = []
    for entry in load_manifest(manifest):
        stem = Path(entry).stem
        key = stem.casefold()
        if key not in visible or key not in infrared:
            raise FileNotFoundError(f"missing pair for {entry}")
        pairs.append((stem, visible[key], infrared[key]))
    return pairs


def image_tensor(path: Path, mode: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode)).copy()
    if mode == "L":
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("msrs", "fmb", "m3fd"), required=True)
    parser.add_argument("--condition", choices=("normal", "smoke"), required=True)
    parser.add_argument("--severity", choices=("clean", "medium", "heavy"), required=True)
    parser.add_argument("--visible-dir", type=Path, required=True)
    parser.add_argument("--infrared-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    from models.a32_prompt_selected_safe_residual import (
        A32PromptSelectedSafeResidualFusion,
    )
    from tools.preflight_a32_prompt_selected_safe_residual import load_bound_source

    args = parse_args()
    if (args.severity == "clean") != (args.condition == "normal"):
        raise ValueError("clean must map to normal and smoke severities must map to smoke")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing A3.3 output: {args.output_dir}")
    if args.canonical_output_dir.exists():
        raise FileExistsError(
            f"refusing existing A3.3 canonical output: {args.canonical_output_dir}"
        )
    if args.output_dir.resolve() == args.canonical_output_dir.resolve():
        raise ValueError("A3.3 candidate and canonical output directories must differ")

    cfg = load_config(args.config)
    final_contract = validate_a33_final_protocol(cfg["a33_final"])
    a32_contract = validate_a32_protocol(cfg["a32_control"])
    dataset_contract = final_contract["datasets"][args.dataset]
    if file_sha256(args.manifest) != dataset_contract["manifest_sha256"]:
        raise RuntimeError("A3.3 dataset manifest hash changed")
    pairs = resolve_pairs(args.visible_dir, args.infrared_dir, args.manifest)
    if len(pairs) != int(dataset_contract["samples"]):
        raise RuntimeError(f"A3.3 sample count changed: {len(pairs)}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A3.3 final inference requires an explicit CUDA device")
    source, _, _ = load_bound_source(cfg, a32_contract, device)
    model = A32PromptSelectedSafeResidualFusion(source).to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A3.3 model unexpectedly has trainable parameters")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.canonical_output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    canonical_rows = []
    maximum_residual = 0.0
    clean_encoded_sha_mismatches = 0
    model.reset_expert_evaluations()
    for index, (stem, visible_path, infrared_path) in enumerate(pairs, start=1):
        visible = image_tensor(visible_path, "RGB").to(device)
        infrared = image_tensor(infrared_path, "L").to(device)
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError(f"size mismatch for {stem}")
        height, width = visible.shape[-2:]
        visible, pad_h, pad_w = pad_to_multiple(visible, 16)
        infrared, ir_pad_h, ir_pad_w = pad_to_multiple(infrared, 16)
        if (pad_h, pad_w) != (ir_pad_h, ir_pad_w):
            raise RuntimeError("VI/IR padding mismatch")
        output, auxiliary = model(
            visible,
            infrared,
            degradation_condition=args.condition,
            return_aux=True,
        )
        output = output[..., :height, :width]
        canonical = auxiliary["canonical"][..., :height, :width]
        residual = output - canonical
        residual_max = float(residual.abs().max())
        maximum_residual = max(maximum_residual, residual_max)
        if args.condition == "normal":
            if auxiliary["expert_evaluated"] or not torch.equal(output, canonical):
                raise RuntimeError("A3.3 normal hard bypass changed")
        elif not auxiliary["expert_evaluated"]:
            raise RuntimeError("A3.3 smoke expert was not evaluated")
        if residual_max > float(final_contract["boundaries"]["maximum_residual"]) + 1e-7:
            raise RuntimeError(f"A3.3 residual cap exceeded: {residual_max}")
        output_path = args.output_dir / f"{stem}.png"
        canonical_output_path = args.canonical_output_dir / f"{stem}.png"
        output_sha256, canonical_output_sha256 = save_grayscale_pair(
            output,
            canonical,
            output_path,
            canonical_output_path,
            require_identity=args.condition == "normal",
        )
        if args.condition == "normal" and output_sha256 != canonical_output_sha256:
            clean_encoded_sha_mismatches += 1
        visible_sha256 = file_sha256(visible_path)
        infrared_sha256 = file_sha256(infrared_path)
        rows.append({
            "dataset": args.dataset,
            "severity": args.severity,
            "condition": args.condition,
            "sample": stem,
            "visible_path": str(visible_path.resolve()),
            "infrared_path": str(infrared_path.resolve()),
            "output_path": str(output_path.resolve()),
            "canonical_output_path": str(canonical_output_path.resolve()),
            "width": width,
            "height": height,
            "expert_evaluated": bool(auxiliary["expert_evaluated"]),
            "residual_max": residual_max,
            "visible_sha256": visible_sha256,
            "infrared_sha256": infrared_sha256,
            "output_sha256": output_sha256,
            "canonical_output_sha256": canonical_output_sha256,
        })
        canonical_rows.append({
            "dataset": args.dataset,
            "severity": args.severity,
            "sample": stem,
            "visible_path": str(visible_path.resolve()),
            "infrared_path": str(infrared_path.resolve()),
            "output_path": str(canonical_output_path.resolve()),
            "width": width,
            "height": height,
            "visible_sha256": visible_sha256,
            "infrared_sha256": infrared_sha256,
            "output_sha256": canonical_output_sha256,
        })
        if index % 25 == 0 or index == len(pairs):
            print(f"A33_INFERENCE_PROGRESS {args.dataset} {args.severity} {index}/{len(pairs)}", flush=True)

    expected_expert = 0 if args.condition == "normal" else len(pairs)
    if model.expert_evaluations != expected_expert:
        raise RuntimeError((model.expert_evaluations, expected_expert))
    if clean_encoded_sha_mismatches:
        raise RuntimeError(
            f"A3.3 clean encoded SHA mismatches: {clean_encoded_sha_mismatches}"
        )
    manifest_path = args.output_dir / "output_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    canonical_manifest_path = args.canonical_output_dir / "output_manifest.tsv"
    with canonical_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(canonical_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(canonical_rows)
    canonical_summary = {
        "protocol": final_contract["protocol"],
        "method": "Canonical-TIFDFusion",
        "dataset": args.dataset,
        "severity": args.severity,
        "samples": len(canonical_rows),
        "trainable_parameters": 0,
        "training_performed": False,
        "metrics_computed": False,
        "output_manifest_sha256": file_sha256(canonical_manifest_path),
    }
    (args.canonical_output_dir / "inference_complete.json").write_text(
        json.dumps(canonical_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "protocol": final_contract["protocol"],
        "candidate": final_contract["candidate"],
        "dataset": args.dataset,
        "severity": args.severity,
        "condition": args.condition,
        "samples": len(rows),
        "expert_evaluations": model.expert_evaluations,
        "clean_encoded_sha_mismatches": clean_encoded_sha_mismatches,
        "maximum_residual": maximum_residual,
        "trainable_parameters": 0,
        "training_performed": False,
        "metrics_computed": False,
        "external_condition_label_disclosed": True,
        "output_manifest_sha256": file_sha256(manifest_path),
        "canonical_output_manifest_sha256": file_sha256(canonical_manifest_path),
    }
    (args.output_dir / "inference_complete.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("A33_SAFIR_A32_CONTROLLED_INFERENCE_COMPLETE")


if __name__ == "__main__":
    main()
