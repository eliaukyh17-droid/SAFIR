"""Frozen SAFIR/Base inference for one A3.8 dataset-condition cell; no metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.a32_protocol import validate_a32_protocol  # noqa: E402
from utils.a38_generalization_protocol import validate_a38_protocol  # noqa: E402
from utils.config import load_config  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor[0].detach().cpu().numpy()
    if image.min() < -0.5:
        image = (np.clip(image, -1, 1) + 1) * 127.5
    else:
        image = np.clip(image, 0, 1) * 255.0
    image = image.astype(np.uint8)
    return image[0] if image.shape[0] == 1 else np.transpose(image, (1, 2, 0))


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, pad_h, pad_w


def image_tensor(path: Path, mode: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode)).copy()
    if mode == "L":
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def load_manifest(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(names) != len(set(name.casefold() for name in names)):
        raise ValueError(f"invalid manifest: {path}")
    return names


def image_index(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f"duplicate image stem: {directory}/{path.stem}")
            result[key] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/tifdfusion_a38_tno_roadscene_generalization.yaml"
    )
    parser.add_argument("--dataset", choices=("tno", "roadscene"), required=True)
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
        raise ValueError("A3.8 clean must map to normal; medium/heavy must map to smoke")
    if args.output_dir.exists() or args.canonical_output_dir.exists():
        raise FileExistsError("refusing an existing A3.8 output directory")
    if args.output_dir.resolve() == args.canonical_output_dir.resolve():
        raise ValueError("SAFIR and Base output roots must differ")

    cfg = load_config(args.config)
    contract = validate_a38_protocol(cfg["a38_generalization"])
    a32_contract = validate_a32_protocol(cfg["a32_control"])
    dataset_contract = contract["datasets"][args.dataset]
    if sha256(args.manifest) != dataset_contract["manifest_sha256"]:
        raise RuntimeError("A3.8 dataset manifest hash changed")

    visible_index = image_index(args.visible_dir)
    infrared_index = image_index(args.infrared_dir)
    entries = load_manifest(args.manifest)
    if len(entries) != dataset_contract["samples"]:
        raise RuntimeError("A3.8 dataset sample count changed")
    pairs = []
    for entry in entries:
        stem = Path(entry).stem
        key = stem.casefold()
        if key not in visible_index or key not in infrared_index:
            raise FileNotFoundError(f"missing A3.8 pair: {args.dataset}/{entry}")
        pairs.append((stem, visible_index[key], infrared_index[key]))

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A3.8 formal inference requires an explicit CUDA device")
    source, _, _ = load_bound_source(cfg, a32_contract, device)
    model = A32PromptSelectedSafeResidualFusion(source).to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A3.8 model unexpectedly has trainable parameters")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.canonical_output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    maximum_residual = 0.0
    clean_sha_mismatches = 0
    model.reset_expert_evaluations()
    for index, (stem, visible_path, infrared_path) in enumerate(pairs, start=1):
        visible = image_tensor(visible_path, "RGB").to(device)
        infrared = image_tensor(infrared_path, "L").to(device)
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError(f"A3.8 source size mismatch: {args.dataset}/{stem}")
        height, width = visible.shape[-2:]
        visible, pad_h, pad_w = pad_to_multiple(visible)
        infrared, ir_pad_h, ir_pad_w = pad_to_multiple(infrared)
        if (pad_h, pad_w) != (ir_pad_h, ir_pad_w):
            raise RuntimeError("A3.8 VI/IR padding mismatch")

        output, auxiliary = model(
            visible,
            infrared,
            degradation_condition=args.condition,
            return_aux=True,
        )
        output = output[..., :height, :width]
        canonical = auxiliary["canonical"][..., :height, :width]
        residual_max = float((output - canonical).abs().max())
        maximum_residual = max(maximum_residual, residual_max)
        if args.condition == "normal":
            if auxiliary["expert_evaluated"] or not torch.equal(output, canonical):
                raise RuntimeError("A3.8 clean hard bypass changed")
        elif not auxiliary["expert_evaluated"]:
            raise RuntimeError("A3.8 smoke expert was not evaluated")
        if residual_max > contract["boundaries"]["maximum_residual"] + 1e-7:
            raise RuntimeError(f"A3.8 residual cap exceeded: {residual_max}")

        output_path = args.output_dir / f"{stem}.png"
        canonical_path = args.canonical_output_dir / f"{stem}.png"
        output_array = tensor_to_uint8(output)
        canonical_array = tensor_to_uint8(canonical)
        if args.condition == "normal" and not np.array_equal(output_array, canonical_array):
            raise RuntimeError("A3.8 clean encoded identity changed")
        Image.fromarray(output_array, mode="L").save(output_path)
        Image.fromarray(canonical_array, mode="L").save(canonical_path)
        output_sha = sha256(output_path)
        canonical_sha = sha256(canonical_path)
        if args.condition == "normal" and output_sha != canonical_sha:
            clean_sha_mismatches += 1
        rows.append({
            "dataset": args.dataset,
            "condition": args.severity,
            "sample": stem,
            "visible_path": str(visible_path.resolve()),
            "infrared_path": str(infrared_path.resolve()),
            "safir_output_path": str(output_path.resolve()),
            "base_output_path": str(canonical_path.resolve()),
            "width": width,
            "height": height,
            "expert_evaluated": bool(auxiliary["expert_evaluated"]),
            "residual_max": residual_max,
            "safir_output_sha256": output_sha,
            "base_output_sha256": canonical_sha,
        })
        if index % 10 == 0 or index == len(pairs):
            print(
                f"A38_INFERENCE_PROGRESS {args.dataset}/{args.severity} "
                f"{index}/{len(pairs)}",
                flush=True,
            )

    expected_expert = 0 if args.condition == "normal" else len(pairs)
    if model.expert_evaluations != expected_expert:
        raise RuntimeError("A3.8 expert evaluation count changed")
    if clean_sha_mismatches:
        raise RuntimeError(f"A3.8 clean encoded SHA mismatches: {clean_sha_mismatches}")

    manifest_path = args.output_dir / "output_manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": contract["protocol"],
        "status": "a38_safir_base_cell_complete",
        "dataset": args.dataset,
        "condition": args.severity,
        "samples": len(rows),
        "expert_evaluations": model.expert_evaluations,
        "maximum_residual": maximum_residual,
        "clean_encoded_sha_mismatches": clean_sha_mismatches,
        "native_output_size_verified": True,
        "output_manifest_sha256": sha256(manifest_path),
        "training_performed": False,
        "tuning_performed": False,
        "metrics_computed": False,
    }
    report_path = args.output_dir / "inference_complete.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("A38_SAFIR_BASE_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
