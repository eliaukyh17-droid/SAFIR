"""Materialize deterministic SAFIR-Fuse smoke benchmarks without touching source data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.smoke import (  # noqa: E402
    MORPHOLOGY_PRESETS,
    SEVERITY_PRESETS,
    synthesize_smoke,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_dir(root: Path, split: str, subdir: str) -> Path:
    candidates = (root / split / subdir, root / subdir / split, root / subdir)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cannot find '{subdir}' for split '{split}' under {root}")


def index_by_stem(directory: Path, *, image_only: bool = True) -> dict[str, Path]:
    """Index paired files by stem.

    Visible/infrared inputs must be images.  Labels may be masks (PNG), YOLO
    annotations (TXT), or another dataset-native file type, so label indexing
    deliberately accepts every regular file.
    """
    index: dict[str, Path] = {}
    canonical: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or (image_only and path.suffix.lower() not in IMAGE_SUFFIXES):
            continue
        key = path.stem.casefold()
        if key in canonical:
            raise ValueError(
                f"ambiguous duplicate stem under {directory}: "
                f"{index[canonical[key]].name!r} and {path.name!r}"
            )
        canonical[key] = path.stem
        index[path.stem] = path
    return index


def load_manifest(path: Path | None, vi_index: dict[str, Path]) -> list[str]:
    if path is None:
        return sorted(vi_index)
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    requested = [Path(name.replace(",", " ").split()[0]).stem for name in names]
    normalized = [stem.casefold() for stem in requested]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"duplicate sample names in manifest: {path}")
    available = {stem.casefold(): stem for stem in vi_index}
    missing = [stem for stem in requested if stem.casefold() not in available]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images missing; first: {missing[0]}")
    return [available[key] for key in normalized]


def save_float16_png(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    Image.fromarray(np.round(array * 65535).astype(np.uint16)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True, choices=("MSRS", "FMB", "M3FD", "TNO", "ROADSCENE"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--vi-subdir", default="vi")
    parser.add_argument("--ir-subdir", default="ir")
    parser.add_argument("--label-subdir", default=None)
    parser.add_argument(
        "--morphology",
        choices=tuple(MORPHOLOGY_PRESETS),
        default="patchy",
        help="Spatial smoke morphology; patchy is the main SAFIR-Fuse benchmark.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--severities", nargs="+", default=list(SEVERITY_PRESETS))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for integration tests/previews.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    invalid = set(args.severities) - set(SEVERITY_PRESETS)
    if invalid:
        raise ValueError(f"unknown severities: {sorted(invalid)}")
    vi_dir = resolve_dir(args.source_root, args.split, args.vi_subdir)
    ir_dir = resolve_dir(args.source_root, args.split, args.ir_subdir)
    label_dir = resolve_dir(args.source_root, args.split, args.label_subdir) if args.label_subdir else None
    vi_index, ir_index = index_by_stem(vi_dir), index_by_stem(ir_dir)
    label_index = index_by_stem(label_dir, image_only=False) if label_dir else {}
    stems = load_manifest(args.manifest, vi_index)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be at least 1")
        stems = stems[: args.max_samples]
    missing_ir = sorted(set(stems) - set(ir_index))
    missing_label = sorted(set(stems) - set(label_index)) if label_dir else []
    if missing_ir or missing_label:
        raise FileNotFoundError(
            f"pairing failed: missing IR={len(missing_ir)}, missing labels={len(missing_label)}")
    print(
        f"{args.dataset_name}/{args.split}: {len(stems)} pairs x "
        f"{len(args.severities)} severities; morphology={args.morphology}"
    )
    if args.dry_run:
        return

    split_root = args.output_root / args.split
    if split_root.exists() and any(split_root.iterdir()):
        raise FileExistsError(f"{split_root} is non-empty; choose a new output root")

    rows: list[dict] = []
    for severity in args.severities:
        level_root = split_root / severity
        for name in ("vi", "ir", "label", "smoke_mask", "transmission"):
            if name != "label" or label_dir:
                (level_root / name).mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(stems, start=1):
            vi_path, ir_path = vi_index[stem], ir_index[stem]
            vi = pil_to_tensor(Image.open(vi_path).convert("RGB")).float() / 255.0
            # Severity is intentionally excluded: mild/medium/heavy for one
            # source image share the same spatial field and airlight.
            seed_key = (
                f"{args.dataset_name}/{args.split}/{stem}/{args.morphology}"
            ).encode("utf-8")
            seed = args.base_seed + (zlib.crc32(seed_key) & 0x7FFFFFFF)
            smoke, mask, transmission, meta = synthesize_smoke(
                vi,
                severity,
                seed,
                morphology=args.morphology,
            )
            output_name = f"{stem}.png"
            smoke_array = np.round(smoke.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(smoke_array, mode="RGB").save(level_root / "vi" / output_name)
            shutil.copy2(ir_path, level_root / "ir" / ir_path.name)
            if label_dir:
                shutil.copy2(label_index[stem], level_root / "label" / label_index[stem].name)
            save_float16_png(mask, level_root / "smoke_mask" / output_name)
            save_float16_png(transmission, level_root / "transmission" / output_name)
            row = {
                "dataset": args.dataset_name,
                "split": args.split,
                "original_name": vi_path.name,
                "stem": stem,
                "severity": severity,
                "seed": seed,
                **meta,
                "source_vi": str(vi_path.resolve()),
                "source_ir": str(ir_path.resolve()),
                "source_label": str(label_index[stem].resolve()) if label_dir else "",
                "vi_smoke": str((level_root / "vi" / output_name).resolve()),
                "ir_copy": str((level_root / "ir" / ir_path.name).resolve()),
                "label_copy": str((level_root / "label" / label_index[stem].name).resolve()) if label_dir else "",
                "smoke_mask": str((level_root / "smoke_mask" / output_name).resolve()),
                "transmission": str((level_root / "transmission" / output_name).resolve()),
            }
            rows.append(row)
            if index % 100 == 0:
                print(f"  {severity}: {index}/{len(stems)}")

    split_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = split_root / "metadata.jsonl"
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with (split_root / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} samples to {split_root}")


if __name__ == "__main__":
    main()
