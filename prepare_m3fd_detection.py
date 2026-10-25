"""Build an auditable YOLO-format M3FD dataset from fused images.

The primary SAFIR-Fuse protocol uses a fixed 3150/1050 train/test split.
The generated ``val`` partition is the held-out M3FD test partition.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import yaml


M3FD_CLASSES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_manifest(path: Path) -> List[str]:
    """Read stems from a manifest; legacy columns after the name are ignored."""
    stems: List[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.replace(",", " ").split()[0]
        stems.append(Path(token).stem)
    duplicates = len(stems) - len(set(stems))
    if duplicates:
        raise ValueError(f"{path} contains {duplicates} duplicate sample names")
    return stems


def find_image(directory: Path, stem: str) -> Optional[Path]:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def materialize(source: Path, target: Path, mode: str, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"Unsupported link mode: {mode}")


def _choose_source(stem: str, common: Optional[Path], split_specific: Optional[Path]) -> Optional[Path]:
    if split_specific is not None:
        found = find_image(split_specific, stem)
        if found is not None:
            return found
    if common is not None:
        return find_image(common, stem)
    return None


def _build_partition(
    stems: Iterable[str],
    partition: str,
    common_fused_dir: Optional[Path],
    split_fused_dir: Optional[Path],
    label_dir: Path,
    dataset_dir: Path,
    link_mode: str,
    overwrite: bool,
    name_prefix: str = "",
) -> tuple[int, int, List[str]]:
    image_count = 0
    label_count = 0
    missing: List[str] = []
    for stem in stems:
        image_source = _choose_source(stem, common_fused_dir, split_fused_dir)
        label_source = label_dir / f"{stem}.txt"
        if image_source is None or not label_source.is_file():
            reasons = []
            if image_source is None:
                reasons.append("image")
            if not label_source.is_file():
                reasons.append("label")
            missing.append(f"{stem} ({'+'.join(reasons)})")
            continue

        target_stem = f"{name_prefix}__{stem}" if name_prefix else stem
        image_target = dataset_dir / "images" / partition / f"{target_stem}{image_source.suffix.lower()}"
        label_target = dataset_dir / "labels" / partition / f"{target_stem}.txt"
        materialize(image_source, image_target, link_mode, overwrite)
        materialize(label_source, label_target, link_mode, overwrite)
        image_count += 1
        label_count += 1
    return image_count, label_count, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed-split M3FD detection dataset from fused images"
    )
    source = parser.add_argument_group("fused image sources")
    source.add_argument(
        "--fused-dir",
        type=Path,
        help="Common directory containing both train and test fused images",
    )
    source.add_argument("--train-fused-dir", type=Path, help="Optional train-only fused directory")
    source.add_argument(
        "--extra-train-fused-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=("Additional train-only fused source with a unique prefix, e.g. "
              "mild=results/.../mild. Repeat for mixed clean+smoke training."),
    )
    source.add_argument("--test-fused-dir", type=Path, help="Optional test-only fused directory")

    parser.add_argument("--label-dir", type=Path, required=True, help="YOLO label directory")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="Dataset/model tag used as the output subdirectory")
    parser.add_argument("--output-dir", type=Path, default=Path("detection_experiments/datasets"))
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize files; copy is the portable default",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit non-3150/1050 manifests or missing image/label pairs (debug only)",
    )
    return parser.parse_args()


def parse_named_sources(values: Iterable[str]) -> List[tuple[str, Path]]:
    sources: List[tuple[str, Path]] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for --extra-train-fused-dir, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or not name.replace("-", "_").isalnum():
            raise ValueError(f"Invalid source name {name!r}; use letters, digits, '_' or '-'")
        if name.casefold() in seen or name.casefold() == "clean":
            raise ValueError(f"Duplicate/reserved mixed-training source name: {name}")
        seen.add(name.casefold())
        sources.append((name, Path(raw_path)))
    return sources


def main() -> None:
    args = parse_args()
    extra_train_sources = parse_named_sources(args.extra_train_fused_dir)
    if args.fused_dir is None and args.train_fused_dir is None and args.test_fused_dir is None:
        raise SystemExit("Provide --fused-dir or split-specific fused directories")

    paths = [args.label_dir, args.train_manifest, args.test_manifest]
    paths += [p for p in (args.fused_dir, args.train_fused_dir, args.test_fused_dir) if p]
    paths += [path for _, path in extra_train_sources]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    train_stems = parse_manifest(args.train_manifest)
    test_stems = parse_manifest(args.test_manifest)
    overlap = sorted(set(train_stems) & set(test_stems))
    if overlap:
        raise RuntimeError(
            f"M3FD train/test leakage: {len(overlap)} overlapping names; first={overlap[0]}"
        )
    if not args.allow_incomplete and (len(train_stems), len(test_stems)) != (3150, 1050):
        raise RuntimeError(
            "Primary M3FD protocol requires exactly 3150/1050 samples; "
            f"found {len(train_stems)}/{len(test_stems)}"
        )

    dataset_dir = args.output_dir / args.tag
    train_images, train_labels, train_missing = _build_partition(
        train_stems,
        "train",
        args.fused_dir,
        args.train_fused_dir,
        args.label_dir,
        dataset_dir,
        args.link_mode,
        args.overwrite,
    )
    extra_counts = []
    for name, source_dir in extra_train_sources:
        images, labels, source_missing = _build_partition(
            train_stems,
            "train",
            None,
            source_dir,
            args.label_dir,
            dataset_dir,
            args.link_mode,
            args.overwrite,
            name_prefix=name,
        )
        train_images += images
        train_labels += labels
        train_missing.extend(f"{name}:{item}" for item in source_missing)
        extra_counts.append((name, images))
    val_images, val_labels, val_missing = _build_partition(
        test_stems,
        "val",
        args.fused_dir,
        args.test_fused_dir,
        args.label_dir,
        dataset_dir,
        args.link_mode,
        args.overwrite,
    )
    missing = train_missing + val_missing
    if missing and not args.allow_incomplete:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"Missing {len(missing)} M3FD image/label pairs: {preview}")

    dataset_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(M3FD_CLASSES),
        "names": M3FD_CLASSES,
    }
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"M3FD dataset: {dataset_dir.resolve()}")
    print(f"split: train={len(train_stems)}, test/val={len(test_stems)}, overlap=0")
    print(f"materialized: train={train_images}/{train_labels}, val={val_images}/{val_labels}")
    if extra_counts:
        print("mixed train sources: clean + " + ", ".join(
            f"{name}({count})" for name, count in extra_counts))
    print(f"missing pairs: {len(missing)}")
    print(f"dataset yaml: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
