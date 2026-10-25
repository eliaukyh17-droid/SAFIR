"""Evaluate FMB fused luminance with condition-matched visible chroma.

This supplementary evaluator does not write colourized images. It constructs
Y(fused)+CbCr(visible) in memory and feeds the result to the same frozen
DeepLabV3+ evaluator used by A3.3.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from train_seg_fmb import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    fast_hist,
    per_class_iou,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROTOCOL = "A33_UNIFORM_CHROMA_FMB_SEGMENTATION_SENSITIVITY"


def index_images(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in indexed:
            raise RuntimeError(f"duplicate case-insensitive image stem: {directory}/{path.stem}")
        indexed[key] = path
    return indexed


def load_condition_visible_paths(
    manifest_path: Path,
    condition: str,
    expected_count: int = 280,
) -> dict[str, Path]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"dataset", "severity", "modality", "sample", "path"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid A3.3 input manifest schema: {manifest_path}")
    return select_condition_visible_paths(rows, condition, expected_count)


def select_condition_visible_paths(
    rows: list[dict[str, str]],
    condition: str,
    expected_count: int,
    verify_files: bool = True,
) -> dict[str, Path]:
    selected = [
        row
        for row in rows
        if row["dataset"] == "fmb"
        and row["severity"] == condition
        and row["modality"] == "vi"
    ]
    visible: dict[str, Path] = {}
    for row in selected:
        key = row["sample"].casefold()
        if key in visible:
            raise RuntimeError(f"duplicate visible manifest sample: {row['sample']}")
        path = Path(row["path"])
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)
        visible[key] = path
    if len(visible) != expected_count:
        raise RuntimeError(
            f"condition-matched visible count differs: {condition} "
            f"{len(visible)} != {expected_count}"
        )
    return visible


def merge_fused_y_visible_chroma(fused: Image.Image, visible: Image.Image) -> Image.Image:
    """Apply the established IVIF Y+CbCr colourization rule."""
    y = fused.convert("L")
    _, cb, cr = visible.convert("YCbCr").split()
    if cb.size != y.size:
        cb = cb.resize(y.size, Image.Resampling.BILINEAR)
        cr = cr.resize(y.size, Image.Resampling.BILINEAR)
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


class UniformChromaFMBSeg:
    def __init__(
        self,
        fused_dir: Path,
        label_dir: Path,
        input_manifest: Path,
        condition: str,
        size: tuple[int, int],
        expected_count: int = 280,
    ) -> None:
        import torch

        self.torch = torch
        self.size = size
        fused = index_images(fused_dir)
        labels = index_images(label_dir)
        visible = load_condition_visible_paths(input_manifest, condition, expected_count)
        expected = set(visible)
        if set(fused) != expected:
            raise RuntimeError(
                f"fused/manifest sample mismatch: missing={sorted(expected - set(fused))[:5]} "
                f"extra={sorted(set(fused) - expected)[:5]}"
            )
        missing_labels = expected - set(labels)
        if missing_labels:
            raise RuntimeError(f"missing FMB labels: {sorted(missing_labels)[:5]}")
        self.pairs = [
            (fused[key], visible[key], labels[key]) for key in sorted(expected)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        fused_path, visible_path, label_path = self.pairs[index]
        height, width = self.size
        with Image.open(fused_path) as fused, Image.open(visible_path) as visible:
            image = merge_fused_y_visible_chroma(fused, visible)
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        with Image.open(label_path) as label_image:
            label = label_image.convert("L").resize(
                (width, height), Image.Resampling.NEAREST
            )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        label_array = np.asarray(label, dtype=np.int64)
        image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
        return (
            self.torch.from_numpy(image_array.transpose(2, 0, 1)),
            self.torch.from_numpy(label_array),
        )


def append_long_csv(path: Path, method: str, condition: str, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "dataset", "split", "severity", "metric", "value"]
    append = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for metric, value in metrics.items():
            if metric != "per_class_IoU":
                writer.writerow(
                    {
                        "method": method,
                        "dataset": "fmb",
                        "split": "test",
                        "severity": condition,
                        "metric": metric,
                        "value": f"{value:.8f}",
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--condition", choices=("clean", "medium", "heavy"), required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = UniformChromaFMBSeg(
        args.fused_dir,
        args.label_dir,
        args.input_manifest,
        args.condition,
        (args.height, args.width),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=NUM_CLASSES
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for image, label in loader:
            prediction = model(image.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            hist += fast_hist(
                prediction.reshape(-1), label.numpy().reshape(-1), NUM_CLASSES
            )
    iou = per_class_iou(hist)
    frequency = hist.sum(1) / (hist.sum() + 1e-10)
    metrics = {
        "mIoU_15": float(np.nanmean(iou)),
        "mIoU_fg14": float(np.nanmean(iou[1:])),
        "aAcc": float(np.diag(hist).sum() / (hist.sum() + 1e-10)),
        "FWIoU": float((frequency * iou).sum()),
        "per_class_IoU": [float(value) for value in iou],
    }
    report = {
        "protocol": PROTOCOL,
        "method": args.method,
        "condition": args.condition,
        "samples": len(dataset),
        "colour_transform": "Y(fused)+CbCr(condition-matched-visible)",
        "training_performed": False,
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    append_long_csv(args.metrics_csv, args.method, args.condition, metrics)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
