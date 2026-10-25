"""Evaluate a fixed FMB segmentation model on saved fused images.

This is deliberately external to the fusion model: it loads only fused RGB
images, FMB labels, and a trained DeepLabV3+ checkpoint.  Consequently no
downstream gradient can update SAFIR-Fuse.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_seg_fmb import FMBSeg, NUM_CLASSES, fast_hist, per_class_iou


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fused FMB images with a fixed DeepLabV3+")
    p.add_argument("--fused_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--checkpoint", required=True, help="Fixed final.pth from train_seg_fmb.py")
    p.add_argument("--method", default="SAFIR-Fuse")
    p.add_argument("--severity", choices=["clean", "mild", "medium", "heavy"], default="clean")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--output_json", default=None)
    p.add_argument("--metrics_csv", default=None,
                   help="Append long-form rows compatible with summarize_smoke_results.py")
    return p.parse_args()


def append_long_csv(path, method, severity, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "dataset", "split", "severity", "metric", "value"]
    append = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for metric, value in metrics.items():
            if metric == "per_class_IoU":
                continue
            writer.writerow({"method": method, "dataset": "fmb", "split": "test",
                             "severity": severity, "metric": metric, "value": f"{value:.8f}"})


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = FMBSeg(args.fused_dir, args.label_dir, (args.height, args.width), train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=torch.cuda.is_available())

    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                              in_channels=3, classes=NUM_CLASSES)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    # Accept DataParallel/DistributedDataParallel checkpoints without
    # weakening strict shape validation.
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for image, label in loader:
            pred = model(image.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            hist += fast_hist(pred.reshape(-1), label.numpy().reshape(-1), NUM_CLASSES)

    iou = per_class_iou(hist)
    freq = hist.sum(1) / (hist.sum() + 1e-10)
    metrics = {
        "mIoU_15": float(np.nanmean(iou)),
        "mIoU_fg14": float(np.nanmean(iou[1:])),
        "aAcc": float(np.diag(hist).sum() / (hist.sum() + 1e-10)),
        "FWIoU": float((freq * iou).sum()),
        "per_class_IoU": [float(x) for x in iou],
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.metrics_csv:
        append_long_csv(args.metrics_csv, args.method, args.severity, metrics)


if __name__ == "__main__":
    main()
