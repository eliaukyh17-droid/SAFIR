"""Evaluate a fixed YOLOv8 detector on clean or smoke fused M3FD images.

The detector consumes a prepared YOLO dataset and never invokes the fusion
network, preserving the no-return-gradient downstream protocol.
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a fixed YOLOv8 detector on M3FD")
    p.add_argument("--data", required=True, help="dataset.yaml from prepare_m3fd_detection.py")
    p.add_argument(
        "--weights",
        required=True,
        help="Fixed detector checkpoint; formal protocol uses final-epoch last.pt",
    )
    p.add_argument("--method", default="SAFIR-Fuse")
    p.add_argument("--severity", choices=["clean", "mild", "medium", "heavy"], default="clean")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="./detection_experiments/eval")
    p.add_argument("--name", default="safir_fuse")
    p.add_argument("--output_json", default=None)
    p.add_argument("--metrics_csv", default=None)
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
            writer.writerow({"method": method, "dataset": "m3fd", "split": "test",
                             "severity": severity, "metric": metric, "value": f"{value:.8f}"})


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model.val(data=args.data, split="val", imgsz=args.imgsz, batch=args.batch,
                       device=args.device, workers=args.workers, project=args.project,
                       name=args.name, exist_ok=True, plots=True, verbose=True)
    metrics = {
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
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
