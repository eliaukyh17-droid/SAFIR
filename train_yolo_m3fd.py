"""
Train YOLOv8s on M3FD fused images for downstream detection evaluation.

All fusion models use the same YOLOv8s + COCO pretrained weights + training config,
ensuring fair comparison of detection performance across different fusion methods.

Usage:
    # Train on SFMFusion MSRS retrained fused images
    python train_yolo_m3fd.py \
        --data /data/coding/IVIF/detection_experiments/datasets/SFMFusion_msrs/dataset.yaml \
        --tag SFMFusion_msrs

    # Train on v13a fused images
    python train_yolo_m3fd.py \
        --data /data/coding/IVIF/detection_experiments/datasets/v13a/dataset.yaml \
        --tag v13a

    # Train on TDFusion fused images
    python train_yolo_m3fd.py \
        --data /data/coding/IVIF/detection_experiments/datasets/TDFusion/dataset.yaml \
        --tag TDFusion
"""

import argparse
import os

# ===== 默认路径（服务器） =====
DEFAULT_MODEL = 'pretrained/yolov8s.pt'
DEFAULT_PROJECT = 'runs/m3fd_detection'


def validation_enabled(evaluate_every_epoch=False):
    """Return whether the legacy per-epoch validation path is enabled."""
    return bool(evaluate_every_epoch)


def effective_patience(patience, evaluate_every_epoch=False):
    """Disable early stopping when the held-out test split is not validation."""
    return int(patience) if evaluate_every_epoch else 0


def reported_weight_name(evaluate_every_epoch=False):
    """Use the final epoch for the formal protocol, never test-selected best."""
    return 'best.pt' if evaluate_every_epoch else 'last.pt'


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8s on M3FD fused images')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='YOLOv8 pretrained weights (COCO)')
    parser.add_argument('--tag', type=str, default='exp',
                        help='Experiment name / model tag')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
                        help='Project directory for saving results')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device')
    parser.add_argument('--workers', type=int, default=8,
                        help='Dataloader workers')
    parser.add_argument('--patience', type=int, default=20,
                        help=('Legacy validation patience; ignored in the formal '
                              'final-epoch protocol'))
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument(
        '--evaluate-every-epoch',
        action='store_true',
        help=(
            'Legacy diagnostic mode: evaluate the dataset YAML val split every '
            'epoch and report best.pt. The formal benchmark leaves this flag '
            'unset and reports last.pt after all epochs.'
        ),
    )
    args = parser.parse_args()

    # Validate
    assert os.path.isfile(args.data), f"Dataset YAML not found: {args.data}"

    print(f"{'='*60}")
    print(f"YOLOv8s Detection Training on M3FD")
    print(f"{'='*60}")
    print(f"  Model tag:  {args.tag}")
    print(f"  Data:       {args.data}")
    print(f"  Weights:    {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  ImgSz:      {args.imgsz}")
    print(f"  Device:     {args.device}")
    use_validation = validation_enabled(args.evaluate_every_epoch)
    patience = effective_patience(args.patience, args.evaluate_every_epoch)
    reported_weight = reported_weight_name(args.evaluate_every_epoch)

    print(f"  Validation: {'every epoch (legacy)' if use_validation else 'final only'}")
    print(f"  Patience:   {patience}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {args.project}/{args.tag}")
    print(f"{'='*60}\n")

    if use_validation:
        print(
            'Legacy diagnostic protocol: the YAML val split is evaluated every '
            'epoch and best.pt is reported. Do not use this mode for the formal '
            'benchmark.\n'
        )
    else:
        print(
            'Formal protocol: the held-out M3FD test/val split is not evaluated '
            'during training. Training runs for all requested epochs and last.pt '
            'is the reported checkpoint.\n'
        )

    # Load COCO-pretrained YOLOv8s
    from ultralytics import YOLO

    model = YOLO(args.model)

    # Train with standardized hyperparameters
    # All models use identical training config for fair comparison
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.tag,
        device=args.device,
        workers=args.workers,
        patience=patience,
        seed=args.seed,
        exist_ok=True,

        # Optimizer (ultralytics defaults)
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation (ultralytics defaults)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        # Save & Logging
        save=True,
        save_period=-1,       # only save best and last
        val=use_validation,
        plots=True,
        verbose=True,
    )

    # Print final results
    print(f"\n{'='*60}")
    print(f"Training Complete: {args.tag}")
    print(f"{'='*60}")
    print(f"  Results dir:  {args.project}/{args.tag}")
    print(f"  Reported weights: {args.project}/{args.tag}/weights/{reported_weight}")
    print(f"  Final weights:    {args.project}/{args.tag}/weights/last.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
