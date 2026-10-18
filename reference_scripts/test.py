"""
Inference script for TIFDFusion.
Loads a trained model and generates fusion images from vis+ir pairs.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.tifd_fusion import build_model
from data.dataset import get_test_loader
from utils.config import load_config
from utils.metrics import compute_fusion_metrics
from tools.summarize_smoke_results import write_per_image_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='TIFDFusion Inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='msrs',
                        choices=['msrs', 'm3fd', 'tno', 'roadscene', 'llvip', 'fmb',
                                 'generic', 'real_smoke'])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_metrics', action='store_true', help='Compute fusion metrics')
    parser.add_argument('--method_name', type=str, default='SAFIR-Fuse',
                        help='Method name written to the long-form metrics CSV')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split name used by the loader and metrics metadata')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Optional strict sample manifest (one filename/stem per line)')
    parser.add_argument('--severity', type=str, default='clean',
                        choices=['clean', 'mild', 'medium', 'heavy'],
                        help='Condition label written to the long-form metrics CSV')
    parser.add_argument('--metrics_csv', type=str, default=None,
                        help='Optional shared long-form CSV to append to')
    parser.add_argument('--original_size', action='store_true', default=True,
                        help='Test at original image resolution (default: True)')
    parser.add_argument('--resize_test', action='store_true', default=False,
                        help='Force resize to config img_size for testing')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Max number of images to process (0=all)')
    return parser.parse_args()


def tensor_to_image(tensor):
    """Convert tensor (1,C,H,W) or (C,H,W) to numpy image [0,255] uint8."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy()
    # Auto-detect range: if min < -0.5, assume Tanh [-1,1]; else assume [0,1]
    if img.min() < -0.5:
        img = np.clip(img, -1, 1)
        img = (img + 1) / 2.0 * 255
    else:
        img = np.clip(img, 0, 1)
        img = img * 255
    img = img.astype(np.uint8)
    if img.shape[0] == 1:
        img = img[0]  # grayscale
    elif img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    return img


def pad_to_multiple(tensor, multiple=16):
    """Pad tensor H,W to be divisible by `multiple`. Returns padded tensor and pad sizes."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, pad_h, pad_w


@torch.no_grad()
def main():
    args = parse_args()

    cfg = load_config(args.config)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    model = build_model(cfg)
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (new params)")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected checkpoint keys")
    model.set_phase(3)  # Full model with all priors
    model.eval()
    metadata = ckpt if isinstance(ckpt, dict) else {}
    print(f"Loaded checkpoint: {args.checkpoint} "
          f"(phase {metadata.get('phase', '?')}, epoch {metadata.get('epoch', '?')})")

    # Data
    data_cfg = cfg['data'].get(args.dataset, {})
    default_roots = {
        'msrs': './datasets/MSRS', 'm3fd': './datasets/M3FD/M3FD_Detection',
        'tno': './datasets/tno', 'roadscene': './datasets/roadscene',
        'llvip': './datasets/LLVIP', 'fmb': './datasets/FMB',
        'generic': './datasets/generic', 'real_smoke': './datasets/RealSmoke',
    }
    data_root = args.data_root or data_cfg.get('root', default_roots.get(args.dataset, f'./datasets/{args.dataset}'))

    # Use original resolution by default; only resize if --resize_test is set
    if args.resize_test:
        img_size = tuple(data_cfg.get('img_size', [256, 256]))
        print(f"Testing with resized images: {img_size}")
    else:
        img_size = None  # Signal to dataset: no resize, use original size
        print(f"Testing at original image resolution (pad to 16x multiple)")

    test_loader = get_test_loader(
        args.dataset, data_root, batch_size=1, img_size=img_size,
        split=args.split, subset_file=args.manifest)
    if args.manifest:
        print(f"Strict manifest: {args.manifest} ({len(test_loader.dataset)} samples)")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to: {args.output_dir}")

    # Inference
    all_metrics = []
    metric_samples = []
    total = len(test_loader) if args.max_images <= 0 else min(args.max_images, len(test_loader))
    processed = 0
    for idx, batch in enumerate(tqdm(test_loader, desc='Inference', total=total)):
        if args.max_images > 0 and idx >= args.max_images:
            break
        vis = batch['vis'].to(device)
        ir = batch['ir'].to(device)
        fname = batch['filename'][0]

        # Pad to multiple of 16 for encoder/decoder compatibility
        vis_padded, pad_h, pad_w = pad_to_multiple(vis, 16)
        ir_padded, _, _ = pad_to_multiple(ir, 16)

        fused = model(vis_padded, ir_padded)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            fused = fused[:, :, :fused.shape[2]-pad_h, :fused.shape[3]-pad_w]

        # Save fused image
        fused_img = tensor_to_image(fused)
        base = os.path.splitext(fname)[0]
        save_path = os.path.join(args.output_dir, f'{base}.png')

        if fused_img.ndim == 2:
            Image.fromarray(fused_img, mode='L').save(save_path)
        else:
            Image.fromarray(fused_img, mode='RGB').save(save_path)
        processed += 1

        # Compute metrics if requested
        if args.eval_metrics:
            metrics = compute_fusion_metrics(fused, vis, ir)
            all_metrics.append(metrics)
            metric_samples.append(fname)

    # Print average metrics
    if args.eval_metrics and all_metrics:
        print("\n--- Average Fusion Metrics ---")
        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            avg_metrics[key] = float(np.mean(vals))
            print(f"  {key}: {avg_metrics[key]:.4f}")

        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            for k, v in avg_metrics.items():
                f.write(f'{k}: {v:.4f}\n')
        print(f"Metrics saved to: {metrics_path}")

        # SAFIR-Fuse protocol: retain every sample score so the fixed
        # clean/mild/medium/heavy pairs support dispersion estimates and
        # paired statistical tests, in addition to dataset means.
        csv_path = Path(args.metrics_csv or os.path.join(args.output_dir, 'metrics_long.csv'))
        append = args.metrics_csv is not None and csv_path.exists() and csv_path.stat().st_size > 0
        sample_count, row_count = write_per_image_metrics(
            csv_path,
            method=args.method_name,
            dataset=args.dataset,
            split=args.split,
            severity=args.severity,
            sample_metrics=zip(metric_samples, all_metrics),
            append=append,
        )
        print(
            f"Per-image metrics saved to: {csv_path} "
            f"({sample_count} samples, {row_count} rows)"
        )

    print(f"\nDone! {processed} images processed.")


if __name__ == '__main__':
    main()
