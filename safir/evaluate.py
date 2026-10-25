"""Evaluate saved fused PNGs using the paper's original fusion metrics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from safir.infer import paired_images, index_images
from utils.metrics import compute_fusion_metrics
from utils.metrics_a37_supplemental import compute_a37_supplemental_metrics, METRIC_DIRECTIONS


def gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fused', required=True)
    p.add_argument('--visible', required=True, help='Condition-matched visible inputs')
    p.add_argument('--infrared', required=True)
    p.add_argument('--manifest')
    p.add_argument('--metrics', choices=['9', '21'], default='21')
    p.add_argument('--clean-base', help='Optional clean TIFDFusion reference outputs for recovery metrics')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    pairs = paired_images(args.visible, args.infrared, args.manifest)
    outputs = index_images(args.fused)
    refs = index_images(args.clean_base) if args.clean_base else {}
    if set(outputs) != {s.casefold() for s, _, _ in pairs}:
        raise ValueError('Fused output population differs from selected input population')
    rows = []
    for stem, vi, ir in pairs:
        fused, visible, infrared = gray(outputs[stem.casefold()]), gray(vi), gray(ir)
        if fused.shape != visible.shape or fused.shape != infrared.shape:
            raise ValueError(f'Image dimensions differ for {stem}; metrics never resize inputs')
        metrics = compute_fusion_metrics(fused, visible, infrared)
        metrics['SSIM_mean'] = metrics.pop('SSIM') / 2
        if args.metrics == '21':
            metrics.update(compute_a37_supplemental_metrics(fused, visible, infrared))
        if refs:
            reference = gray(refs[stem.casefold()]) / 255
            value = fused / 255
            if reference.shape != value.shape:
                raise ValueError(f'Recovery reference size differs for {stem}')
            metrics.update(Recovery_MAE=float(np.abs(value-reference).mean()),
                           Recovery_PSNR=float(-10*np.log10(max(float(np.square(value-reference).mean()), 1e-12))),
                           Recovery_SSIM=float(structural_similarity(value, reference, data_range=1)))
        rows.append({'sample': stem, **{k: float(v) for k,v in metrics.items()}})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'sample'}
    report = {'count': len(rows), 'arguments': vars(args), 'mean': summary,
              'directions': {k: METRIC_DIRECTIONS.get(k, 'lower' if k == 'Recovery_MAE' else 'higher') for k in summary}}
    (out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
