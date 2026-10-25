"""Native-resolution paired-image inference with the released SAFIR weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from safir.model import load_model, sha256

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def index_images(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f'Duplicate image stem: {path}')
            result[key] = path
    return result


def paired_images(visible, infrared, manifest=None):
    vi, ir = index_images(visible), index_images(infrared)
    if manifest:
        names = [Path(x.strip()).stem.casefold() for x in Path(manifest).read_text(encoding='utf-8-sig').splitlines() if x.strip() and not x.lstrip().startswith('#')]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate samples in manifest')
    else:
        names = sorted(vi)
        if set(vi) != set(ir):
            raise ValueError('Visible/infrared sample sets differ; use a manifest to select a subset')
    if not names:
        raise ValueError('No paired images selected')
    missing = set(names) - (vi.keys() & ir.keys())
    if missing:
        raise FileNotFoundError(f'Missing paired samples: {sorted(missing)[:10]}')
    return [(vi[k].stem, vi[k], ir[k]) for k in names]


def load_pair(visible, infrared, device):
    vi = np.array(Image.open(visible).convert('RGB'), dtype=np.float32) / 255
    ir = np.array(Image.open(infrared).convert('L'), dtype=np.float32) / 255
    if vi.shape[:2] != ir.shape:
        raise ValueError(f'Unaligned image sizes: {visible}, {infrared}')
    height, width = ir.shape
    if min(height, width) < 32:
        raise ValueError('Use images at least 32 pixels along each dimension')
    vi = torch.from_numpy(vi).permute(2, 0, 1).unsqueeze(0).to(device)
    ir = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(device)
    pad = (0, (-width) % 16, 0, (-height) % 16)
    return F.pad(vi, pad, mode='reflect'), F.pad(ir, pad, mode='reflect'), (height, width)


def save_luminance(value, path, size):
    height, width = size
    # Match paper inference: truncate to uint8 after clipping (do not round).
    array = (value[0, 0, :height, :width].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visible', required=True)
    p.add_argument('--infrared', required=True)
    p.add_argument('--manifest')
    p.add_argument('--condition', required=True, choices=['normal', 'smoke'])
    p.add_argument('--checkpoint', default='weights/SAFIR_final.safetensors')
    p.add_argument('--segformer', default='pretrained/segformer-b2')
    p.add_argument('--yolo', default='pretrained/yolov8s.pt')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--scan-backend', default='auto', choices=['auto', 'cuda', 'reference'])
    p.add_argument('--output', required=True)
    p.add_argument('--save-base', action='store_true')
    p.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    p.add_argument('--custom-checkpoint', action='store_true', help='Allow a locally trained checkpoint with a different SHA256')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    pairs = paired_images(args.visible, args.infrared, args.manifest)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    if args.save_base:
        (out / 'base').mkdir()
    model = load_model(args.checkpoint, segformer_path=args.segformer, yolo_path=args.yolo,
                       device=args.device, scan_backend=args.scan_backend,
                       verify_checkpoint=not args.custom_checkpoint)
    records = []
    with torch.inference_mode():
        for index, (stem, visible, infrared) in enumerate(pairs, 1):
            vi, ir, size = load_pair(visible, infrared, args.device)
            started = time.perf_counter()
            fused, aux = model(vi, ir, degradation_condition=args.condition, return_aux=True)
            delta = float((fused - aux['canonical']).abs().max())
            if args.condition == 'normal' and (delta != 0 or aux['expert_evaluated']):
                raise RuntimeError('Normal route identity check failed')
            if delta > 0.150001 or not bool(torch.isfinite(fused).all()):
                raise RuntimeError('Nonfinite output or residual-bound violation')
            target = out / f'{stem}.png'
            save_luminance(fused, target, size)
            if args.save_base:
                save_luminance(aux['canonical'], out / 'base' / target.name, size)
            records.append({'sample': stem, 'height': size[0], 'width': size[1],
                            'maximum_applied_change': delta, 'seconds': time.perf_counter()-started,
                            'visible_sha256': sha256(visible), 'infrared_sha256': sha256(infrared),
                            'output_sha256': sha256(target)})
            print(f'[{index}/{len(pairs)}] {stem}: {size[1]}x{size[0]}', flush=True)
    report = {'arguments': vars(args), 'load_report': model.load_report,
              'checkpoint_sha256': sha256(args.checkpoint), 'torch': torch.__version__, 'samples': records}
    (out / 'inference.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
