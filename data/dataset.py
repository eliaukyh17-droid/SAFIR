"""
Dataset and DataLoader for TIFDFusion.
Supports MSRS (fusion + segmentation) and M3FD (fusion + detection) datasets.
"""

import os
import zlib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .smoke import sample_morphology, synthesize_smoke


def _filter_filenames_by_manifest(filenames, subset_file):
    """Return image filenames in manifest order, matched by case-insensitive stem.

    A manifest is a protocol boundary rather than a best-effort filter.  Missing
    or duplicate stems therefore raise immediately instead of silently changing
    the requested train/test population.
    """
    if subset_file is None:
        return filenames

    with open(subset_file, 'r', encoding='utf-8') as handle:
        requested = [
            os.path.splitext(os.path.basename(line.strip()))[0]
            for line in handle
            if line.strip() and not line.lstrip().startswith('#')
        ]
    if not requested:
        raise ValueError(f'Empty subset manifest: {subset_file}')

    normalized = [stem.casefold() for stem in requested]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f'Duplicate stems in subset manifest: {subset_file}')

    available = {}
    for filename in filenames:
        stem = os.path.splitext(filename)[0].casefold()
        if stem in available:
            raise ValueError(
                f'Ambiguous image stem {stem!r} in dataset directory; '
                f'cannot apply manifest {subset_file}'
            )
        available[stem] = filename

    missing = [stem for stem in requested if stem.casefold() not in available]
    if missing:
        preview = ', '.join(missing[:5])
        raise FileNotFoundError(
            f'{len(missing)} manifest samples are absent from the visible-image '
            f'directory (first: {preview}); manifest={subset_file}'
        )
    return [available[stem] for stem in normalized]


class MSRSDataset(Dataset):
    """
    MSRS Dataset for infrared-visible image fusion + semantic segmentation.
    Directory structure:
      root/
        train/ or test/
          vi/       - visible images (RGB)
          ir/       - infrared images (grayscale)
          label/    - segmentation labels (optional, for training)
    """
    def __init__(self, root, split='train', img_size=(256, 256), has_label=True,
                 augment=None, subset_file=None, smoke_config=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.img_size = img_size
        self.has_label = has_label
        self.split = split
        self.smoke_config = smoke_config or {}
        # augment: None=auto (True for train, False for test), or explicit bool
        if augment is None:
            self.augment = (split == 'train')
        else:
            self.augment = augment

        self.vi_dir = os.path.join(self.root, 'vi')
        self.ir_dir = os.path.join(self.root, 'ir')
        self.label_dir = os.path.join(self.root, 'Segmentation_labels')
        if not os.path.exists(self.label_dir):
            self.label_dir = os.path.join(self.root, 'label')

        # Get file list
        self.filenames = sorted([f for f in os.listdir(self.vi_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
        self.filenames = _filter_filenames_by_manifest(self.filenames, subset_file)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load visible image (RGB)
        vi_path = os.path.join(self.vi_dir, fname)
        vi = Image.open(vi_path).convert('RGB')

        # Load infrared image (grayscale)
        ir_path = os.path.join(self.ir_dir, fname)
        if not os.path.exists(ir_path):
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                ir_path = os.path.join(self.ir_dir, base + ext)
                if os.path.exists(ir_path):
                    break
        ir = Image.open(ir_path).convert('L')

        # Load segmentation label
        label = None
        if self.has_label and os.path.exists(self.label_dir):
            label_path = os.path.join(self.label_dir, fname)
            if not os.path.exists(label_path):
                base = os.path.splitext(fname)[0]
                for ext in ['.png', '.jpg', '.bmp']:
                    label_path = os.path.join(self.label_dir, base + ext)
                    if os.path.exists(label_path):
                        break
            if os.path.exists(label_path):
                label = Image.open(label_path)

        if self.augment and self.img_size is not None:
            # Training: RandomCrop + RandomHorizontalFlip + RandomVerticalFlip
            # Joint transform for vis, ir, label
            crop_h, crop_w = self.img_size
            w, h = vi.size  # PIL size is (W, H)
            # Ensure image is large enough for crop
            if h < crop_h or w < crop_w:
                scale = max(crop_h / h, crop_w / w) + 0.01
                new_h, new_w = int(h * scale), int(w * scale)
                vi = vi.resize((new_w, new_h), Image.BILINEAR)
                ir = ir.resize((new_w, new_h), Image.BILINEAR)
                if label is not None:
                    label = label.resize((new_w, new_h), Image.NEAREST)
                w, h = new_w, new_h
            # Random crop
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
            vi = vi.crop((left, top, left + crop_w, top + crop_h))
            ir = ir.crop((left, top, left + crop_w, top + crop_h))
            if label is not None:
                label = label.crop((left, top, left + crop_w, top + crop_h))
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                vi = vi.transpose(Image.FLIP_LEFT_RIGHT)
                ir = ir.transpose(Image.FLIP_LEFT_RIGHT)
                if label is not None:
                    label = label.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                vi = vi.transpose(Image.FLIP_TOP_BOTTOM)
                ir = ir.transpose(Image.FLIP_TOP_BOTTOM)
                if label is not None:
                    label = label.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.img_size is not None:
            # Testing with resize
            vi = vi.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            ir = ir.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            if label is not None:
                label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        # else: img_size is None, use original size

        vi = self.to_tensor(vi)
        ir = self.to_tensor(ir)

        # SAFIR-Fuse smoke is applied only to VI.  The legacy ``vis`` key is
        # retained and points to the actual model input, while the clean VI is
        # always available for auditing and smoke-aware objectives.
        vi_clean = vi
        vi_smoke = vi_clean
        smoke_mask = torch.zeros((1, *vi.shape[-2:]), dtype=vi.dtype)
        transmission = torch.ones_like(smoke_mask)
        severity = 'clean'
        morphology = 'clean'
        smoke_seed = -1
        smoke_meta = {}
        smoke_enabled = bool(self.smoke_config.get('enabled', False))
        probability = float(self.smoke_config.get('probability', 0.5))
        base_seed = int(self.smoke_config.get('base_seed', 2026))
        deterministic = bool(self.smoke_config.get('deterministic', False))
        selector = None
        if deterministic:
            sample_seed = base_seed + (zlib.crc32(fname.encode('utf-8')) & 0x7fffffff)
            selector = torch.Generator(device='cpu').manual_seed(sample_seed)
        apply_smoke = smoke_enabled and torch.rand((), generator=selector).item() < probability
        if apply_smoke:
            choices = tuple(self.smoke_config.get(
                'severities', ['mild', 'medium', 'heavy']))
            severity = choices[
                int(torch.randint(0, len(choices), (), generator=selector).item())
            ]
            morphologies = tuple(self.smoke_config.get(
                'morphologies', ['local', 'patchy', 'diffuse']))
            morphology_probabilities = tuple(self.smoke_config.get(
                'morphology_probabilities', [0.30, 0.50, 0.20]))
            morphology = sample_morphology(
                selector,
                morphologies=morphologies,
                probabilities=morphology_probabilities,
            )
            if deterministic:
                smoke_seed = sample_seed + 1
            else:
                smoke_seed = base_seed + int(torch.randint(0, 2**30, ()).item())
            vi_smoke, smoke_mask, transmission, smoke_meta = synthesize_smoke(
                vi_clean,
                severity=severity,
                seed=smoke_seed,
                morphology=morphology,
            )

        result = {
            'vis': vi_smoke,
            'vi_clean': vi_clean,
            'vi_smoke': vi_smoke,
            'ir': ir,
            'filename': fname,
            'smoke_mask': smoke_mask,
            'transmission': transmission,
            'severity': severity,
            'morphology': morphology,
            'seed': smoke_seed,
            'smoke_metadata': smoke_meta,
        }

        if label is not None:
            result['label'] = torch.from_numpy(np.array(label)).long()
        elif self.has_label and self.img_size is not None:
            result['label'] = torch.zeros(self.img_size, dtype=torch.long)
        else:
            result['label'] = None

        return result


class GenericPairDataset(Dataset):
    """
    Generic paired dataset for TNO, RoadScene, LLVIP, FMB etc.
    Supports multiple directory structures:
      root/vi/ + root/ir/
      root/Vis/ + root/Ir/
      root/visible/ + root/infrared/
      root/Visible/ + root/Infrared/
      root/{ir_subdir}/{split}/ + root/{vi_subdir}/{split}/  (e.g. LLVIP)
      root/{split}/{ir_subdir}/ + root/{split}/{vi_subdir}/  (e.g. FMB)
    """
    def __init__(self, root, img_size=(256, 256), split=None,
                 subset_file=None):
        super().__init__()
        self.root = root
        self.img_size = img_size

        # Auto-detect directory names with multiple patterns
        self.vi_dir = None
        self.ir_dir = None
        vi_ir_pairs = [
            ('vi', 'ir'), ('Vis', 'Ir'), ('visible', 'infrared'),
            ('Visible', 'Infrared'), ('color', 'Infrared'),
        ]
        # Pattern 1: root/vi_name/ or root/vi_name/split/
        for vi_name, ir_name in vi_ir_pairs:
            vi_candidate = os.path.join(root, vi_name)
            ir_candidate = os.path.join(root, ir_name)
            if split and os.path.isdir(os.path.join(vi_candidate, split)):
                self.vi_dir = os.path.join(vi_candidate, split)
                self.ir_dir = os.path.join(ir_candidate, split)
                break
            elif os.path.isdir(vi_candidate) and any(
                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
                for f in os.listdir(vi_candidate) if os.path.isfile(os.path.join(vi_candidate, f))
            ):
                self.vi_dir = vi_candidate
                self.ir_dir = ir_candidate
                break
        # Pattern 2: root/split/vi_name/ (e.g. FMB/test/Visible)
        if self.vi_dir is None and split:
            for vi_name, ir_name in vi_ir_pairs:
                vi_candidate = os.path.join(root, split, vi_name)
                ir_candidate = os.path.join(root, split, ir_name)
                if os.path.isdir(vi_candidate):
                    self.vi_dir = vi_candidate
                    self.ir_dir = ir_candidate
                    break
        # Fallback: original simple logic
        if self.vi_dir is None:
            self.vi_dir = os.path.join(root, 'vi')
            self.ir_dir = os.path.join(root, 'ir')

        filenames = sorted([f for f in os.listdir(self.vi_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
        self.filenames = _filter_filenames_by_manifest(filenames, subset_file)

        if img_size is not None:
            self.vi_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
            self.ir_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
        else:
            self.vi_transform = transforms.ToTensor()
            self.ir_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        vi_path = os.path.join(self.vi_dir, fname)
        vi = Image.open(vi_path).convert('RGB')
        vi = self.vi_transform(vi)

        ir_path = os.path.join(self.ir_dir, fname)
        if not os.path.exists(ir_path):
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                ir_path = os.path.join(self.ir_dir, base + ext)
                if os.path.exists(ir_path):
                    break
        ir = Image.open(ir_path).convert('L')
        ir = self.ir_transform(ir)

        return {'vis': vi, 'ir': ir, 'filename': fname, 'label': None, 'det_labels': None}


class M3FDDataset(Dataset):
    """
    M3FD Dataset for infrared-visible image fusion + object detection.
    Directory structure:
      root/
        Vis/        - visible images (RGB)
        Ir/         - infrared images
        Annotation/ - YOLO format labels (optional)
    """
    def __init__(self, root, img_size=(640, 640), has_label=True,
                 subset_file=None):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.has_label = has_label

        self.vi_dir = os.path.join(root, 'Vis')
        self.ir_dir = os.path.join(root, 'Ir')
        self.label_dir = os.path.join(root, 'Annotation')

        if not os.path.exists(self.vi_dir):
            self.vi_dir = os.path.join(root, 'vi')
            self.ir_dir = os.path.join(root, 'ir')
        if not os.path.exists(self.label_dir):
            self.label_dir = os.path.join(root, 'labels')

        filenames = sorted([f for f in os.listdir(self.vi_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.filenames = _filter_filenames_by_manifest(filenames, subset_file)

        if img_size is not None:
            self.vi_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
            self.ir_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
        else:
            self.vi_transform = transforms.ToTensor()
            self.ir_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        vi_path = os.path.join(self.vi_dir, fname)
        vi = Image.open(vi_path).convert('RGB')
        vi = self.vi_transform(vi)

        ir_path = os.path.join(self.ir_dir, fname)
        if not os.path.exists(ir_path):
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                ir_path = os.path.join(self.ir_dir, base + ext)
                if os.path.exists(ir_path):
                    break
        ir = Image.open(ir_path).convert('L')
        ir = self.ir_transform(ir)

        result = {'vis': vi, 'ir': ir, 'filename': fname}

        # Load detection labels (YOLO format) if available
        if self.has_label and os.path.exists(self.label_dir):
            base = os.path.splitext(fname)[0]
            label_path = os.path.join(self.label_dir, base + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = [float(p) for p in parts[:5]]
                        boxes.append([cls, cx, cy, w, h])
                result['det_labels'] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
            else:
                result['det_labels'] = torch.zeros((0, 5))
        else:
            result['det_labels'] = None

        return result


def collate_fn(batch):
    """Custom collate function to handle variable-length detection labels."""
    result = {}
    result['vis'] = torch.stack([item['vis'] for item in batch])
    result['ir'] = torch.stack([item['ir'] for item in batch])
    result['filename'] = [item['filename'] for item in batch]

    # Optional SAFIR-Fuse audit fields.  Clean datasets remain compatible and
    # simply omit these entries.
    for key in ('vi_clean', 'vi_smoke', 'smoke_mask', 'transmission'):
        if batch[0].get(key) is not None:
            result[key] = torch.stack([item[key] for item in batch])
    for key in ('severity', 'morphology', 'seed', 'smoke_metadata'):
        if key in batch[0]:
            result[key] = [item[key] for item in batch]

    if batch[0].get('label') is not None:
        result['label'] = torch.stack([item['label'] for item in batch])
    else:
        result['label'] = None

    if batch[0].get('det_labels') is not None:
        result['det_labels'] = [item['det_labels'] for item in batch]
    else:
        result['det_labels'] = None

    return result


def get_train_loader(dataset_name, root, batch_size=8, num_workers=4,
                     img_size=(256, 256), subset_file=None, smoke_config=None):
    """Create training DataLoader."""
    if dataset_name.lower() == 'msrs':
        dataset = MSRSDataset(
            root, split='train', img_size=img_size, has_label=True,
            subset_file=subset_file, smoke_config=smoke_config)
    elif dataset_name.lower() == 'm3fd':
        dataset = M3FDDataset(
            root, img_size=img_size, has_label=True,
            subset_file=subset_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn, drop_last=True)


def get_test_loader(dataset_name, root, batch_size=1, num_workers=4,
                    img_size=(256, 256), subset_file=None, smoke_config=None,
                    split='test', has_label=False):
    """Create test DataLoader."""
    name = dataset_name.lower()
    if name == 'msrs':
        # Materialized Smoke-MSRS severity folders are self-contained
        # ``vi/`` + ``ir/`` roots, whereas clean MSRS uses ``test/vi`` and
        # ``test/ir``.  Accept both without renaming or copying source data.
        direct_pair = (os.path.isdir(os.path.join(root, 'vi')) and
                       os.path.isdir(os.path.join(root, 'ir')))
        split_pair = (os.path.isdir(os.path.join(root, split, 'vi')) and
                      os.path.isdir(os.path.join(root, split, 'ir')))
        if direct_pair and not split_pair:
            dataset = GenericPairDataset(
                root, img_size=img_size, subset_file=subset_file)
        else:
            dataset = MSRSDataset(
                root, split=split, img_size=img_size, has_label=has_label,
                augment=False, subset_file=subset_file,
                smoke_config=smoke_config)
    elif name == 'm3fd':
        # Fixed Smoke-M3FD severity folders produced by
        # tools/generate_smoke_dataset.py contain direct ``vi/`` + ``ir/``
        # pairs.  The clean source dataset commonly uses ``Vis/`` + ``Ir/``.
        # GenericPairDataset supports both layouts and keeps evaluation from
        # depending on the annotation directory.
        dataset = GenericPairDataset(
            root, img_size=img_size, subset_file=subset_file)
    elif name in ('tno', 'roadscene', 'generic', 'real_smoke'):
        dataset = GenericPairDataset(
            root, img_size=img_size, subset_file=subset_file)
    elif name == 'llvip':
        dataset = GenericPairDataset(
            root, img_size=img_size, split=split, subset_file=subset_file)
    elif name == 'fmb':
        dataset = GenericPairDataset(
            root, img_size=img_size, split=split, subset_file=subset_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn)
