"""
Three-phase training pipeline for TIFDFusion.
  Phase 1: Encoder + Decoder only (L_fusion)
  Phase 2: + TIFM with segmentation prior (L_fusion + L_seg_q)
  Phase 3: + TIFM with both priors (L_fusion + L_seg_q + L_det_q)
"""

import os
import sys
import time
import random
import numpy as np
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.tifd_fusion import build_model
from models.losses import TIFDLoss
from data.dataset import get_train_loader, get_test_loader
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='TIFDFusion Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--phase', type=int, default=0, help='Start from phase (0=auto)')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, phase, path):
    # Only the model weights are ever reloaded (phase transitions / --resume use
    # model_state_dict; optimizer/scheduler state is never restored). Dropping them
    # roughly halves checkpoint size to save disk across many ablation runs.
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'phase': phase,
    }, path)


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, reserved
    return 0, 0


def forward_smoke_aware(model, batch, criterion, device, smoke_supervision=None):
    """Run one paired fusion batch and assemble optional smoke supervision."""
    smoke_supervision = smoke_supervision or {}
    supervision_enabled = bool(smoke_supervision.get('enabled', False))

    vis = batch['vis'].to(device)
    ir = batch['ir'].to(device)
    seg_labels = batch.get('label')
    if seg_labels is not None:
        seg_labels = seg_labels.to(device)

    clean_vis = batch.get('vi_clean')
    transmission = batch.get('transmission')
    smoke_mask = batch.get('smoke_mask')
    if clean_vis is not None:
        clean_vis = clean_vis.to(device)
    if transmission is not None:
        transmission = transmission.to(device)
    if smoke_mask is not None:
        smoke_mask = smoke_mask.to(device)

    use_clean_reference = (
        supervision_enabled and
        bool(smoke_supervision.get('use_clean_reference', True)) and
        clean_vis is not None
    )
    need_reliability = (
        supervision_enabled and
        float(criterion.weights.get('reliability', 0.0)) > 0.0
    )
    need_clean_teacher = (
        supervision_enabled and
        bool(smoke_supervision.get('use_clean_teacher', True)) and
        (float(criterion.weights.get('smoke_consistency', 0.0)) > 0.0 or
         float(criterion.weights.get(
             'smoke_edge_consistency', 0.0)) > 0.0) and
        clean_vis is not None and
        smoke_mask is not None and
        bool((smoke_mask > 1e-6).any().item())
    )

    clean_fused = None
    if need_clean_teacher:
        # The clean branch is a deterministic, stop-gradient anchor.  Running
        # it in eval mode avoids a second BatchNorm update in each batch.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            clean_fused = model(clean_vis, ir)
        model.train(was_training)

    if need_reliability:
        fused, aux = model(vis, ir, return_aux=True)
        reliability_maps = aux.get('reliability_vis', [])
    else:
        fused = model(vis, ir)
        reliability_maps = None

    loss, loss_dict = criterion(
        fused,
        vis,
        ir,
        seg_labels,
        clean_vis=clean_vis if use_clean_reference else None,
        transmission=transmission if supervision_enabled else None,
        smoke_mask=smoke_mask if supervision_enabled else None,
        reliability_maps=reliability_maps,
        clean_fused=clean_fused,
    )
    return fused, loss, loss_dict


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer,
                    log_interval=50, smoke_supervision=None):
    model.train()
    total_loss = 0.0
    loss_accum = {}
    epoch_start = time.time()
    lr = optimizer.param_groups[0]['lr']

    pbar = tqdm(loader, desc=f'Ep{epoch}', ncols=80, ascii=True,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for step, batch in enumerate(pbar):
        _, loss, loss_dict = forward_smoke_aware(
            model, batch, criterion, device, smoke_supervision)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            elif isinstance(v, (int, float)):
                loss_accum[k] = loss_accum.get(k, 0.0) + float(v)

        # Update progress bar description with compact loss info
        avg_loss = total_loss / (step + 1)
        pbar.set_description(f'Ep{epoch} L={avg_loss:.3f}')

        # TensorBoard logging
        if (step + 1) % log_interval == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            for k, v in loss_dict.items():
                if k.startswith('_'):
                    continue  # skip internal tracking keys
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                writer.add_scalar(f'train/{k}', val, global_step)
            writer.add_scalar('train/lr', lr, global_step)
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(loader)
    avg_losses = {k: v / len(loader) for k, v in loss_accum.items()
                  if not k.startswith('_')}

    # Epoch summary
    gpu_alloc, gpu_rsv = get_gpu_memory_mb()
    print(f"  Epoch {epoch} Summary | Time: {epoch_time:.1f}s | LR: {lr:.1e} | "
          f"GPU: {gpu_alloc:.0f}/{gpu_rsv:.0f}MB")

    return avg_loss, avg_losses


@torch.no_grad()
def validate(model, loader, criterion, device, smoke_supervision=None):
    model.eval()
    total_loss = 0.0
    loss_accum = {}
    val_start = time.time()

    for batch in tqdm(loader, desc='  Val', ncols=60, ascii=True, leave=False):
        _, loss, loss_dict = forward_smoke_aware(
            model, batch, criterion, device, smoke_supervision)
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k.startswith('_'):
                continue
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            loss_accum[k] = loss_accum.get(k, 0.0) + val

    val_time = time.time() - val_start
    avg_loss = total_loss / max(len(loader), 1)
    avg_losses = {k: v / max(len(loader), 1) for k, v in loss_accum.items()}
    print(f"  Val Summary | Time: {val_time:.1f}s | Loss: {avg_loss:.4f} | "
          f"pixel: {avg_losses.get('pixel', 0):.4f} | ssim: {avg_losses.get('ssim', 0):.4f} | "
          f"grad: {avg_losses.get('gradient', 0):.4f} | scd: {avg_losses.get('scd', 0):.4f}")

    return avg_loss


def run_phase(model, train_loader, val_loader, criterion, cfg_phase, device, 
              phase_num, start_epoch, writer, save_dir,
              smoke_supervision=None):
    """Run one training phase."""
    epochs = cfg_phase['epochs']
    lr = cfg_phase['lr']

    # Update loss weights for this phase
    criterion.update_weights(cfg_phase['loss_weights'])

    # Set model phase
    model.set_phase(phase_num)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')

    phase_start = time.time()
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_count / 1e6:.2f}M | LR: {lr} | Epochs: {epochs}")
    print(f"  Loss weights: {cfg_phase['loss_weights']}")

    for epoch in range(start_epoch, start_epoch + epochs):
        ep_in_phase = epoch - start_epoch + 1
        print(f"\n{'─' * 70}")
        print(f"  Phase {phase_num} | Epoch {ep_in_phase}/{epochs} (global: {epoch})")
        print(f"{'─' * 70}")

        # Train
        train_loss, train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            smoke_supervision=smoke_supervision)
        print(f"  Train Loss: {train_loss:.4f}  |  ", end='')
        print(' | '.join(f'{k}: {v:.4f}' for k, v in train_losses.items() if k != 'total'))

        # Validate
        if val_loader is not None:
            val_loss = validate(
                model, val_loader, criterion, device,
                smoke_supervision=smoke_supervision)
            writer.add_scalar(f'val/loss_phase{phase_num}', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, phase_num,
                              os.path.join(save_dir, f'best_phase{phase_num}.pth'))
                print(f"  ★ Best model saved (val_loss={val_loss:.4f})")

        scheduler.step()

        # (Periodic per-epoch checkpoints removed: only best_phase{N}.pth is kept,
        #  to avoid filling the disk across many ablation runs.)

        # Phase ETA
        phase_elapsed = time.time() - phase_start
        phase_eta = phase_elapsed / ep_in_phase * (epochs - ep_in_phase)
        print(f"  Phase ETA: {phase_eta / 60:.1f} min remaining")

        writer.add_scalar(f'train/loss_phase{phase_num}', train_loss, epoch)
        writer.add_scalar(f'train/lr_phase{phase_num}', optimizer.param_groups[0]['lr'], epoch)

    # When no validation set is used, the fixed final epoch is the protocol
    # checkpoint.  This keeps the official test split completely untouched
    # during training and phase transitions.
    if val_loader is None:
        final_path = os.path.join(save_dir, f'final_phase{phase_num}.pth')
        save_checkpoint(
            model, optimizer, scheduler, start_epoch + epochs - 1,
            phase_num, final_path)
        print(f"  Final phase checkpoint saved: {final_path}")

    phase_time = time.time() - phase_start
    if val_loader is None:
        print(f"\n  Phase {phase_num} Complete! Total time: {phase_time / 60:.1f} min")
    else:
        print(f"\n  Phase {phase_num} Complete! Total time: {phase_time / 60:.1f} min | Best val loss: {best_val_loss:.4f}")

    return start_epoch + epochs


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = cfg.get('output_dir', './experiments')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Seed
    seed = cfg['train'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Build loss.  Task evaluators are instantiated only when a configured
    # phase assigns them a non-zero loss weight.  SAFIR-Fuse deliberately
    # keeps downstream segmentation/detection evaluation outside the fusion
    # training graph; the frozen backbones in ``model.prior`` remain available
    # as one-way feature extractors inside TIFDFusion.
    prior_cfg = cfg['model'].get('prior', {})
    phase_cfgs = [cfg['train'].get(f'phase{i}', {}) for i in (1, 2, 3)]

    def any_positive_loss(name):
        return any(float(phase.get('loss_weights', {}).get(name, 0.0)) > 0.0
                   for phase in phase_cfgs)

    criterion = TIFDLoss(
        seg_model_name=(prior_cfg.get('seg_model')
                        if any_positive_loss('seg_quality') else None),
        det_model_path=(prior_cfg.get('det_model')
                        if any_positive_loss('det_quality') else None),
        num_classes=cfg['data']['msrs'].get('num_classes', 9),
        pixel_adaptive_temperature=cfg['train'].get('pixel_adaptive_temperature', 5.0),
        seg_class_mapping=cfg['data']['msrs'].get('segformer_class_mapping'),
    )
    criterion = criterion.to(device)

    # Build dataloaders
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    msrs_cfg = data_cfg.get('msrs', {})

    train_loader = get_train_loader(
        'msrs', msrs_cfg.get('root', './datasets/MSRS'),
        batch_size=train_cfg.get('batch_size', 8),
        num_workers=train_cfg.get('num_workers', 4),
        img_size=tuple(msrs_cfg.get('img_size', [256, 256])),
        subset_file=msrs_cfg.get('train_manifest'),
        smoke_config=msrs_cfg.get('smoke'),
    )
    validation_enabled = bool(train_cfg.get('validation_enabled', True))
    val_loader = None
    if validation_enabled:
        val_manifest = msrs_cfg.get('val_manifest')
        val_loader = get_test_loader(
            'msrs', msrs_cfg.get('root', './datasets/MSRS'),
            batch_size=1,
            num_workers=train_cfg.get('num_workers', 4),
            img_size=tuple(msrs_cfg.get('img_size', [256, 256])),
            subset_file=val_manifest,
            smoke_config=msrs_cfg.get('validation_smoke'),
            split='train' if val_manifest else 'test',
            has_label=bool(val_manifest),
        )
    smoke_supervision = train_cfg.get('smoke_supervision', {})

    # Resume if specified
    start_phase = args.phase if args.phase > 0 else 1
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_phase = ckpt.get('phase', 1)
        start_epoch = ckpt.get('epoch', 0) + 1

    # ===== Ablation: single-stage training (no 3-phase warmup) =====
    abl_cfg = cfg.get('ablation', {})
    if abl_cfg.get('single_stage', False):
        print("\n" + "=" * 60)
        print("SINGLE-STAGE Training (ablation: all priors/losses from start, no phase warmup)")
        print("=" * 60)
        total_epochs = (cfg['train']['phase1']['epochs']
                        + cfg['train']['phase2']['epochs']
                        + cfg['train']['phase3']['epochs'])
        single_cfg = dict(cfg['train']['phase3'])   # phase-3 loss weights (all priors + task losses)
        single_cfg['epochs'] = total_epochs
        run_phase(model, train_loader, val_loader, criterion,
                  single_cfg, device, 3, start_epoch, writer, save_dir,
                  smoke_supervision=smoke_supervision)
        print("\n" + "=" * 60)
        print("Single-stage training complete!")
        print("=" * 60)
        writer.close()
        return

    # ===== Three-Phase Training =====
    epoch_counter = start_epoch

    # Phase 1: Encoder + Decoder (fusion loss only)
    if start_phase <= 1:
        print("\n" + "=" * 60)
        print("PHASE 1: Encoder + Decoder Training (L_fusion only)")
        print("=" * 60)
        epoch_counter = run_phase(
            model, train_loader, val_loader, criterion,
            cfg['train']['phase1'], device, 1, epoch_counter, writer, save_dir,
            smoke_supervision=smoke_supervision)

    # Phase 2: + TIFM with segmentation prior
    if start_phase <= 2:
        print("\n" + "=" * 60)
        print("PHASE 2: + TIFM with Segmentation Prior")
        print("=" * 60)
        if start_phase <= 2:
            # Load best phase 1 checkpoint
            checkpoint_name = 'best_phase1.pth' if validation_enabled else 'final_phase1.pth'
            ckpt_path = os.path.join(save_dir, checkpoint_name)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                if missing:
                    print(f"  New params (randomly init): {len(missing)} keys")
                print(f"  Loaded best Phase 1 checkpoint")
        epoch_counter = run_phase(
            model, train_loader, val_loader, criterion,
            cfg['train']['phase2'], device, 2, epoch_counter, writer, save_dir,
            smoke_supervision=smoke_supervision)

    # Phase 3: + TIFM with both priors
    if start_phase <= 3:
        print("\n" + "=" * 60)
        print("PHASE 3: + TIFM with Detection Prior")
        print("=" * 60)
        if start_phase <= 3:
            checkpoint_name = 'best_phase2.pth' if validation_enabled else 'final_phase2.pth'
            ckpt_path = os.path.join(save_dir, checkpoint_name)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                if missing:
                    print(f"  New params (randomly init): {len(missing)} keys")
                print(f"  Loaded best Phase 2 checkpoint")
        epoch_counter = run_phase(
            model, train_loader, val_loader, criterion,
            cfg['train']['phase3'], device, 3, epoch_counter, writer, save_dir,
            smoke_supervision=smoke_supervision)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total epochs: {epoch_counter}")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()
