"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

import os

import time

import torch

from tqdm import tqdm

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
