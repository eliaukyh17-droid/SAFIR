"""Train DeepLabV3+ (ResNet-50, ImageNet) on a fusion model's fused FMB images and
evaluate semantic-segmentation mIoU — the segmentation analogue of the M3FD detection
protocol (an external, neutral pretrained network fine-tuned on each model's fused
images, then measured on a held-out test split).

FMB (SegMiF) has 15-class labels (index 0..14); class 0 = unlabeled/background. We report
both the 15-class mIoU and the 14-class foreground mIoU (ignoring class 0). The exact
same config is used for every model, so the comparison is fair.

Example:
    python train_seg_fmb.py \
        --fused_train /data/coding/IVIF/开源模型/FNet-MMIF-main/results_fmb_seg/FMB_TRAIN \
        --fused_test  /data/coding/IVIF/开源模型/FNet-MMIF-main/results_msrs_retrain/FMB \
        --label_train /data/coding/IVIF/TIFDFusion/datasets/FMB/train/Label \
        --label_test  /data/coding/IVIF/TIFDFusion/datasets/FMB/test/Label \
        --tag FNet_msrs --epochs 100 --gpu 0
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


NUM_CLASSES = 15
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Fused-FMB semantic segmentation (DeepLabV3+/ResNet-50)")
    p.add_argument("--fused_train", required=True, help="dir of fused FMB *train* images")
    p.add_argument(
        "--extra_fused_train",
        action="append",
        default=[],
        help=("Additional fused FMB train directory. Repeat for mild/medium/heavy "
              "to run the clean+smoke mixed-training protocol."),
    )
    p.add_argument("--fused_test", required=True, help="dir of fused FMB *test* images")
    p.add_argument("--label_train", required=True, help="FMB/train/Label")
    p.add_argument("--label_test", required=True, help="FMB/test/Label")
    p.add_argument("--tag", required=True, help="model tag, e.g. FNet_msrs")
    p.add_argument("--out_dir", default="runs/fmb_segmentation")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--encoder_weights_file", default="",
                   help="Local ResNet-50 ImageNet weights (.safetensors or .pth). If given, "
                        "skip the HuggingFace download and load this file into the encoder.")
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--evaluate_every_epoch",
        action="store_true",
        help=("Legacy diagnostic mode: evaluate the official test split after "
              "every epoch and additionally save best.pth. The formal protocol "
              "leaves this disabled, evaluates test once after the final epoch, "
              "and reports final.pth."),
    )
    return p.parse_args()


def evaluation_epochs(epochs, evaluate_every_epoch=False):
    """Return the held-out evaluation schedule for the selected protocol."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if evaluate_every_epoch:
        return tuple(range(1, epochs + 1))
    return (epochs,)


class FMBSeg:
    """Pairs fused images with FMB labels by filename stem."""
    def __init__(self, img_dir, lab_dir, size, train):
        import torch  # local import (after CUDA env is set)
        self.torch = torch
        self.size = size  # (H, W)
        self.train = train
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        labs = {Path(p).stem: p for p in glob_files(lab_dir) if Path(p).suffix.lower() in exts}
        image_dirs = [img_dir] if isinstance(img_dir, (str, os.PathLike)) else list(img_dir)
        self.pairs = []
        for image_dir in image_dirs:
            for ip in glob_files(image_dir):
                if Path(ip).suffix.lower() not in exts:
                    continue
                lp = labs.get(Path(ip).stem)
                if lp is not None:
                    self.pairs.append((ip, lp))
        if not self.pairs:
            raise RuntimeError(f"No fused-image/label pairs between {image_dirs} and {lab_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        H, W = self.size
        img = Image.open(ip).convert("RGB").resize((W, H), Image.BILINEAR)
        lab = Image.open(lp).convert("L").resize((W, H), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        lab = np.asarray(lab, dtype=np.int64)
        if self.train and random.random() < 0.5:  # horizontal flip
            img = img[:, ::-1, :].copy()
            lab = lab[:, ::-1].copy()
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.torch.from_numpy(img.transpose(2, 0, 1))
        lab = self.torch.from_numpy(lab)
        return img, lab


def glob_files(d):
    import glob
    return sorted(glob.glob(os.path.join(d, "*")))


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)


def main():
    args = parse_args()
    # Respect platform GPU isolation: only override CVD if the platform set one.
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp
    from tqdm import tqdm

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (args.height, args.width)
    train_sources = [args.fused_train, *args.extra_fused_train]
    missing_sources = [source for source in train_sources if not Path(source).is_dir()]
    if missing_sources:
        raise FileNotFoundError(f"Missing fused training directories: {missing_sources}")
    train_set = FMBSeg(train_sources, args.label_train, size, train=True)
    test_set = FMBSeg(args.fused_test, args.label_test, size, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    iters_per_epoch = len(train_loader)
    max_iter = iters_per_epoch * args.epochs
    print(f"[{args.tag}] train {len(train_set)} / test {len(test_set)} | "
          f"iters/epoch {iters_per_epoch} | classes {NUM_CLASSES}")
    print(f"[{args.tag}] train sources ({len(train_sources)}): {train_sources}")

    enc_w = None if args.encoder_weights_file else "imagenet"
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=enc_w,
                              in_channels=3, classes=NUM_CLASSES)
    if args.encoder_weights_file:
        wp = args.encoder_weights_file
        if wp.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(wp)
        else:
            sd = torch.load(wp, map_location="cpu")
        ret = model.encoder.load_state_dict(sd, strict=False)
        if ret is not None:
            print(f"Loaded encoder weights from {wp} (missing {len(ret.missing_keys)}, unexpected {len(ret.unexpected_keys)})")
        else:
            print(f"Loaded encoder weights from {wp}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    def poly_lr(it):
        return args.lr * (1 - it / max_iter) ** 0.9

    @torch.no_grad()
    def evaluate():
        model.eval()
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for img, lab in test_loader:
            img = img.to(device, non_blocking=True)
            pred = model(img).argmax(1).cpu().numpy()  # (B,H,W)
            hist += fast_hist(pred.flatten(), lab.numpy().flatten(), NUM_CLASSES)
        iou = per_class_iou(hist)
        acc_cls = np.diag(hist) / (hist.sum(1) + 1e-10)      # per-class recall
        aacc = float(np.diag(hist).sum() / (hist.sum() + 1e-10))  # overall pixel acc
        freq = hist.sum(1) / (hist.sum() + 1e-10)
        fwiou = float((freq * iou).sum())
        return iou, acc_cls, aacc, fwiou

    evaluation_schedule = set(evaluation_epochs(
        args.epochs, args.evaluate_every_epoch))
    if args.evaluate_every_epoch:
        print("WARNING: legacy per-epoch test evaluation is enabled; "
              "do not use best.pth for the formal benchmark.")
    else:
        print("Formal protocol: the official FMB test split is evaluated once "
              "after the final epoch; final.pth is the reported checkpoint.")

    best_miou_fg = float("-inf")
    best_report = None
    final_report = None
    global_it = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for img, lab in tqdm(train_loader, desc=f"{args.tag} ep{epoch+1}/{args.epochs}", ncols=100):
            lr = poly_lr(global_it)
            for g in optimizer.param_groups:
                g["lr"] = lr
            img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(img), lab)
            loss.backward()
            optimizer.step()
            running += loss.item()
            global_it += 1

        if epoch + 1 not in evaluation_schedule:
            print(f"[{args.tag}] epoch {epoch+1:3d} "
                  f"loss={running/iters_per_epoch:.4f} lr={lr:.2e} | "
                  "held-out evaluation deferred", flush=True)
            continue

        iou, acc_cls, aacc, fwiou = evaluate()
        miou_all = float(np.nanmean(iou))
        miou_fg = float(np.nanmean(iou[1:]))  # ignore class 0 (unlabeled/background)
        macc = float(np.nanmean(acc_cls))
        macc_fg = float(np.nanmean(acc_cls[1:]))
        print(f"[{args.tag}] epoch {epoch+1:3d} loss={running/iters_per_epoch:.4f} "
              f"lr={lr:.2e} | mIoU(15)={miou_all:.4f} mIoU_fg(14)={miou_fg:.4f} "
              f"mAcc={macc:.4f} aAcc={aacc:.4f} FWIoU={fwiou:.4f}", flush=True)
        report = {"epoch": epoch + 1, "mIoU_15": miou_all,
                  "mIoU_fg14": miou_fg, "mAcc": macc,
                  "mAcc_fg": macc_fg, "aAcc": aacc, "FWIoU": fwiou,
                  "per_class_IoU": [round(float(x), 4) for x in iou]}

        if args.evaluate_every_epoch and miou_fg > best_miou_fg:
            best_miou_fg = miou_fg
            best_report = report
            torch.save(model.state_dict(), out_dir / "best.pth")

        if epoch + 1 == args.epochs:
            final_report = report
            torch.save(model.state_dict(), out_dir / "final.pth")

    print("\n" + "=" * 60)
    if final_report is None:
        raise RuntimeError("Final held-out evaluation was not completed")
    print(f"FINAL [{args.tag}]  (epoch {final_report['epoch']})")
    print(f"  checkpoint           = {out_dir / 'final.pth'}")
    print(f"  mIoU(15 classes)     = {final_report['mIoU_15']:.4f}")
    print(f"  mIoU_fg(14, ignore0) = {final_report['mIoU_fg14']:.4f}")
    print(f"  mAcc(15)             = {final_report['mAcc']:.4f}")
    print(f"  aAcc                 = {final_report['aAcc']:.4f}")
    print(f"  FWIoU                = {final_report['FWIoU']:.4f}")
    print(f"  per-class IoU        = {final_report['per_class_IoU']}")
    print("=" * 60)
    (out_dir / "result.txt").write_text(str(final_report), encoding="utf-8")
    (out_dir / "final_result.txt").write_text(
        str(final_report), encoding="utf-8")
    if best_report is not None:
        (out_dir / "legacy_best_result.txt").write_text(
            str(best_report), encoding="utf-8")


if __name__ == "__main__":
    main()
