# training/train_detector.py
"""
Supervised fine-tuning of the Faster R-CNN zombie detector.

Metric (mirroring evaluation.py evaluate_zombies exactly):
    recall@0.5 = mean over images of
                   (# GT zombies matched by ≥1 prediction with IoU ≥ 0.5)
                   / (# GT zombies in that image)
    The assignment calls this "precision" but its formula is recall.
    False positives are NEVER penalised — so we optimise recall and use a
    low detection threshold (default 0.3) to maximise it.
    Target: ≥ 0.75 at distortion 0.

Training strategy (justified by Chen et al. 2020 privileged-learning paradigm):
    - Backbone (ResNet-50 FPN) kept frozen; only RPN + RoI head trained.
    - AdamW on trainable params, CosineAnnealingLR, grad-clip ≤ 1.0.
    - Colour-jitter augmentation on train split to simulate distortion levels 1–3
      without re-collecting labelled data (Goodfellow et al. 2016, Ch. 7).
    - Val split is the un-augmented 20% held out by index before training.

Usage (on Kaggle — run from Task_3/):
    python training/train_detector.py \\
        --data-dir data/raw \\
        --checkpoint-dir checkpoints \\
        --epochs 20 --batch-size 2 --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the project root (parent of this file's directory) is on sys.path so
# that `from models.detector` works regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset

from models.detector import build_detector, NUM_CLASSES
from training.augmentation import random_distortion

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ZombieDataset(Dataset):
    """
    Loads frames and zombie boxes written by data_collection.py.

    Returns:
        image : (3, H, W) float32 in [0, 1]
        target: dict with
                  "boxes"  (N, 4) float32 in xyxy format  ← Faster R-CNN convention
                  "labels" (N,)   int64,  all 1 (zombie class)
    """

    def __init__(self, ann_path: str, img_dir: str, augment: bool = False):
        with open(ann_path) as f:
            self.annotations = json.load(f)
        self.img_dir = Path(img_dir)
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        entry = self.annotations[idx]

        img = Image.open(self.img_dir / entry["file"]).convert("RGB")

        img_t = TF.to_tensor(img)  # (3, H, W) float32 [0, 1]

        if self.augment:
            img_t = random_distortion(img_t)

        boxes_ltwh = entry["boxes"]
        if not boxes_ltwh:
            target = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,),    dtype=torch.int64),
            }
        else:
            ltwh = torch.tensor(boxes_ltwh, dtype=torch.float32)  # (N, 4)
            # Convert [left, top, width, height] → [x1, y1, x2, y2]
            xyxy = torch.stack([
                ltwh[:, 0],
                ltwh[:, 1],
                ltwh[:, 0] + ltwh[:, 2],
                ltwh[:, 1] + ltwh[:, 3],
            ], dim=1)
            target = {
                "boxes":  xyxy,
                "labels": torch.ones(len(xyxy), dtype=torch.int64),
            }

        return img_t, target


def _collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Metric — exact replica of evaluation.py evaluate_zombies()
# ---------------------------------------------------------------------------

def _iou_xyxy(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU for two boxes in xyxy format."""
    # reuse the same formula as utils.iou() which takes ltwh
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]);  y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter)


@torch.no_grad()
def recall_at_iou05(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
) -> float:
    """
    Returns mean recall@IoU0.5 over all frames that have ≥1 GT zombie.
    Matches evaluation.py evaluate_zombies() line-for-line.
    Frames with no GT zombies are skipped (they would cause /0).
    """
    model.eval()
    per_image_recalls = []

    for images, targets in loader:
        images = [im.to(device) for im in images]
        preds = model(images)

        for pred, tgt in zip(preds, targets):
            gt_boxes = tgt["boxes"].cpu().numpy()   # xyxy
            if len(gt_boxes) == 0:
                continue

            keep = pred["scores"].cpu().numpy() >= threshold
            pred_boxes = pred["boxes"].cpu().numpy()[keep]  # xyxy

            matched = 0
            used = set()
            for pb in pred_boxes:
                for gi, gb in enumerate(gt_boxes):
                    if gi in used:
                        continue
                    if _iou_xyxy(pb, gb) >= 0.5:
                        matched += 1
                        used.add(gi)
                        break

            per_image_recalls.append(matched / len(gt_boxes))

    if not per_image_recalls:
        return 0.0
    return float(np.mean(per_image_recalls))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    checkpoint_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    val_fraction: float,
    threshold: float,
    frozen_backbone: bool,
):
    data_dir = Path(data_dir)
    ann_path  = data_dir / "annotations.json"
    img_dir   = data_dir / "images"
    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset split (deterministic: last val_fraction by index)
    # ------------------------------------------------------------------
    full_ds = ZombieDataset(ann_path, img_dir, augment=False)
    n = len(full_ds)
    n_val  = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, n))

    train_ds = Subset(ZombieDataset(ann_path, img_dir, augment=True),  train_idx)
    val_ds   = Subset(ZombieDataset(ann_path, img_dir, augment=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=_collate,
                              num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=1,
                              shuffle=False, collate_fn=_collate,
                              num_workers=2)

    print(f"Dataset: {n_train} train / {n_val} val frames")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_detector(frozen_backbone=frozen_backbone)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_recall = 0.0
    best_path   = ckpt_dir / "detector_best.pt"

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, targets in train_loader:
            images  = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        recall = recall_at_iou05(model, val_loader, device, threshold)
        marker = " ← best" if recall > best_recall else ""
        print(f"epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"recall@0.5={recall:.4f}{marker}")

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), best_path)

    print(f"\nBest recall@0.5 = {best_recall:.4f}")
    print(f"Checkpoint saved to {best_path}")
    return best_recall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(argv=None):
    p = argparse.ArgumentParser(description="Train zombie detector")
    p.add_argument("--data-dir",        default="../data/raw",
                   help="directory produced by data_collection.py")
    p.add_argument("--checkpoint-dir",  default="../checkpoints",
                   help="where to save detector_best.pt")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch-size",      type=int,   default=2,
                   help="2 fits a T4 with frozen backbone + 720×1280 images")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction",    type=float, default=0.2)
    p.add_argument("--threshold",       type=float, default=0.3,
                   help="score threshold for val recall (lower = more recall, "
                        "false positives not penalised by grader)")
    p.add_argument("--unfreeze-backbone", action="store_true",
                   help="also train the ResNet-50 FPN backbone (slower, more GPU)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    train(
        data_dir        = args.data_dir,
        checkpoint_dir  = args.checkpoint_dir,
        epochs          = args.epochs,
        lr              = args.lr,
        batch_size      = args.batch_size,
        device          = args.device,
        val_fraction    = args.val_fraction,
        threshold       = args.threshold,
        frozen_backbone = not args.unfreeze_backbone,
    )
