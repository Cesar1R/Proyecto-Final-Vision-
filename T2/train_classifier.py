"""
train_classifier.py
===================
Trains a multi-label action + location classifier on agent crops for T2.

Architecture: EfficientNet-B0 backbone + two sigmoid heads
Loss: BCEWithLogitsLoss (multi-label)
Input: 112x112 agent crops from prepare_crops.py

Usage:
    python train_classifier.py --crops_dir /path/to/waymo_crops
    python train_classifier.py --crops_dir /path/to/waymo_crops --epochs 10 --batch 256
"""

import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Constants
N_ACTIONS = 22
N_LOCS    = 16
CROP_SIZE = 112

ACTION_NAMES = ["Red","Amber","Green","MovAway","MovTow","Mov","Rev","Brake",
                "Stop","IncatLft","IncatRht","HazLit","TurLft","TurRht",
                "MovRht","MovLft","Ovtak","Wait2X","XingFmLft","XingFmRht",
                "Xing","PushObj"]
LOC_NAMES    = ["VehLane","OutgoLane","OutgoCycLane","OutgoBusLane","IncomLane",
                "IncomCycLane","IncomBusLane","Pav","LftPav","RhtPav","Jun",
                "xing","BusStop","parking","LftParking","rightParking"]


# Dataset
class CropDataset(Dataset):
    """
    Loads agent crops and their multi-label action/location vectors.
    """
    def __init__(self, crops_dir, split, transform=None):
        self.crops_dir = os.path.join(crops_dir, split)
        self.transform = transform

        labels_path = os.path.join(crops_dir, f"{split}_labels.json")
        with open(labels_path) as f:
            self.labels = json.load(f)

        self.stems = list(self.labels.keys())
        print(f"{split}: {len(self.stems):,} crops loaded", flush=True)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        label = self.labels[stem]

        img_path = os.path.join(self.crops_dir, f"{stem}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        action_vec = torch.tensor(label['action_vec'], dtype=torch.float32)
        loc_vec = torch.tensor(label['loc_vec'], dtype=torch.float32)

        return img, action_vec, loc_vec


# Model
class ActionLocClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with two independent multi-label heads:
      - action_head: 22 classes (vehicle actions)
      - loc_head: 16 classes (road locations)
    """
    def __init__(self, n_actions=N_ACTIONS, n_locs=N_LOCS, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        backbone.load_state_dict(torch.load("/home/est_licenciatura_julian.rodrig/Proyecto-final-vision/T1/efficientnet_b0_rwightman-7f5810bc.pth", map_location="cpu"))
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.action_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_actions),
        )
        self.loc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_locs),
        )

    def forward(self, x):
        feat = self.backbone(x)
        actions = self.action_head(feat)  # logits — BCEWithLogitsLoss handles sigmoid
        locs = self.loc_head(feat)
        return actions, locs


# Metrics
def compute_map(preds, targets, threshold=0.5):
    """Compute mean Average Precision for multi-label classification."""
    preds = torch.sigmoid(preds).cpu().numpy()
    targets = targets.cpu().numpy()
    aps = []

    for c in range(targets.shape[1]):
        gt = targets[:, c]
        pred = preds[:, c]
        if gt.sum() == 0:
            continue
        # Simple AP: correlation between prediction and ground truth
        sorted_idx = np.argsort(-pred)
        gt_sorted = gt[sorted_idx]
        tp_cumsum = np.cumsum(gt_sorted)
        precision = tp_cumsum / (np.arange(len(gt_sorted)) + 1)
        recall = tp_cumsum / gt.sum()
        ap = np.trapz(precision, recall) if recall[-1] > 0 else 0.0
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


# Training loop
def train(
    crops_dir,
    epochs = 10,
    batch = 256,
    lr = 1e-3,
    workers = 8,
    device = None,
    save_dir = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}", flush=True)

    save_dir = save_dir or os.path.join(crops_dir, "classifier_runs")
    os.makedirs(save_dir, exist_ok=True)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = CropDataset(crops_dir, 'train', transform=train_tf)
    val_ds   = CropDataset(crops_dir, 'val',   transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)

    # Model
    model = ActionLocClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_map  = 0.0
    best_path = os.path.join(save_dir, "best_classifier.pt")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0

        for i, (imgs, action_gt, loc_gt) in enumerate(train_dl):
            imgs = imgs.to(device)
            action_gt = action_gt.to(device)
            loc_gt = loc_gt.to(device)

            action_pred, loc_pred = model(imgs)

            loss = criterion(action_pred, action_gt) + criterion(loc_pred,   loc_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 200 == 0:
                gpu_mem = torch.cuda.memory_reserved(0) / 1e9
                print(f"  Epoch {epoch}/{epochs} "
                    f"batch {i+1}/{len(train_dl)} "
                    f"loss={loss.item():.4f} "
                    f"GPU={gpu_mem:.1f}GB", flush=True)
        scheduler.step()
        avg_train_loss = train_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss = 0.0
        all_action_pred, all_action_gt = [], []
        all_loc_pred, all_loc_gt = [], []

        with torch.no_grad():
            for imgs, action_gt, loc_gt in val_dl:
                imgs = imgs.to(device)
                action_gt = action_gt.to(device)
                loc_gt = loc_gt.to(device)

                action_pred, loc_pred = model(imgs)
                loss = criterion(action_pred, action_gt) + criterion(loc_pred, loc_gt)
                val_loss += loss.item()

                all_action_pred.append(action_pred.cpu())
                all_action_gt.append(action_gt.cpu())
                all_loc_pred.append(loc_pred.cpu())
                all_loc_gt.append(loc_gt.cpu())

        avg_val_loss = val_loss / len(val_dl)
        action_map = compute_map(
            torch.cat(all_action_pred), torch.cat(all_action_gt)
        )
        loc_map = compute_map(
            torch.cat(all_loc_pred), torch.cat(all_loc_gt)
        )
        mean_map = (action_map + loc_map) / 2

        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | "
              f"action_mAP={action_map:.4f} | "
              f"loc_mAP={loc_map:.4f} | "
              f"mean_mAP={mean_map:.4f}", flush=True)

        if mean_map > best_map:
            best_map = mean_map
            torch.save({
                'epoch' : epoch,
                'model_state': model.state_dict(),
                'action_map' : action_map,
                'loc_map' : loc_map,
                'mean_map': mean_map,
            }, best_path)
            print(f"  Best model saved (mean_mAP={best_map:.4f})", flush=True)

    print(f"\nTraining complete. Best mean_mAP: {best_map:.4f}", flush=True)
    print(f"Best model: {best_path}", flush=True)
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action/location classifier for T2")
    parser.add_argument("--crops_dir", required=True,
                        help="Path to waymo_crops/ directory")
    parser.add_argument("--epochs",  type=int,   default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--batch",   type=int,   default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr",      type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--workers", type=int,   default=8,
                        help="Dataloader workers (default: 8)")
    parser.add_argument("--device",  default=None,
                        help="Device: cuda / cpu (auto-detected if not set)")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    train(
        crops_dir = args.crops_dir,
        epochs = args.epochs,
        batch = args.batch,
        lr = args.lr,
        workers = args.workers,
        device = args.device,
        save_dir  = args.save_dir,
    )
