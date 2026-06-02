"""
train.py
========
Trains YOLOv8 on the prepared ROAD-Waymo YOLO dataset.

Usage:
    python train.py --dataset_dir /path/to/waymo_yolo
    python train.py --dataset_dir /path/to/waymo_yolo --model yolov8m.pt --epochs 10 --batch 64
    python train.py --dataset_dir /path/to/waymo_yolo --resume --resume_ckpt /path/to/last.pt
"""

import os
import argparse
import torch
from ultralytics import YOLO


def train(
    dataset_dir,
    model_name  = "yolov8m.pt",
    epochs = 10,
    imgsz = 640,
    batch = 64,
    fraction = 1.0,
    patience = 10,
    device = None,
    workers = 8,
    resume = False,
    resume_ckpt = None,
):
    """
    Train YOLOv8 on the ROAD-Waymo dataset.

    Args:
        dataset_dir: Path to YOLO dataset (must contain dataset.yaml)
        model_name: YOLOv8 variant (yolov8s.pt / yolov8m.pt / yolov8x.pt)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (adjust to fit GPU VRAM)
        fraction: Fraction of dataset to use (0.0-1.0)
        patience: Early stopping patience
        device: Device string ('cuda:0', 'cpu'). Auto-detected if None.
        workers: Dataloader workers (0 for Windows, 4-8 for Linux servers)
        resume: Whether to resume from a checkpoint
        resume_ckpt: Path to checkpoint for resuming (last.pt)
    """
    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    runs_dir  = os.path.join(dataset_dir, "runs")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. Run prepare_dataset.py first."
        )

    # Resume from checkpoint or start fresh
    if resume and resume_ckpt:
        print(f"\nResuming from: {resume_ckpt}")
        model = YOLO(resume_ckpt)
        train_results = model.train(resume=True)
    else:
        print(f"\nStarting training with {model_name}")
        model = YOLO(model_name)
        train_results = model.train(
            data      = yaml_path,
            epochs    = epochs,
            imgsz     = imgsz,
            batch     = batch,
            rect      = True,       # rectangular batches better for non-square frames
            device    = device,
            project   = runs_dir,
            name      = "waymo_agent_det",
            pretrained= True,
            patience  = patience,
            fraction  = fraction,
            save      = True,
            plots     = True,
            workers   = workers,
            # Data augmentation by default in YOLO: mosaic, fliplr, hsv_h/s/v, translate
            # Added:
            mixup      = 0.1,   # blends two images helps minority classes
            copy_paste = 0.1,   # copies objects across images key for
                                # Cyclist and LargeVeh which are underrepresented
            degrees    = 5.0,   # slight rotation camera angle variation
            flipud     = 0.0,   # disabled sky is always up in driving footage
        )

    best_model = os.path.join(runs_dir, "waymo_agent_det", "weights", "best.pt")
    print(f"\nTraining complete.")
    print(f"Best model: {best_model}")
    return best_model


def validate(dataset_dir, model_path, device=None):
    """Run validation and print per-class AP."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    yaml_path   = os.path.join(dataset_dir, "dataset.yaml")
    model       = YOLO(model_path)
    results     = model.val(data=yaml_path, device=device)
    CLASS_NAMES = ["Pedestrian", "Car", "MedVeh", "LargeVeh", "Cyclist", "TrafficLight"]

    print(f"\n=== Validation results ===")
    print(f"mAP@0.50: {results.box.map50:.4f}")
    print(f"mAP@0.50:0.95: {results.box.map:.4f}")
    print(f"\nPer-class AP@0.50:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s}: {results.box.ap50[i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ROAD-Waymo dataset")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to YOLO dataset directory")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="YOLOv8 model variant (default: yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (default: 64 for Titan RTX 24GB)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Dataset fraction to use (default: 1.0)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--device",  default=None,
                        help="Device: cuda:0 / cpu (auto-detected if not set)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers (default: 8) ")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--resume_ckpt", default=None,
                        help="Path to last.pt for resuming")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")
    args = parser.parse_args()

    best_model = train(
        dataset_dir = args.dataset_dir,
        model_name  = args.model,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        fraction    = args.fraction,
        patience    = args.patience,
        device      = args.device,
        workers     = args.workers,
        resume      = args.resume,
        resume_ckpt = args.resume_ckpt,
    )

    if args.validate:
        validate(args.dataset_dir, best_model, device=args.device)