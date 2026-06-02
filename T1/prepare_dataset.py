"""
prepare_dataset.py
==================
Converts ROAD-Waymo dataset to YOLOv8 format.

- Extracts annotated frames from .mp4 videos, indicated by road_waymo_trainval_v1.0.json.
- Applies gamma correction + CLAHE to every frame.
- Groups 10 original agent classes into 6 broader classes.
- Splits 798 annotated videos into train/val (default 85%/15%).
- Writes YOLO .txt labels and dataset.yaml.

Usage:
    python prepare_dataset.py --waymo_dir /path/to/Waymo --output_dir /path/to/output
    python prepare_dataset.py --waymo_dir /path/to/Waymo --output_dir /path/to/output --workers 6 --val_split 0.15
"""

import os
import json
import argparse
import cv2
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

# Agent grouping
AGENT_GROUPS = {
    "Ped" : (0, "Pedestrian"),
    "Car" : (1, "Car"),
    "SmalVeh" : (1, "Car"),       # grouped into Car
    "MedVeh" : (2, "MedVeh"),
    "LarVeh" : (3, "LargeVeh"),
    "Bus" : (3, "LargeVeh"),  # grouped into LargeVeh
    "EmVeh" : (3, "LargeVeh"),  # grouped into LargeVeh
    "Cyc" : (4, "Cyclist"),
    "Mobike" : (4, "Cyclist"),   # grouped into Cyclist
    "TL" : (5, "TrafficLight"),
}
CLASS_NAMES = ["Pedestrian", "Car", "MedVeh", "LargeVeh", "Cyclist", "TrafficLight"]
NUM_CLASSES = len(CLASS_NAMES)


# Image enhancement
def apply_enhancement(img_bgr, gamma=0.7, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Gamma correction followed by CLAHE on the L channel of LAB colorspace.

    gamma < 1.0 → brightens
    gamma > 1.0 → darkens
    gamma = 1.0 → no change
    """
    # LUT function precomputes gamma correction for all pixel values (0-255) and applies it to all pixels.
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    img_gamma = cv2.LUT(img_bgr, table)

    # CLAHE on L channel of LAB 
    lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L_eq = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L_eq, a, b]), cv2.COLOR_LAB2BGR)


# Function to process one video, meant to be called in parallel. 
def process_single_video(args):
    """
    Reads annotated frames from one video, applies enhancement,
    and writes YOLO images + label files.

    Returns a dict with per-class instance counts.
    """
    video_key, video_data, videos_dir, output_dir, agent_labels, split = args
    stats = defaultdict(int)

    # Collect annotated frames and build YOLO labels 
    frame_annos = {}
    for frame_key, frame_data in video_data['frames'].items():
        if not frame_data.get('annotated', False):
            continue
        if 'annos' not in frame_data:
            continue

        frame_id = int(frame_key)
        labels = []

        for anno in frame_data['annos'].values():
            if not anno.get('agent_ids'):
                continue
            agent_name = agent_labels[anno['agent_ids'][0]]
            if agent_name not in AGENT_GROUPS:
                continue

            cls_id, _ = AGENT_GROUPS[agent_name]
            x1, y1, x2, y2 = anno['box']

            cx = max(0.0, min(1.0, (x1 + x2) / 2))
            cy = max(0.0, min(1.0, (y1 + y2) / 2))
            w = max(0.001, min(1.0, x2 - x1))
            h = max(0.001, min(1.0, y2 - y1))

            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[CLASS_NAMES[cls_id]] += 1

        if labels:
            frame_annos[frame_id] = labels

    if not frame_annos:
        return dict(stats)

    # Open video
    video_path = os.path.join(videos_dir, f"{video_key}.mp4")
    if not os.path.exists(video_path):
        stats['missing_videos'] += 1
        return dict(stats)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        stats['missing_videos'] += 1
        return dict(stats)

    # Extract frames with a simple optimization
    # if annotated frames are close together,
    # read them sequentially instead of seeking each time.
    prev_frame = -1

    for frame_id in sorted(frame_annos.keys()):
        gap = frame_id - prev_frame

        if prev_frame == -1 or gap > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
        else:
            for _ in range(gap - 1):
                cap.read()
            ret, frame = cap.read()

        prev_frame = frame_id

        if not ret:
            continue

        frame = apply_enhancement(frame, gamma=0.7, clip_limit=2.0)

        stem = f"{video_key}_f{frame_id:06d}"
        dst_img = os.path.join(output_dir, 'images', split, f"{stem}.jpg")
        dst_lbl = os.path.join(output_dir, 'labels', split, f"{stem}.txt")

        if not os.path.exists(dst_img):
            cv2.imwrite(dst_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(dst_lbl, 'w') as f:
            f.write('\n'.join(frame_annos[frame_id]))

    cap.release()
    return dict(stats)



def process_dataset(json_path, videos_dir, output_dir, val_split=0.15, n_workers=6):
    """
    Main pipeline: reads JSON, splits videos into train/val,
    parallels video processing and writes YOLO dataset.

    Args:
        json_path:   Path to road_waymo_trainval_v1.0.json
        videos_dir:  Directory containing train_XXXXX.mp4 files
        output_dir:  Output directory for YOLO dataset
        val_split:   Fraction of videos for validation (default 0.15)
        n_workers:   Number of worker processes (default 6)
    """
    print(f"Loading JSON from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    agent_labels = data['agent_labels']

    # Create output directories
    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Random train/val split at video level
    all_videos = list(data['db'].keys())
    np.random.seed(42) # The answer to the Ultimate Question of Life, The Universe, Everything and hopefully processing.
    np.random.shuffle(all_videos)
    n_val = max(1, int(len(all_videos) * val_split))
    val_set = set(all_videos[:n_val])

    print(f"Videos total: {len(all_videos)}")
    print(f"Videos train: {len(all_videos) - n_val}")
    print(f"Videos val: {n_val}")
    print(f"Workers: {n_workers}")

    # Build argument list for workers
    worker_args = [
        (
            vk,
            data['db'][vk],
            videos_dir,
            output_dir,
            agent_labels,
            'val' if vk in val_set else 'train',
        )
        for vk in all_videos
    ]

    total = len(worker_args)
    done = 0
    total_stats = defaultdict(int)

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_video, worker_args):
            for k, v in result.items():
                total_stats[k] += v
            done += 1
            if done % 50 == 0 or done == total:
                print(f"Progress: {done}/{total} videos ({100*done/total:.1f}%)", flush=True)

    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:images/val\n")
        f.write(f"nc: {NUM_CLASSES}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\ndataset.yaml written to: {yaml_path}")
    print(f"\nClass distribution ")
    for cls in CLASS_NAMES:
        print(f"  {cls:15s}: {total_stats[cls]:>8,}")
    if total_stats['missing_videos']:
        print(f"\n  WARNING: {int(total_stats['missing_videos'])} videos not found on disk")

    return yaml_path


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ROAD-Waymo dataset for YOLOv8 training")
    parser.add_argument("--waymo_dir",  required=True,
                        help="Path to Waymo root directory (must contain videos/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--val_split",  type=float, default=0.15,
                        help="Fraction of videos for validation (default: 0.15)")
    parser.add_argument("--workers",    type=int,   default=6,
                        help="Number of worker processes (default: 6). ")
    args = parser.parse_args()

    process_dataset(
        json_path = os.path.join(args.waymo_dir, "road_waymo_trainval_v1.0.json"),
        videos_dir = os.path.join(args.waymo_dir, "videos"),
        output_dir = args.output_dir,
        val_split = args.val_split,
        n_workers = args.workers,
    )