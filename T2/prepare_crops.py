"""
prepare_crops.py
================
Generates agent crops from existing YOLO dataset for T2 classifier training.
"""

import os, json, cv2
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
import argparse

CROP_SIZE = 112
ACTION_NAMES = ["Red","Amber","Green","MovAway","MovTow","Mov","Rev","Brake",
                "Stop","IncatLft","IncatRht","HazLit","TurLft","TurRht",
                "MovRht","MovLft","Ovtak","Wait2X","XingFmLft","XingFmRht",
                "Xing","PushObj"]

LOC_NAMES = ["VehLane","OutgoLane","OutgoCycLane","OutgoBusLane","IncomLane",
                "IncomCycLane","IncomBusLane","Pav","LftPav","RhtPav","Jun",
                "xing","BusStop","parking","LftParking","rightParking"]

N_ACTIONS = len(ACTION_NAMES)
N_LOCS = len(LOC_NAMES)

AGENT_GROUPS = {
    "Ped":0, "Car":1, "SmalVeh":1, "MedVeh":2,
    "LarVeh":3, "Bus":3, "EmVeh":3, "Cyc":4, "Mobike":4, "TL":5
}


def load_json_index(json_path):
    """Build lookup list of {cls, action_ids, loc_ids, bbox}"""
    print("Building JSON index...")
    with open(json_path) as f:
        data = json.load(f)

    agent_labels = data['agent_labels']
    index = defaultdict(list)

    for video_key, video_data in data['db'].items():
        for frame_key, frame_data in video_data['frames'].items():
            if not frame_data.get('annotated', False):
                continue
            if 'annos' not in frame_data:
                continue

            stem = f"{video_key}_f{int(frame_key):06d}"

            for anno in frame_data['annos'].values():
                if not anno.get('agent_ids'):
                    continue
                agent_name = agent_labels[anno['agent_ids'][0]]
                if agent_name not in AGENT_GROUPS:
                    continue

                index[stem].append({
                    'cls': AGENT_GROUPS[agent_name],
                    'action_ids': anno.get('action_ids', []),
                    'loc_ids': anno.get('loc_ids', []),
                    'bbox': anno['box'],
                })

    print(f"Index built: {len(index)} annotated frames", flush=True)
    return dict(index)

# Function to process one image and extract crops, meant for parallel.
def process_single_image(args):
    """Extracts all crops from one image."""
    fname, img_dir, out_dir, annos = args
    stem = fname[:-4]
    img_bgr = cv2.imread(os.path.join(img_dir, fname))
    if img_bgr is None:
        return {}

    H, W = img_bgr.shape[:2]
    labels  = {}

    for j, anno in enumerate(annos):
        x1, y1, x2, y2 = anno['bbox']
        px1 = max(0, int(x1 * W))
        py1 = max(0, int(y1 * H))
        px2 = min(W, int(x2 * W))
        py2 = min(H, int(y2 * H))

        if px2 <= px1 or py2 <= py1:
            continue

        crop = img_bgr[py1:py2, px1:px2]
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

        action_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        loc_vec = np.zeros(N_LOCS,    dtype=np.float32)
        for aid in anno['action_ids']:
            action_vec[aid] = 1.0
        for lid in anno['loc_ids']:
            loc_vec[lid]    = 1.0

        crop_stem = f"{stem}_a{j}"
        cv2.imwrite(
            os.path.join(out_dir, f"{crop_stem}.jpg"),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        labels[crop_stem] = {
            'cls': anno['cls'],
            'action_vec': action_vec.tolist(),
            'loc_vec': loc_vec.tolist(),
        }

    return labels


def extract_crops(split, json_index, waymo_yolo, crops_dir, n_workers):
    img_dir = os.path.join(waymo_yolo, 'images', split)
    out_dir = os.path.join(crops_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    worker_args = []
    for fname in img_files:
        stem = fname[:-4]
        annos = json_index.get(stem)
        if annos:
            worker_args.append((fname, img_dir, out_dir, annos))

    print(f"\n{split}: {len(worker_args)}/{len(img_files)} images have annotations")
    print(f"Workers: {n_workers}", flush=True)

    total = len(worker_args)
    done = 0
    labels = {}

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_image, worker_args):
            labels.update(result)
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"Progress: {done}/{total} images ({100*done/total:.1f}%)",
                      flush=True)

    labels_path = os.path.join(crops_dir, f"{split}_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(labels, f)

    print(f"Crops: {len(labels):,}  |  Labels saved: {labels_path}", flush=True)
    return len(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract agent crops for T2 classifier")
    parser.add_argument("--waymo_yolo", required=True,
                        help="Path to prepared YOLO dataset (waymo_yolo/)")
    parser.add_argument("--waymo_json", required=True,
                        help="Path to road_waymo_trainval_v1.0.json")
    parser.add_argument("--crops_dir",  required=True,
                        help="Output directory for crops")
    parser.add_argument("--workers",    type=int, default=6,
                        help="Number of worker processes (default: 6)")
    args = parser.parse_args()

    json_index = load_json_index(args.waymo_json)

    for split in ('train', 'val'):
        n = extract_crops(
            split      = split,
            json_index = json_index,
            waymo_yolo = args.waymo_yolo,
            crops_dir  = args.crops_dir,
            n_workers  = args.workers,
        )

    print("\n=== Crop dataset ready ===")
    print(f"Location: {args.crops_dir}")