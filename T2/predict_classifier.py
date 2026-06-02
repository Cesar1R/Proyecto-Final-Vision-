"""
predict_classifier.py
==================
T1 + T2 inference pipeline for ROAD++ Challenge.

Ad-hoc to T1 and T2 we generate agent tubes and event tubes, respectively:
T1: Spatiotemporal agent detection (YOLO + ByteTrack = agent tubes) [basically agent trajectories]
T2: Road event detection (agent tubes + action/loc classifier = event tubes)

Agent tube contains: {frame, bbox, class, class_name, conf}
Event tube contains: {track_id, agent, actions, locations, frames}

Usage:
    python predict_classifier.py --video /path/to/video.mp4
    python predict_classifier.py --video /path/to/video.mp4 --save_video
    python predict_classifier.py --videos_dir /path/to/videos/ --save_video
"""

import os
import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# Constants
CROP_SIZE    = 112
CONF_THRESH  = 0.3
IOU_THRESH   = 0.45
ACTION_THRESH= 0.4   # sigmoid threshold for action/loc prediction
LOC_THRESH   = 0.4

CLASS_NAMES  = ["Pedestrian", "Car", "MedVeh", "LargeVeh", "Cyclist", "TrafficLight"]
ACTION_NAMES = ["Red","Amber","Green","MovAway","MovTow","Mov","Rev","Brake",
                "Stop","IncatLft","IncatRht","HazLit","TurLft","TurRht",
                "MovRht","MovLft","Ovtak","Wait2X","XingFmLft","XingFmRht",
                "Xing","PushObj"]
LOC_NAMES    = ["VehLane","OutgoLane","OutgoCycLane","OutgoBusLane","IncomLane",
                "IncomCycLane","IncomBusLane","Pav","LftPav","RhtPav","Jun",
                "xing","BusStop","parking","LftParking","rightParking"]

# Colors for bboxes (BGR format)
AGENT_COLORS = {
    "Pedestrian" : (255, 255, 0),
    "Car" : (0, 255, 255),
    "MedVeh" : (0, 165, 255),
    "LargeVeh" : (0, 0, 255),
    "Cyclist" : (0, 255, 0),
    "TrafficLight": (255, 0, 255),
}


# Classifier model
class ActionLocClassifier(nn.Module):
    def __init__(self, n_actions=22, n_locs=16, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.action_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_actions))
        self.loc_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_locs))

    def forward(self, x):
        feat = self.backbone(x)
        return self.action_head(feat), self.loc_head(feat)


def load_classifier(ckpt_path, device):
    model = ActionLocClassifier().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Classifier loaded — epoch {ckpt['epoch']} "
          f"mean_mAP={ckpt['mean_map']:.4f}", flush=True)
    return model


# Crop transform
crop_tf = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def enhance_frame(frame_bgr, gamma=0.7, clip_limit=2.0):
    """Apply same enhancement as in prepare_dataset.py."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i/255.0)**inv_gamma)*255
                           for i in range(256)], dtype=np.uint8)
    frame = cv2.LUT(frame_bgr, table)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(L), a, b]), cv2.COLOR_LAB2BGR)


def classify_crop(frame_bgr, box, classifier, device):
    """Crop agent bbox from frame and run action/loc classifier."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return [], []

    crop = frame_bgr[y1:y2, x1:x2]
    crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = crop_tf(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        action_logits, loc_logits = classifier(tensor)
        action_probs = torch.sigmoid(action_logits)[0].cpu().numpy()
        loc_probs = torch.sigmoid(loc_logits)[0].cpu().numpy()

    actions = [ACTION_NAMES[i] for i, p in enumerate(action_probs) if p >= ACTION_THRESH]
    locs = [LOC_NAMES[i]    for i, p in enumerate(loc_probs) if p >= LOC_THRESH]

    return actions, locs


# Main pipeline
def run_pipeline(
    video_path,
    yolo_ckpt,
    classifier_ckpt,
    output_dir,
    device = None,
    save_video = False,
    max_frames = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print(f"Loading YOLO from: {yolo_ckpt}", flush=True)
    yolo = YOLO(yolo_ckpt)

    print(f"Loading classifier from: {classifier_ckpt}", flush=True)
    classifier = load_classifier(classifier_ckpt, device)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames= max_frames or total_fr

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing: {video_name} ({W}x{H} @ {fps:.1f}fps, "
          f"{total_fr} frames)", flush=True)

    # Video writer
    writer = None
    if save_video:
        out_video = os.path.join(output_dir, f"{video_name}_t2.mp4")
        writer    = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # T1 + T2 tubes
    agent_tubes = defaultdict(list)
    event_tubes = defaultdict(list)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply same enhancement as training data
        frame_enh = enhance_frame(frame)

        # T1: YOLO + ByteTrack 
        results = yolo.track(
            source = frame_enh,
            tracker = "bytetrack.yaml",
            conf = CONF_THRESH,
            iou = IOU_THRESH,
            imgsz = 640,
            persist = True,
            verbose = False,
        )[0]

        frame_events = []

        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.id.cpu().int().numpy(),
                results.boxes.cls.cpu().int().numpy(),
                results.boxes.conf.cpu().numpy(),
            ):
                tid = int(track_id)
                cls_name = CLASS_NAMES[int(cls_id)]

                # Agent tube entry
                agent_tubes[tid].append({
                    "frame" : frame_idx,
                    "bbox" : [float(x) for x in box],
                    "cls" : int(cls_id),
                    "cls_name": cls_name,
                    "conf" : float(conf),
                })

                # Classify action + location
                actions, locs = classify_crop(frame_enh, box, classifier, device)

                event = {
                    "frame" : frame_idx,
                    "bbox" : [float(x) for x in box],
                    "agent" : cls_name,
                    "actions" : actions,
                    "locations": locs,
                    "conf" : float(conf),
                }
                event_tubes[tid].append(event)
                frame_events.append((tid, event, box))

        # Draw on frame
        if writer is not None:
            vis = frame.copy()
            for tid, event, box in frame_events:
                x1, y1, x2, y2 = map(int, box)
                color = AGENT_COLORS.get(event["agent"], (255, 255, 255))

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Label = ID + agent
                label = f"ID:{tid} {event['agent']}"
                cv2.putText(vis, label, (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Actions and locations (if any)
                if event["actions"]:
                    act_str = " | ".join(event["actions"][:2])
                    cv2.putText(vis, act_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

                # Location
                if event["locations"]:
                    loc_str = event["locations"][0]
                    cv2.putText(vis, loc_str, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            writer.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{max_frames} | "
                  f"active tracks: {len(event_tubes)}", flush=True)

    cap.release()
    if writer:
        writer.release()
        print(f"Video saved: {out_video}", flush=True)

    # Reset ByteTrack state for next video
    yolo.predictor = None

    # Save tubes
    t1_path = os.path.join(output_dir, f"{video_name}_t1_tubes.json")
    t2_path = os.path.join(output_dir, f"{video_name}_t2_events.json")

    with open(t1_path, "w") as f:
        json.dump(dict(agent_tubes), f)
    with open(t2_path, "w") as f:
        json.dump(dict(event_tubes), f)

    print(f"\nResults for {video_name}:")
    print(f"  T1 tubes : {len(agent_tubes)}  → {t1_path}", flush=True)
    print(f"  T2 events: {len(event_tubes)}  → {t2_path}", flush=True)

    return dict(agent_tubes), dict(event_tubes)


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T1 + T2 pipeline on Waymo videos")
    parser.add_argument("--yolo_ckpt", required=True,
                        help="Path to YOLOv8 best.pt")
    parser.add_argument("--classifier_ckpt", required=True,
                        help="Path to best_classifier.pt")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save tubes and videos")
    parser.add_argument("--video", default=None,
                        help="Path to a single video file")
    parser.add_argument("--videos_dir", default=None,
                        help="Directory of videos to process all .mp4 files")
    parser.add_argument("--save_video", action="store_true",
                        help="Save annotated video with T2 labels")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames per video (default: all)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detected if not set)")
    args = parser.parse_args()

    if args.video:
        run_pipeline(
            video_path = args.video,
            yolo_ckpt = args.yolo_ckpt,
            classifier_ckpt = args.classifier_ckpt,
            output_dir = args.output_dir,
            device = args.device,
            save_video = args.save_video,
            max_frames = args.max_frames,
        )

    elif args.videos_dir:
        videos = sorted(
            os.path.join(args.videos_dir, f)
            for f in os.listdir(args.videos_dir)
            if f.endswith(".mp4")
        )
        print(f"Found {len(videos)} videos in {args.videos_dir}", flush=True)
        for vp in videos:
            run_pipeline(
                video_path = vp,
                yolo_ckpt = args.yolo_ckpt,
                classifier_ckpt = args.classifier_ckpt,
                output_dir = args.output_dir,
                device = args.device,
                save_video = args.save_video,
                max_frames = args.max_frames,
            )
    else:
        parser.error("Provide --video or --videos_dir")