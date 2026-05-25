"""
Visualize inference results on video with bounding boxes and predicted activities.
"""
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path

def visualize(video_path, predictions_json, output_video, tracks_dir=None):
    # Load predictions
    with open(predictions_json, 'r') as f:
        preds = json.load(f)
    
    top_preds = preds['top_predictions'][:5]  # Top 5 predictions
    
    # Load tracks from numpy if available
    tracklets = {}
    if tracks_dir is None:
        video_dir = str(Path(video_path).parent)
        tracks_dir = os.path.join(video_dir, 'tracks', 'gt')
    
    if os.path.exists(tracks_dir):
        import numpy as np
        npy_files = sorted([f for f in os.listdir(tracks_dir) if f.endswith('.npy')])
        
        for file_idx, npy_file in enumerate(npy_files):
            data = np.load(os.path.join(tracks_dir, npy_file))
            num_frames, num_agents, _ = data.shape
            
            for frame_offset in range(num_frames):
                global_frame = file_idx * num_frames + frame_offset
                
                for agent_id in range(num_agents):
                    bbox = data[frame_offset, agent_id]
                    if bbox.sum() > 0:
                        if agent_id not in tracklets:
                            tracklets[agent_id] = {'bboxes': {}, 'trail': []}
                        tracklets[agent_id]['bboxes'][global_frame] = bbox
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        tracklets[agent_id]['trail'].append((cx, cy))
    
    print(f"Loaded {len(tracklets)} tracklets")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Colors for agents
    colors = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 255, 0),
        (0, 128, 255),
    ]
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw tracklets active in this frame
        for agent_id, data in tracklets.items():
            if frame_idx in data['bboxes']:
                bbox = data['bboxes'][frame_idx]
                x1, y1, x2, y2 = map(int, bbox)
                color = colors[agent_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Agent label
                label = f"Agent_{agent_id}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw prediction overlay panel (semi-transparent)
        overlay = frame.copy()
        panel_h = 30 + len(top_preds) * 25
        cv2.rectangle(overlay, (10, 10), (350, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Title
        cv2.putText(frame, "Top Predictions:", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Predictions
        y_offset = 60
        for pred in top_preds:
            activity = pred['activity']
            prob = pred['probability']
            text = f"{activity}: {prob:.2f}"
            color = (0, 255, 0) if prob > 0.5 else (255, 255, 0)
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 25
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", 
                   (width-200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    print(f"\nVideo saved to: {output_video}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--predictions', required=True, help='Predictions JSON')
    parser.add_argument('--output', default='output_visualization.mp4', help='Output video')
    args = parser.parse_args()
    
    visualize(args.video, args.predictions, args.output)
