"""
predict_yolo.py
==================
Inference script for T1: Spatiotemporal agent detection (YOLO + ByteTrack = agent tubes).

Usage (normal because it's random):
    python predict_yolo.py
"""

import cv2, random, glob
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

BEST_MODEL = "/home/est_licenciatura_julian.rodrig/Proyecto-final-vision/T1/waymo_yolo/runs/waymo_agent_det-2/weights/best.pt"
VAL_IMGS = "/home/est_licenciatura_julian.rodrig/Proyecto-final-vision/T1/waymo_yolo/images/val"
VIDEO_DIR = "/home/est_licenciatura_julian.rodrig/Proyecto-final-vision/T1/Waymo/videos"
VIDEO_OUT = "/home/est_licenciatura_julian.rodrig/Proyecto-final-vision/T1/demo_bytetrack.mp4"

model = YOLO(BEST_MODEL)

# Predict on random images
samples = random.sample(glob.glob(f"{VAL_IMGS}/*.jpg"), 6)
results = model.predict(samples, conf=0.3, imgsz=640, verbose=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, r in zip(axes.flat, results):
    img = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.savefig("val_predictions.png", dpi=150)
print("Guardado: val_predictions.png")

# Predict on random video
video_files = glob.glob(f"{VIDEO_DIR}/*.mp4")
video_path  = random.choice(video_files)
print(f"\nGenerando demo sobre: {video_path}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 10
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

frame_count = 0
MAX_FRAMES = 300

while frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    inv_gamma = 1.0 / 0.7
    table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)], dtype=np.uint8)
    frame = cv2.LUT(frame, table)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    frame = cv2.cvtColor(cv2.merge([clahe.apply(L), a, b]), cv2.COLOR_LAB2BGR)

    result = model.track(frame, conf=0.3, imgsz=640, verbose=False, persist = True, tracker="bytetrack.yaml")[0]
    writer.write(result.plot())
    frame_count += 1

cap.release()
writer.release()
print(f"Video guardado: {VIDEO_OUT}")