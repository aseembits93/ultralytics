import time
from pathlib import Path

import cv2

from ultralytics import YOLO

VIDEO = "vehicles_312px.mp4"
PT_MODEL = "yolo26n-seg.pt"
ENGINE = "yolo26n-seg.engine"

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames at {frames[0].shape}")

if not Path(ENGINE).exists():
    YOLO(PT_MODEL).export(format="engine", device=0, imgsz=frames[0].shape[0])

model = YOLO(ENGINE, task="segment")

model.predict(frames[0], device=0, verbose=False)

pre = inf = post = 0.0
t0 = time.perf_counter()
for f in frames:
    r = model.predict(f, device=0, verbose=False)[0]
    pre += r.speed["preprocess"]
    inf += r.speed["inference"]
    post += r.speed["postprocess"]
elapsed = time.perf_counter() - t0

n = len(frames)
fps = n / elapsed
print(f"Processed {n} frames in {elapsed:.3f}s -> {fps:.2f} FPS")
print(f"Per-frame avg (ms): preprocess={pre / n:.2f}  inference={inf / n:.2f}  postprocess={post / n:.2f}")
