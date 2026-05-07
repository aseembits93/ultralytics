"""Depth-2 pipelined benchmark for the TRT segmentation path.

Two modes, selected by `MODE`:
- `MODE=baseline`: synchronous `model.predict(frame)` with every optimization disabled
  (no CUDA graph, no triton pre/post, no pipelining). Reproduces the ~298 FPS number.
- `MODE=best` (default): depth-2 pipeline — submit frame N+1's preprocess+forward
  while frame N's postprocess runs on a dedicated stream. Reproduces ~402 FPS.

Both modes time 538 frames of `vehicles_312px.mp4` (pre-loaded before the model loads).
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

MODE = os.environ.get("MODE", "best").lower()

if MODE == "baseline":
    os.environ["ULTRALYTICS_TRT_CUDA_GRAPH"] = "0"
    os.environ["ULTRALYTICS_TRT_PIPELINE_POST"] = "0"
    os.environ["ULTRALYTICS_TRITON_PRE"] = "0"
    os.environ["ULTRALYTICS_TRITON_POST"] = "0"
    os.environ["ULTRALYTICS_PROFILE_SYNC"] = "1"
else:
    os.environ.setdefault("ULTRALYTICS_TRT_CUDA_GRAPH", "1")
    os.environ.setdefault("ULTRALYTICS_TRT_PIPELINE_POST", "1")
    os.environ.setdefault("ULTRALYTICS_PROFILE_SYNC", "0")

from ultralytics import YOLO
from ultralytics.nn.triton_preprocess import letterbox_preprocess
from ultralytics.nn.triton_postprocess import fused_process_mask
from ultralytics.utils.nms import non_max_suppression

VIDEO = "vehicles_312px.mp4"
PT_MODEL = "yolo26n-seg.pt"
ENGINE = "yolo26n-seg.engine"
CONF = 0.25

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
cap.release()
print(f"[mode={MODE}] Loaded {len(frames)} frames at {frames[0].shape}")

if not Path(ENGINE).exists():
    YOLO(PT_MODEL).export(format="engine", device=0, imgsz=frames[0].shape[0])

model = YOLO(ENGINE, task="segment")
# Warmup — one real predict so the AutoBackend is built and TRT is loaded.
model.predict(frames[0], device=0, verbose=False, conf=CONF)


def run_baseline():
    """Synchronous predict loop — reproduces the ~298 FPS number."""
    t0 = time.perf_counter()
    for f in frames:
        model.predict(f, device=0, verbose=False, conf=CONF)
    elapsed = time.perf_counter() - t0
    return elapsed


def run_best():
    """Depth-2 pipelined loop — reproduces the ~402 FPS number."""
    backend = model.predictor.model.backend
    assert backend.post_stream is not None, "pipelining not enabled"
    device = backend.device
    input_binding = backend.input_tensor
    img_shape_hw = tuple(input_binding.shape[2:])

    src_pinned = torch.empty(frames[0].shape, dtype=torch.uint8, pin_memory=True)
    src_gpu = torch.empty(frames[0].shape, dtype=torch.uint8, device=device)

    def submit_inference(frame_np: np.ndarray):
        src_pinned.copy_(torch.from_numpy(frame_np))
        src_gpu.copy_(src_pinned, non_blocking=True)
        letterbox_preprocess(src_gpu, input_binding)
        return backend.forward(input_binding)

    def run_postprocess(preds):
        post_stream = backend.post_stream
        post_stream.wait_event(backend.produce_event)
        with torch.cuda.stream(post_stream):
            dets_batched = preds[0]
            protos = preds[1][0]
            filtered = non_max_suppression(
                dets_batched, CONF, 0.45, None, end2end=True, max_det=300
            )[0]
            if filtered.shape[0] == 0:
                return filtered[:, :6], None
            masks = fused_process_mask(
                protos, filtered[:, 6:], filtered[:, :4], img_shape_hw
            )
            keep = masks.amax((-2, -1)) > 0
            if not bool(keep.all()):
                filtered = filtered[keep]
                masks = masks[keep]
            boxes = filtered[:, :6]
        return boxes, masks

    with torch.inference_mode():
        # Prime the clone ring and triton JIT before timing.
        for _ in range(2):
            p = submit_inference(frames[0])
            run_postprocess(p)
    torch.cuda.synchronize()

    def run_pipelined(frames_list):
        n = len(frames_list)
        out_boxes = [None] * n
        out_masks = [None] * n
        prev = submit_inference(frames_list[0])
        for i in range(1, n):
            cur = submit_inference(frames_list[i])
            b, m = run_postprocess(prev)
            out_boxes[i - 1] = b
            out_masks[i - 1] = m
            prev = cur
        b, m = run_postprocess(prev)
        out_boxes[n - 1] = b
        out_masks[n - 1] = m
        torch.cuda.current_stream().wait_stream(backend.post_stream)
        torch.cuda.synchronize()
        return out_boxes, out_masks

    with torch.inference_mode():
        t0 = time.perf_counter()
        run_pipelined(frames)
        elapsed = time.perf_counter() - t0
    return elapsed


elapsed = run_baseline() if MODE == "baseline" else run_best()
fps = len(frames) / elapsed
print(f"[mode={MODE}] Processed {len(frames)} frames in {elapsed:.3f}s -> {fps:.2f} FPS")
