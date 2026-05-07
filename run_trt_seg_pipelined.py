"""Depth-2 pipelined benchmark for the TRT segmentation path.

Instead of `model.predict(frame)` which serializes preprocess -> forward -> postprocess,
this script submits frame N+1's preprocess + forward while frame N's postprocess runs
on a dedicated stream. Requires `ULTRALYTICS_TRT_PIPELINE_POST=1` so the backend clones
its outputs per-call (ring buffer) and exposes `post_stream` / `produce_event`.

Produces the same detection set as the synchronous baseline, frame-for-frame.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("ULTRALYTICS_TRT_CUDA_GRAPH", "1")
os.environ.setdefault("ULTRALYTICS_TRT_PIPELINE_POST", "1")
os.environ.setdefault("ULTRALYTICS_PROFILE_SYNC", "0")

from ultralytics import YOLO
from ultralytics.nn.triton_postprocess import fused_process_mask
from ultralytics.nn.triton_preprocess import letterbox_preprocess
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
print(f"Loaded {len(frames)} frames at {frames[0].shape}")

if not Path(ENGINE).exists():
    YOLO(PT_MODEL).export(format="engine", device=0, imgsz=frames[0].shape[0])

model = YOLO(ENGINE, task="segment")
model.predict(frames[0], device=0, verbose=False, conf=CONF)

backend = model.predictor.model.backend
assert backend.post_stream is not None, "pipelining not enabled"

device = backend.device
input_binding = backend.input_tensor
img_shape_hw = tuple(input_binding.shape[2:])
orig_h, orig_w, _ = frames[0].shape

# Preallocated staging buffers (one of each; preprocess is a strict predecessor of
# forward on the default stream so we never need a ring here).
src_pinned = torch.empty(frames[0].shape, dtype=torch.uint8, pin_memory=True)
src_gpu = torch.empty(frames[0].shape, dtype=torch.uint8, device=device)


def submit_inference(frame_np: np.ndarray):
    """Upload a frame, run triton preprocess, replay the graph (async), return clone tuple."""
    src_pinned.copy_(torch.from_numpy(frame_np))
    src_gpu.copy_(src_pinned, non_blocking=True)
    letterbox_preprocess(src_gpu, input_binding)
    return backend.forward(input_binding)


def run_postprocess(preds):
    """Run postprocess on backend.post_stream. Returns (boxes, masks) tensors.

    Reads from the cloned output buffers so the next forward can overwrite the live TRT bindings without corrupting this
    work.
    """
    post_stream = backend.post_stream
    post_stream.wait_event(backend.produce_event)
    with torch.cuda.stream(post_stream):
        dets_batched = preds[0]  # (1, 300, 38) clone
        protos = preds[1][0]  # (32, 48, 48) clone
        filtered = non_max_suppression(dets_batched, CONF, 0.45, None, end2end=True, max_det=300)[0]
        if filtered.shape[0] == 0:
            return filtered[:, :6], None
        masks = fused_process_mask(protos, filtered[:, 6:], filtered[:, :4], img_shape_hw)
        keep = masks.amax((-2, -1)) > 0
        if not bool(keep.all()):
            filtered = filtered[keep]
            masks = masks[keep]
        boxes = filtered[:, :6]
    return boxes, masks


# Warmup the pipeline path: submit+drain twice so triton kernels are JIT-compiled
# and the clone ring is primed before we measure.
with torch.inference_mode():
    for _ in range(2):
        p = submit_inference(frames[0])
        run_postprocess(p)
torch.cuda.synchronize()


def run_pipelined(frames_list):
    """Depth-2 pipeline: submit(N+1) before reading(N)."""
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
    # Final sync so wall-time includes all GPU work.
    torch.cuda.current_stream().wait_stream(backend.post_stream)
    torch.cuda.synchronize()
    return out_boxes, out_masks


with torch.inference_mode():
    t0 = time.perf_counter()
    boxes_list, masks_list = run_pipelined(frames)
    elapsed = time.perf_counter() - t0
fps = len(frames) / elapsed
print(f"Processed {len(frames)} frames in {elapsed:.3f}s -> {fps:.2f} FPS")

# Print a sanity check for a few frames (forces DtoH on boxes only).
for idx in [0, 50, 100, 200, 400]:
    b = boxes_list[idx].cpu().tolist()
    print(f"frame {idx}: {len(b)} boxes, conf={[round(x[4], 4) for x in b][:3]}")
