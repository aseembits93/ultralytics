"""Fused Triton postprocessing: matmul(coeffs, protos) + crop + bilinear upsample + threshold.

Two kernel variants:
- `_fused_process_mask_kernel` — det-major grid (N, BH, BW). Cheap per-det but high launch overhead when N is large.
- `_fused_process_mask_kernel_tile_major` — tile-major grid (BH, BW). Fixed 144-block launch; each block iterates all N dets and early-exits per-det when conf <= thres. Useful for eliminating the mid-NMS conf-filter sync by running on all max_det predictions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_process_mask_kernel(
    coeffs_ptr,  # (N_DET, N_COEFF) fp32
    protos_ptr,  # (N_COEFF, PROTO_H, PROTO_W) fp32
    boxes_ptr,  # (N_DET, 4) xyxy in IMG space
    out_ptr,  # (N_DET, OUT_H, OUT_W) uint8
    H_RATIO,  # PROTO_H / IMG_H (float)
    W_RATIO,  # PROTO_W / IMG_W (float)
    N_COEFF: tl.constexpr,
    PROTO_H: tl.constexpr,
    PROTO_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    det_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    bw_id = tl.program_id(2)

    x1 = tl.load(boxes_ptr + det_id * 4 + 0) * W_RATIO
    y1 = tl.load(boxes_ptr + det_id * 4 + 1) * H_RATIO
    x2 = tl.load(boxes_ptr + det_id * 4 + 2) * W_RATIO
    y2 = tl.load(boxes_ptr + det_id * 4 + 3) * H_RATIO

    hs = bh_id * BLOCK_H + tl.arange(0, BLOCK_H)
    ws = bw_id * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (hs[:, None] < OUT_H) & (ws[None, :] < OUT_W)

    hp = (hs[:, None].to(tl.float32) + 0.5) * H_RATIO - 0.5
    wp = (ws[None, :].to(tl.float32) + 0.5) * W_RATIO - 0.5

    hp = tl.maximum(0.0, tl.minimum(hp, PROTO_H - 1.0))
    wp = tl.maximum(0.0, tl.minimum(wp, PROTO_W - 1.0))
    hp0 = hp.to(tl.int32)
    wp0 = wp.to(tl.int32)
    hp1 = tl.minimum(hp0 + 1, PROTO_H - 1)
    wp1 = tl.minimum(wp0 + 1, PROTO_W - 1)
    fy = hp - hp0.to(tl.float32)
    fx = wp - wp0.to(tl.float32)

    wp0f = wp0.to(tl.float32)
    wp1f = wp1.to(tl.float32)
    hp0f = hp0.to(tl.float32)
    hp1f = hp1.to(tl.float32)
    in00 = (wp0f >= x1) & (wp0f < x2) & (hp0f >= y1) & (hp0f < y2)
    in01 = (wp1f >= x1) & (wp1f < x2) & (hp0f >= y1) & (hp0f < y2)
    in10 = (wp0f >= x1) & (wp0f < x2) & (hp1f >= y1) & (hp1f < y2)
    in11 = (wp1f >= x1) & (wp1f < x2) & (hp1f >= y1) & (hp1f < y2)

    v00 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    v01 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    v10 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    v11 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    plane = PROTO_H * PROTO_W
    for c in tl.static_range(0, N_COEFF):
        coeff_c = tl.load(coeffs_ptr + det_id * N_COEFF + c)
        p_base = c * plane
        p00 = tl.load(protos_ptr + p_base + hp0 * PROTO_W + wp0, mask=out_mask, other=0.0)
        p01 = tl.load(protos_ptr + p_base + hp0 * PROTO_W + wp1, mask=out_mask, other=0.0)
        p10 = tl.load(protos_ptr + p_base + hp1 * PROTO_W + wp0, mask=out_mask, other=0.0)
        p11 = tl.load(protos_ptr + p_base + hp1 * PROTO_W + wp1, mask=out_mask, other=0.0)
        v00 += coeff_c * p00
        v01 += coeff_c * p01
        v10 += coeff_c * p10
        v11 += coeff_c * p11

    v00 = tl.where(in00, v00, 0.0)
    v01 = tl.where(in01, v01, 0.0)
    v10 = tl.where(in10, v10, 0.0)
    v11 = tl.where(in11, v11, 0.0)

    w00 = (1.0 - fy) * (1.0 - fx)
    w01 = (1.0 - fy) * fx
    w10 = fy * (1.0 - fx)
    w11 = fy * fx
    val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11

    out_val = (val > 0.0).to(tl.int8)
    offs = det_id * OUT_H * OUT_W + hs[:, None] * OUT_W + ws[None, :]
    tl.store(out_ptr + offs, out_val, mask=out_mask)


@triton.jit
def _fused_process_mask_kernel_tile_major(
    coeffs_ptr,  # (N_DET_MAX, N_COEFF) fp32
    protos_ptr,  # (N_COEFF, PROTO_H, PROTO_W) fp32
    boxes_ptr,  # (N_DET_MAX, 4) xyxy in IMG space
    confs_ptr,  # (N_DET_MAX,) fp32
    out_ptr,  # (N_DET_MAX, OUT_H, OUT_W) uint8 — pre-zeroed
    conf_thres,
    H_RATIO,
    W_RATIO,
    N_DET_MAX: tl.constexpr,
    N_COEFF: tl.constexpr,
    PROTO_H: tl.constexpr,
    PROTO_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    bh_id = tl.program_id(0)
    bw_id = tl.program_id(1)

    hs = bh_id * BLOCK_H + tl.arange(0, BLOCK_H)
    ws = bw_id * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (hs[:, None] < OUT_H) & (ws[None, :] < OUT_W)

    hp = (hs[:, None].to(tl.float32) + 0.5) * H_RATIO - 0.5
    wp = (ws[None, :].to(tl.float32) + 0.5) * W_RATIO - 0.5
    hp = tl.maximum(0.0, tl.minimum(hp, PROTO_H - 1.0))
    wp = tl.maximum(0.0, tl.minimum(wp, PROTO_W - 1.0))
    hp0 = hp.to(tl.int32)
    wp0 = wp.to(tl.int32)
    hp1 = tl.minimum(hp0 + 1, PROTO_H - 1)
    wp1 = tl.minimum(wp0 + 1, PROTO_W - 1)
    fy = hp - hp0.to(tl.float32)
    fx = wp - wp0.to(tl.float32)

    hp0f = hp0.to(tl.float32)
    hp1f = hp1.to(tl.float32)
    wp0f = wp0.to(tl.float32)
    wp1f = wp1.to(tl.float32)

    plane = PROTO_H * PROTO_W
    tile_hw = OUT_H * OUT_W

    # Pre-load protos corners for this tile once, shared across all dets
    tl.zeros((N_COEFF, BLOCK_H, BLOCK_W), dtype=tl.float32)
    # Triton doesn't easily support 3D arrays with constexpr first dim for loads; fall back to per-coeff inner loop
    # with per-det wrapping.

    zero = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int8)
    for det_id in range(0, N_DET_MAX):
        conf = tl.load(confs_ptr + det_id)
        offs = det_id * tile_hw + hs[:, None] * OUT_W + ws[None, :]
        if conf > conf_thres:
            x1 = tl.load(boxes_ptr + det_id * 4 + 0) * W_RATIO
            y1 = tl.load(boxes_ptr + det_id * 4 + 1) * H_RATIO
            x2 = tl.load(boxes_ptr + det_id * 4 + 2) * W_RATIO
            y2 = tl.load(boxes_ptr + det_id * 4 + 3) * H_RATIO

            in00 = (wp0f >= x1) & (wp0f < x2) & (hp0f >= y1) & (hp0f < y2)
            in01 = (wp1f >= x1) & (wp1f < x2) & (hp0f >= y1) & (hp0f < y2)
            in10 = (wp0f >= x1) & (wp0f < x2) & (hp1f >= y1) & (hp1f < y2)
            in11 = (wp1f >= x1) & (wp1f < x2) & (hp1f >= y1) & (hp1f < y2)

            v00 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            v01 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            v10 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            v11 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            for c in tl.static_range(0, N_COEFF):
                coeff_c = tl.load(coeffs_ptr + det_id * N_COEFF + c)
                p_base = c * plane
                p00 = tl.load(protos_ptr + p_base + hp0 * PROTO_W + wp0, mask=out_mask, other=0.0)
                p01 = tl.load(protos_ptr + p_base + hp0 * PROTO_W + wp1, mask=out_mask, other=0.0)
                p10 = tl.load(protos_ptr + p_base + hp1 * PROTO_W + wp0, mask=out_mask, other=0.0)
                p11 = tl.load(protos_ptr + p_base + hp1 * PROTO_W + wp1, mask=out_mask, other=0.0)
                v00 += coeff_c * p00
                v01 += coeff_c * p01
                v10 += coeff_c * p10
                v11 += coeff_c * p11

            v00 = tl.where(in00, v00, 0.0)
            v01 = tl.where(in01, v01, 0.0)
            v10 = tl.where(in10, v10, 0.0)
            v11 = tl.where(in11, v11, 0.0)

            w00 = (1.0 - fy) * (1.0 - fx)
            w01 = (1.0 - fy) * fx
            w10 = fy * (1.0 - fx)
            w11 = fy * fx
            val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11

            out_val = (val > 0.0).to(tl.int8)
            tl.store(out_ptr + offs, out_val, mask=out_mask)
        else:
            tl.store(out_ptr + offs, zero, mask=out_mask)


def fused_process_mask(
    protos: torch.Tensor,
    coeffs: torch.Tensor,
    boxes: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Equivalent to process_mask(protos, coeffs, boxes, shape, upsample=True).byte()."""
    assert protos.is_cuda and coeffs.is_cuda and boxes.is_cuda
    assert protos.ndim == 3 and coeffs.ndim == 2 and boxes.ndim == 2
    C, Ph, Pw = protos.shape
    N, Cc = coeffs.shape
    assert Cc == C
    out_h, out_w = int(shape[0]), int(shape[1])

    out = torch.empty((N, out_h, out_w), dtype=torch.uint8, device=protos.device)
    if N == 0:
        return out

    protos_c = protos.contiguous()
    coeffs_c = coeffs.contiguous()
    boxes_c = boxes.contiguous()

    h_ratio = Ph / out_h
    w_ratio = Pw / out_w

    BLOCK_H, BLOCK_W = 16, 16
    grid = (N, triton.cdiv(out_h, BLOCK_H), triton.cdiv(out_w, BLOCK_W))
    _fused_process_mask_kernel[grid](
        coeffs_c,
        protos_c,
        boxes_c,
        out,
        h_ratio,
        w_ratio,
        C,
        Ph,
        Pw,
        out_h,
        out_w,
        BLOCK_H,
        BLOCK_W,
    )
    return out


def fused_process_mask_full(
    protos: torch.Tensor,
    coeffs: torch.Tensor,
    boxes: torch.Tensor,
    confs: torch.Tensor,
    conf_thres: float,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Tile-major kernel: runs on all N detections, zeros rows with conf<=thres.

    Output is pre-zeroed so low-conf rows remain all-zero and can be dropped downstream via amax>0.
    """
    assert protos.is_cuda and coeffs.is_cuda and boxes.is_cuda and confs.is_cuda
    C, Ph, Pw = protos.shape
    N, Cc = coeffs.shape
    assert Cc == C
    out_h, out_w = int(shape[0]), int(shape[1])

    out = torch.empty((N, out_h, out_w), dtype=torch.uint8, device=protos.device)
    if N == 0:
        return out

    BLOCK_H, BLOCK_W = 16, 16
    grid = (triton.cdiv(out_h, BLOCK_H), triton.cdiv(out_w, BLOCK_W))
    _fused_process_mask_kernel_tile_major[grid](
        coeffs.contiguous(),
        protos.contiguous(),
        boxes.contiguous(),
        confs.contiguous(),
        out,
        float(conf_thres),
        Ph / out_h,
        Pw / out_w,
        N,
        C,
        Ph,
        Pw,
        out_h,
        out_w,
        BLOCK_H,
        BLOCK_W,
    )
    return out
