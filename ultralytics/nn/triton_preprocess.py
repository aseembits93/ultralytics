"""Fused Triton preprocessing for YOLO: raw HWC uint8 BGR -> CHW fp16/fp32 RGB, letterboxed and /255."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _letterbox_bgr_to_chw_rgb(
    src_ptr,
    dst_ptr,
    src_h,
    src_w,
    dst_h: tl.constexpr,
    dst_w: tl.constexpr,
    new_h,
    new_w,
    pad_top,
    pad_left,
    inv_scale_y,
    inv_scale_x,
    pad_value: tl.constexpr,
    is_fp16: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    rows = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_rc = (rows[:, None] < dst_h) & (cols[None, :] < dst_w)

    ry = rows[:, None] - pad_top  # [BH,1]
    rx = cols[None, :] - pad_left  # [1,BW]

    in_img = (ry >= 0) & (ry < new_h) & (rx >= 0) & (rx < new_w)

    sy = (ry.to(tl.float32) + 0.5) * inv_scale_y - 0.5
    sx = (rx.to(tl.float32) + 0.5) * inv_scale_x - 0.5

    sy_clamped = tl.maximum(sy, 0.0)
    sx_clamped = tl.maximum(sx, 0.0)
    sy_clamped = tl.minimum(sy_clamped, (src_h - 1).to(tl.float32))
    sx_clamped = tl.minimum(sx_clamped, (src_w - 1).to(tl.float32))

    y0 = sy_clamped.to(tl.int32)
    x0 = sx_clamped.to(tl.int32)
    y1 = tl.minimum(y0 + 1, src_h - 1)
    x1 = tl.minimum(x0 + 1, src_w - 1)

    fy = sy_clamped - y0.to(tl.float32)
    fx = sx_clamped - x0.to(tl.float32)

    # Load BGR bytes: src stride is src_w*3 per row, 3 per pixel
    # B = ch 0, G = ch 1, R = ch 2 (we want R,G,B out)
    row_stride = src_w * 3
    # corners [BH,BW] for each channel
    off00_b = y0 * row_stride + x0 * 3 + 0
    off01_b = y0 * row_stride + x1 * 3 + 0
    off10_b = y1 * row_stride + x0 * 3 + 0
    off11_b = y1 * row_stride + x1 * 3 + 0

    p00_b = tl.load(src_ptr + off00_b, mask=in_img, other=pad_value).to(tl.float32)
    p01_b = tl.load(src_ptr + off01_b, mask=in_img, other=pad_value).to(tl.float32)
    p10_b = tl.load(src_ptr + off10_b, mask=in_img, other=pad_value).to(tl.float32)
    p11_b = tl.load(src_ptr + off11_b, mask=in_img, other=pad_value).to(tl.float32)

    p00_g = tl.load(src_ptr + off00_b + 1, mask=in_img, other=pad_value).to(tl.float32)
    p01_g = tl.load(src_ptr + off01_b + 1, mask=in_img, other=pad_value).to(tl.float32)
    p10_g = tl.load(src_ptr + off10_b + 1, mask=in_img, other=pad_value).to(tl.float32)
    p11_g = tl.load(src_ptr + off11_b + 1, mask=in_img, other=pad_value).to(tl.float32)

    p00_r = tl.load(src_ptr + off00_b + 2, mask=in_img, other=pad_value).to(tl.float32)
    p01_r = tl.load(src_ptr + off01_b + 2, mask=in_img, other=pad_value).to(tl.float32)
    p10_r = tl.load(src_ptr + off10_b + 2, mask=in_img, other=pad_value).to(tl.float32)
    p11_r = tl.load(src_ptr + off11_b + 2, mask=in_img, other=pad_value).to(tl.float32)

    w00 = (1.0 - fy) * (1.0 - fx)
    w01 = (1.0 - fy) * fx
    w10 = fy * (1.0 - fx)
    w11 = fy * fx

    b = w00 * p00_b + w01 * p01_b + w10 * p10_b + w11 * p11_b
    g = w00 * p00_g + w01 * p01_g + w10 * p10_g + w11 * p11_g
    r = w00 * p00_r + w01 * p01_r + w10 * p10_r + w11 * p11_r

    pad_f = tl.full(b.shape, pad_value, tl.float32)
    b = tl.where(in_img, b, pad_f)
    g = tl.where(in_img, g, pad_f)
    r = tl.where(in_img, r, pad_f)

    inv_255 = 1.0 / 255.0
    r = r * inv_255
    g = g * inv_255
    b = b * inv_255

    # Write to CHW, channels RGB
    plane = dst_h * dst_w
    base = rows[:, None] * dst_w + cols[None, :]
    if is_fp16:
        tl.store(dst_ptr + 0 * plane + base, r.to(tl.float16), mask=mask_rc)
        tl.store(dst_ptr + 1 * plane + base, g.to(tl.float16), mask=mask_rc)
        tl.store(dst_ptr + 2 * plane + base, b.to(tl.float16), mask=mask_rc)
    else:
        tl.store(dst_ptr + 0 * plane + base, r, mask=mask_rc)
        tl.store(dst_ptr + 1 * plane + base, g, mask=mask_rc)
        tl.store(dst_ptr + 2 * plane + base, b, mask=mask_rc)


def letterbox_preprocess(
    src_uint8_cuda: torch.Tensor,  # (H, W, 3) uint8 BGR on cuda
    dst: torch.Tensor,  # (1, 3, Hd, Wd) fp16 or fp32 on cuda
    pad_value: int = 114,
) -> None:
    """Fused letterbox + BGR->RGB + /255 + HWC->CHW into dst buffer."""
    assert src_uint8_cuda.is_cuda and src_uint8_cuda.dtype == torch.uint8
    assert src_uint8_cuda.ndim == 3 and src_uint8_cuda.shape[-1] == 3
    assert dst.is_cuda and dst.ndim == 4 and dst.shape[0] == 1 and dst.shape[1] == 3
    assert dst.dtype in (torch.float16, torch.float32)

    src_h, src_w = int(src_uint8_cuda.shape[0]), int(src_uint8_cuda.shape[1])
    dst_h, dst_w = int(dst.shape[2]), int(dst.shape[3])

    r = min(dst_h / src_h, dst_w / src_w)
    new_w = round(src_w * r)
    new_h = round(src_h * r)
    pad_left = (dst_w - new_w) // 2
    pad_top = (dst_h - new_h) // 2

    inv_scale_y = src_h / new_h
    inv_scale_x = src_w / new_w

    BLOCK_H, BLOCK_W = 16, 16
    grid = (triton.cdiv(dst_h, BLOCK_H), triton.cdiv(dst_w, BLOCK_W))
    _letterbox_bgr_to_chw_rgb[grid](
        src_uint8_cuda,
        dst,
        src_h,
        src_w,
        dst_h,
        dst_w,
        new_h,
        new_w,
        pad_top,
        pad_left,
        inv_scale_y,
        inv_scale_x,
        pad_value,
        dst.dtype == torch.float16,
        BLOCK_H,
        BLOCK_W,
    )
