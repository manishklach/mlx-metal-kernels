from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source

_KERNEL_PATH = KERNEL_DIR / "rope.metal"
_THREADS = 256


@lru_cache(maxsize=4)
def _get_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="rope_forward",
        input_names=["x", "cos", "sin", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


def _validate_inputs(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_offset: int,
) -> tuple[int, int, int, int, int]:
    if x.ndim != 4:
        raise ValueError(f"x must be 4-D [B,S,H,D], got shape {x.shape}")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError(f"cos and sin must be 2-D [S_total,D/2], got {cos.shape}, {sin.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have identical shapes, got {cos.shape}, {sin.shape}")
    B, S, H, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"RoPE requires even D, got D={D}")
    if cos.shape[1] != D // 2:
        raise ValueError(f"cos/sin last dim must be D/2={D // 2}, got {cos.shape[1]}")
    if position_offset < 0:
        raise ValueError("position_offset must be non-negative")
    if position_offset + S > cos.shape[0]:
        raise ValueError(
            f"position_offset + S must fit in cos/sin rows, got offset={position_offset}, S={S}, rows={cos.shape[0]}"
        )
    if x.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"x dtype must be float16 or bfloat16, got {x.dtype}")
    if cos.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError(f"cos dtype must be float16/bfloat16/float32, got {cos.dtype}")
    if sin.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError(f"sin dtype must be float16/bfloat16/float32, got {sin.dtype}")
    return B, S, H, D, cos.shape[0]


def reference_apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_offset: int = 0,
) -> mx.array:
    _, S, _, D, _ = _validate_inputs(x, cos, sin, position_offset)
    cos_slice = cos[position_offset:position_offset + S].astype(mx.float32).reshape(1, S, 1, D // 2)
    sin_slice = sin[position_offset:position_offset + S].astype(mx.float32).reshape(1, S, 1, D // 2)
    x_even = x[..., 0::2].astype(mx.float32)
    x_odd = x[..., 1::2].astype(mx.float32)
    y_even = x_even * cos_slice - x_odd * sin_slice
    y_odd = x_even * sin_slice + x_odd * cos_slice
    y = mx.stack([y_even, y_odd], axis=-1).reshape(x.shape)
    return y.astype(x.dtype)


def apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_offset: int = 0,
    *,
    backend: str = "auto",
) -> mx.array:
    B, S, H, D, cos_rows = _validate_inputs(x, cos, sin, position_offset)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_apply_rope(x, cos, sin, position_offset=position_offset)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = x.dtype
    source = load_metal_source(_KERNEL_PATH)
    header = _make_header(dtype)
    kernel = _get_kernel(str(dtype), source, header)
    pair_count = B * S * H * (D // 2)
    meta = mx.array([B, S, H, D, cos_rows, position_offset], dtype=mx.int32)
    return kernel(
        inputs=[x, cos.astype(mx.float32), sin.astype(mx.float32), meta],
        output_shapes=[x.shape],
        output_dtypes=[dtype],
        grid=(pair_count, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]
