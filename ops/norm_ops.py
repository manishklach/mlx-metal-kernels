from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx

from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source

_KERNEL_PATH = KERNEL_DIR / "rms_norm.metal"
_THREADS = 256


@lru_cache(maxsize=4)
def _get_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="rms_norm_forward",
        input_names=["x", "weight", "meta", "eps"],
        output_names=["y"],
        source=source,
        header=header,
    )


def _validate_inputs(x: mx.array, weight: mx.array) -> tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must be 3-D [B,S,D], got shape {x.shape}")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D [D], got shape {weight.shape}")
    B, S, D = x.shape
    if D != weight.shape[0]:
        raise ValueError(f"weight shape must be [D] with D={D}, got {weight.shape}")
    if B <= 0 or S <= 0 or D <= 0:
        raise ValueError(f"Invalid x shape: {x.shape}")
    if x.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"x dtype must be float16 or bfloat16, got {x.dtype}")
    if weight.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"weight dtype must be float16 or bfloat16, got {weight.dtype}")
    return B, S, D


def reference_rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    _, _, D = _validate_inputs(x, weight)
    variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
    inv_rms = mx.rsqrt(variance + float(eps))
    out = x.astype(mx.float32) * inv_rms * weight.astype(mx.float32).reshape(1, 1, D)
    return out.astype(x.dtype)


def rms_norm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-5,
    *,
    backend: str = "auto",
) -> mx.array:
    B, S, D = _validate_inputs(x, weight)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_rms_norm(x, weight, eps=eps)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = x.dtype
    source = load_metal_source(_KERNEL_PATH)
    header = _make_header(dtype)
    kernel = _get_kernel(str(dtype), source, header)
    meta = mx.array([B, S, D], dtype=mx.int32)
    eps_arr = mx.array([float(eps)], dtype=mx.float32)
    return kernel(
        inputs=[x, weight.astype(dtype), meta, eps_arr],
        output_shapes=[x.shape],
        output_dtypes=[dtype],
        grid=(B * S * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]
