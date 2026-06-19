from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .attention_ops import reference_attention

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "decode_attention.metal"


def _make_header(dtype: mx.Dtype, *, max_head_dim: int = 128) -> str:
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"decode_attention supports only float16/bfloat16, got {dtype}")
    return f"""
#include <metal_stdlib>
using namespace metal;
#define ELEM_TYPE {elem_type}
#define MAX_HEAD_DIM {max_head_dim}
"""


def _load_source() -> str:
    if not _KERNEL_PATH.exists():
        raise FileNotFoundError(f"Missing Metal kernel source: {_KERNEL_PATH}")
    return _KERNEL_PATH.read_text()


@lru_cache(maxsize=4)
def _get_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="decode_attention_forward",
        input_names=["q", "K_cache", "V_cache", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _validate_inputs(q: mx.array, K_cache: mx.array, V_cache: mx.array) -> tuple[int, int, int, int]:
    if q.ndim != 4 or K_cache.ndim != 4 or V_cache.ndim != 4:
        raise ValueError(
            f"q, K_cache, and V_cache must be 4-D, got {q.shape}, {K_cache.shape}, {V_cache.shape}"
        )
    if q.shape[1] != 1:
        raise ValueError(f"decode_attention expects q.shape[1] == 1, got {q.shape}")
    if K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must match, got {K_cache.shape}, {V_cache.shape}")
    if q.shape[0] != K_cache.shape[0] or q.shape[2] != K_cache.shape[2] or q.shape[3] != K_cache.shape[3]:
        raise ValueError(
            "q and cache tensors must agree on batch, heads, and head_dim. "
            f"Got q={q.shape}, K_cache={K_cache.shape}, V_cache={V_cache.shape}."
        )
    B, _, H, D = q.shape
    S = K_cache.shape[1]
    if D > 128:
        raise ValueError(f"decode_attention currently supports D <= 128, got {D}")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"q dtype must be float16 or bfloat16, got {q.dtype}")
    if K_cache.dtype not in (mx.float16, mx.bfloat16) or V_cache.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K_cache/V_cache dtype must be float16 or bfloat16, got {K_cache.dtype}, {V_cache.dtype}")
    return B, S, H, D


def reference_decode_attention(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    scale: Optional[float] = None,
) -> mx.array:
    _, _, _, D = _validate_inputs(q, K_cache, V_cache)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    return reference_attention(q, K_cache, V_cache, scale=scale, causal=False)


def decode_attention(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    scale: Optional[float] = None,
    *,
    backend: str = "auto",
) -> mx.array:
    B, S, H, D = _validate_inputs(q, K_cache, V_cache)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_decode_attention(q, K_cache, V_cache, scale=scale)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = q.dtype
    source = _load_source()
    header = _make_header(dtype)
    kernel = _get_kernel(str(dtype), source, header)
    meta = mx.array([B, S, H, D], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    total_rows = B * H
    return kernel(
        inputs=[q, K_cache.astype(dtype), V_cache.astype(dtype), meta, scale_arr],
        output_shapes=[(B, 1, H, D)],
        output_dtypes=[dtype],
        grid=(total_rows, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]
