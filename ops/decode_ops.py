from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .kv_cache_ops import kv_cache_update, normalize_positions, reference_kv_cache_update

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "decode_attention_optimized.metal"


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
        name="decode_attention_optimized_forward",
        input_names=["q", "K_cache", "V_cache", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _validate_inputs(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    lengths,
) -> tuple[mx.array, int, int, int, int]:
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
    MAX_S = K_cache.shape[1]
    if D > 128:
        raise ValueError(f"decode_attention currently supports D <= 128, got {D}")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"q dtype must be float16 or bfloat16, got {q.dtype}")
    if K_cache.dtype not in (mx.float16, mx.bfloat16) or V_cache.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K_cache/V_cache dtype must be float16 or bfloat16, got {K_cache.dtype}, {V_cache.dtype}")
    lengths_arr = normalize_positions(lengths if lengths is not None else MAX_S, B, MAX_S + 1)
    lengths_arr = mx.minimum(lengths_arr, mx.array(MAX_S, dtype=mx.int32))
    return lengths_arr, B, MAX_S, H, D


def reference_decode_attention(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    lengths=None,
    scale: Optional[float] = None,
    *,
    causal: bool = False,
) -> mx.array:
    lengths_arr, B, MAX_S, H, D = _validate_inputs(q, K_cache, V_cache, lengths)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if causal:
        # Decode attends to a prefix cache; `lengths` already defines the valid prefix.
        causal = False

    qf = q.astype(mx.float32)
    Kf = K_cache.astype(mx.float32)
    Vf = V_cache.astype(mx.float32)
    q_exp = qf.transpose(0, 2, 1, 3)  # [B,H,1,D]
    k_exp = Kf.transpose(0, 2, 3, 1)  # [B,H,D,S]
    v_exp = Vf.transpose(0, 2, 1, 3)  # [B,H,S,D]
    scores = mx.matmul(q_exp, k_exp) * float(scale)  # [B,H,1,S]

    positions = mx.arange(MAX_S).reshape(1, 1, 1, MAX_S)
    valid_mask = positions < lengths_arr.reshape(B, 1, 1, 1)
    neg_inf = mx.array(-1.0e9, dtype=scores.dtype)
    masked_scores = mx.where(valid_mask, scores, neg_inf)
    probs = mx.softmax(masked_scores, axis=-1)
    out = mx.matmul(probs, v_exp).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


def decode_attention(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    lengths=None,
    scale: Optional[float] = None,
    *,
    causal: bool = False,
    backend: str = "auto",
) -> mx.array:
    lengths_arr, B, MAX_S, H, D = _validate_inputs(q, K_cache, V_cache, lengths)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_decode_attention(q, K_cache, V_cache, lengths=lengths_arr, scale=scale, causal=causal)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = q.dtype
    source = _load_source()
    header = _make_header(dtype)
    kernel = _get_kernel(str(dtype), source, header)
    meta = mx.array([B, MAX_S, H, D, int(causal)], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    total_rows = B * H
    return kernel(
        inputs=[q, K_cache.astype(dtype), V_cache.astype(dtype), lengths_arr, meta, scale_arr],
        output_shapes=[(B, 1, H, D)],
        output_dtypes=[dtype],
        grid=(total_rows, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def reference_decode_step(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    position,
    scale: Optional[float] = None,
) -> tuple[mx.array, mx.array, mx.array]:
    K_updated, V_updated = reference_kv_cache_update(K_cache, V_cache, k_new, v_new, position)
    if isinstance(position, int):
        lengths = position + 1
    else:
        pos_arr = normalize_positions(position, K_cache.shape[0], K_cache.shape[1])
        lengths = pos_arr + 1
    out = reference_decode_attention(q, K_updated, V_updated, lengths=lengths, scale=scale, causal=False)
    return out, K_updated, V_updated


def decode_step(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    position,
    scale: Optional[float] = None,
    *,
    backend: str = "auto",
) -> tuple[mx.array, mx.array, mx.array]:
    backend_name = backend.lower()
    if backend_name == "reference":
        return reference_decode_step(q, k_new, v_new, K_cache, V_cache, position, scale=scale)

    updated_K, updated_V = kv_cache_update(K_cache, V_cache, k_new, v_new, position, backend=backend_name)
    if isinstance(position, int):
        lengths = position + 1
    else:
        lengths = normalize_positions(position, K_cache.shape[0], K_cache.shape[1]) + 1
    out = decode_attention(q, updated_K, updated_V, lengths=lengths, scale=scale, causal=False, backend=backend_name)
    return out, updated_K, updated_V
