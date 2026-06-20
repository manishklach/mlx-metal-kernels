from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source
from .kv_cache_ops import kv_cache_update, normalize_positions, reference_kv_cache_update

_KERNEL_DIR = KERNEL_DIR
_KERNEL_PATH = _KERNEL_DIR / "decode_attention_optimized.metal"
_KERNEL_PATH_THREADGROUP = _KERNEL_DIR / "decode_attention_threadgroup.metal"
_SPECIALIZED_KERNELS = {
    "metal_d64": _KERNEL_DIR / "decode_attention_d64.metal",
    "metal_d128": _KERNEL_DIR / "decode_attention_d128.metal",
}
_THREADGROUP_THREADS = 128


def _make_header(dtype: mx.Dtype, *, max_head_dim: int = 128, fixed_head_dim: int | None = None) -> str:
    kwargs = dict(MAX_HEAD_DIM=max_head_dim, TG_THREADS=_THREADGROUP_THREADS)
    if fixed_head_dim is not None:
        kwargs["HEAD_DIM"] = fixed_head_dim
    return make_metal_header(dtype, **kwargs)


@lru_cache(maxsize=4)
def _get_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_cache", "V_cache", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _resolve_backend(backend_name: str, D: int) -> str:
    if backend_name == "auto":
        if os.environ.get("MLX_METAL_CI_SAFE_MODE", "0") == "1":
            return "reference"
        if os.environ.get("MLX_METAL_USE_THREADGROUP_ATTENTION", "0") == "1":
            return "metal_threadgroup"
        if os.environ.get("MLX_METAL_USE_SPECIALIZED", "0") == "1":
            if D == 64:
                return "metal_d64"
            if D == 128:
                return "metal_d128"
        return "metal"
    if backend_name == "metal_d64" and D != 64:
        raise ValueError(f"backend='metal_d64' requires D == 64, got D={D}")
    if backend_name == "metal_d128" and D != 128:
        raise ValueError(f"backend='metal_d128' requires D == 128, got D={D}")
    return backend_name


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
            f"Got q={q.shape}, K_cache={K_cache.shape}, V_cache={V_cache.shape}. "
            "For GQA/MQA decode with Hq != Hkv, use ops.gqa_ops.reference_gqa_decode_attention."
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

    backend_name = _resolve_backend(backend.lower(), D)
    if backend_name == "reference":
        return reference_decode_attention(q, K_cache, V_cache, lengths=lengths_arr, scale=scale, causal=causal)
    if backend_name not in ("metal", "metal_threadgroup", "metal_d64", "metal_d128"):
        raise ValueError("backend must be one of 'reference', 'metal', 'metal_threadgroup', 'metal_d64', 'metal_d128', 'auto'")

    dtype = q.dtype
    total_rows = B * H
    if backend_name == "metal":
        kernel_path = _KERNEL_PATH
        kernel_name = "decode_attention_optimized_forward"
        header = _make_header(dtype)
        grid = (total_rows, 1, 1)
        threadgroup = (1, 1, 1)
    elif backend_name == "metal_threadgroup":
        kernel_path = _KERNEL_PATH_THREADGROUP
        kernel_name = "decode_attention_threadgroup_forward"
        header = _make_header(dtype)
        grid = (total_rows * _THREADGROUP_THREADS, 1, 1)
        threadgroup = (_THREADGROUP_THREADS, 1, 1)
    else:
        kernel_path = _SPECIALIZED_KERNELS[backend_name]
        kernel_name = f"decode_attention_{D}_forward"
        header = _make_header(dtype, fixed_head_dim=D)
        grid = (total_rows, 1, 1)
        threadgroup = (1, 1, 1)
    source = load_metal_source(kernel_path)
    kernel = _get_kernel(kernel_name, str(dtype), source, header)
    meta = mx.array([B, MAX_S, H, D, int(causal)], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    return kernel(
        inputs=[q, K_cache.astype(dtype), V_cache.astype(dtype), lengths_arr, meta, scale_arr],
        output_shapes=[(B, 1, H, D)],
        output_dtypes=[dtype],
        grid=grid,
        threadgroup=threadgroup,
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
