"""
MLX bindings for correctness-first FlashAttention-style Metal kernels.

This module intentionally starts with a simple streaming kernel:
    O = softmax(Q K^T * scale) V
without materializing the full [S, S] attention matrix.

Layout: BSHD, i.e. Q/K/V shape [batch, sequence, heads, head_dim].
Supported dtypes: mx.float16 and mx.bfloat16.
Supported head_dim: <= 128 by default.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
_DEFAULT_KERNEL = _KERNEL_DIR / "fast_attention.metal"
_ROW_PARALLEL_KERNEL = _KERNEL_DIR / "fast_attention_row_parallel.metal"
_TILED_KV_KERNEL = _KERNEL_DIR / "fast_attention_tiled_kv.metal"
_SPECIALIZED_BASELINE_KERNELS = {
    "baseline_d64": _KERNEL_DIR / "fast_attention_d64.metal",
    "baseline_d128": _KERNEL_DIR / "fast_attention_d128.metal",
}
_MAX_HEAD_DIM = 128
_ROW_PARALLEL_THREADS = 128
_TILED_KV_THREADS = 64
_KV_TILE = 16


def _make_header(dtype: mx.Dtype, *, max_head_dim: int = _MAX_HEAD_DIM, fixed_head_dim: int | None = None) -> str:
    """Emit header code shared by MLX custom Metal kernels."""
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"fast_attention supports only float16/bfloat16, got {dtype}")

    fixed_dim_line = f"#define HEAD_DIM {fixed_head_dim}" if fixed_head_dim is not None else ""
    return f"""
#include <metal_stdlib>
using namespace metal;
#define ELEM_TYPE {elem_type}
#define MAX_HEAD_DIM {max_head_dim}
#define KV_TILE {_KV_TILE}
{fixed_dim_line}
"""


def _load_kernel_source(path: Path = _DEFAULT_KERNEL) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Metal kernel source: {path}."
        )
    return path.read_text()


@lru_cache(maxsize=8)
def _get_fast_attention_kernel(
    kernel_name: str,
    dtype_name: str,
    source: str,
    header: str,
):
    # MLX caches compiled kernels internally too, but this avoids rebuilding the
    # Python wrapper object on every call.
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["Q", "K", "V", "meta", "scale"],
        output_names=["O"],
        source=source,
        header=header,
    )


def _validate_qkv(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    require_same_sequence: bool = True,
) -> tuple[int, int, int, int, int]:
    if Q.ndim != 4:
        raise ValueError(f"Q must be 4-D [B,S,H,D], got shape {Q.shape}")
    if K.ndim != 4 or V.ndim != 4:
        raise ValueError(f"K and V must be 4-D [B,S,H,D], got {K.shape}, {V.shape}")
    if K.shape != V.shape:
        raise ValueError(
            f"K and V must have identical shapes. Got K={K.shape}, V={V.shape}."
        )
    if Q.shape[0] != K.shape[0] or Q.shape[2] != K.shape[2] or Q.shape[3] != K.shape[3]:
        raise ValueError(
            "Q, K, and V must agree on batch, heads, and head_dim. "
            f"Got Q={Q.shape}, K={K.shape}, V={V.shape}."
        )
    if require_same_sequence and (Q.shape[1] != K.shape[1] or Q.shape[1] != V.shape[1]):
        raise ValueError(
            "This Metal attention path currently supports self-attention only: "
            "Q, K, and V must have the same sequence length. "
            f"Got Q={Q.shape}, K={K.shape}, V={V.shape}."
        )

    B, Sq, H, D = Q.shape
    Sk = K.shape[1]
    if D > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim D must be <= {_MAX_HEAD_DIM}, got {D}")
    if D <= 0 or Sq <= 0 or Sk <= 0 or H <= 0 or B <= 0:
        raise ValueError(f"Invalid Q/K/V shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    if Q.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"Q dtype must be float16 or bfloat16, got {Q.dtype}")
    if K.dtype not in (mx.float16, mx.bfloat16) or V.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K/V dtype must be float16 or bfloat16, got {K.dtype}, {V.dtype}")
    return B, Sq, Sk, H, D


def _metal_attention_common(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    scale: float,
    causal: bool,
    kernel_path: Path,
    kernel_name: str,
    threadgroup: tuple[int, int, int],
    grid_x: int,
    fixed_head_dim: int | None = None,
) -> mx.array:
    B, Sq, _, H, D = _validate_qkv(Q, K, V, require_same_sequence=True)
    dtype = Q.dtype
    K = K.astype(dtype)
    V = V.astype(dtype)
    meta = mx.array([B, Sq, H, D, int(causal)], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = _load_kernel_source(kernel_path)
    header = _make_header(dtype, fixed_head_dim=fixed_head_dim)
    kernel = _get_fast_attention_kernel(
        kernel_name,
        str(dtype),
        source,
        header,
    )
    return kernel(
        inputs=[Q, K, V, meta, scale_arr],
        output_shapes=[(B, Sq, H, D)],
        output_dtypes=[dtype],
        grid=(grid_x, 1, 1),
        threadgroup=threadgroup,
    )[0]


def _fast_attention_baseline(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    scale: float,
    causal: bool,
) -> mx.array:
    B, Sq, _, H, _ = _validate_qkv(Q, K, V, require_same_sequence=True)
    total_rows = B * Sq * H
    return _metal_attention_common(
        Q,
        K,
        V,
        scale=scale,
        causal=causal,
        kernel_path=_DEFAULT_KERNEL,
        kernel_name="fast_attention_forward",
        threadgroup=(1, 1, 1),
        grid_x=total_rows,
    )


def _fast_attention_row_parallel(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    scale: float,
    causal: bool,
) -> mx.array:
    B, Sq, _, H, _ = _validate_qkv(Q, K, V, require_same_sequence=True)
    total_rows = B * Sq * H
    return _metal_attention_common(
        Q,
        K,
        V,
        scale=scale,
        causal=causal,
        kernel_path=_ROW_PARALLEL_KERNEL,
        kernel_name="fast_attention_row_parallel_forward",
        threadgroup=(_ROW_PARALLEL_THREADS, 1, 1),
        grid_x=total_rows * _ROW_PARALLEL_THREADS,
    )


def _fast_attention_tiled_kv(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    scale: float,
    causal: bool,
) -> mx.array:
    B, Sq, _, H, _ = _validate_qkv(Q, K, V, require_same_sequence=True)
    total_rows = B * Sq * H
    return _metal_attention_common(
        Q,
        K,
        V,
        scale=scale,
        causal=causal,
        kernel_path=_TILED_KV_KERNEL,
        kernel_name="fast_attention_tiled_kv_forward",
        threadgroup=(_TILED_KV_THREADS, 1, 1),
        grid_x=total_rows * _TILED_KV_THREADS,
    )


def _fast_attention_specialized(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    scale: float,
    causal: bool,
    backend_name: str,
) -> mx.array:
    B, Sq, _, H, D = _validate_qkv(Q, K, V, require_same_sequence=True)
    expected_d = 64 if backend_name == "baseline_d64" else 128
    if D != expected_d:
        raise ValueError(f"backend={backend_name!r} requires D == {expected_d}, got D={D}")
    total_rows = B * Sq * H
    return _metal_attention_common(
        Q,
        K,
        V,
        scale=scale,
        causal=causal,
        kernel_path=_SPECIALIZED_BASELINE_KERNELS[backend_name],
        kernel_name=f"fast_attention_d{expected_d}_forward",
        threadgroup=(1, 1, 1),
        grid_x=total_rows,
        fixed_head_dim=expected_d,
    )


def fast_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    scale: Optional[float] = None,
    *,
    causal: bool = False,
    backend: str = "auto",
) -> mx.array:
    """Compute fused streaming attention with an MLX custom Metal kernel.

    Parameters
    ----------
    Q, K, V:
        MLX arrays with shape [B, S, H, D] in BSHD layout.
        This v0.1 kernel requires Q/K/V to have identical shapes.
    scale:
        Attention scale. Defaults to 1/sqrt(D).
    causal:
        If True, applies a triangular causal mask so query q attends only to
        keys <= q.
    backend:
        Backend selector. Supported values:
        - "reference": pure MLX materialized reference implementation
        - "baseline": correctness-first one-thread-per-row Metal kernel
        - "row_parallel": experimental one-threadgroup-per-row Metal kernel
        - "tiled_kv": experimental threadgroup-tiled K/V streaming kernel
        - "auto": currently aliases to "baseline"

    Returns
    -------
    mx.array
        Output array with shape [B, S, H, D] and same dtype as Q.

    Notes
    -----
    This is a correctness-first implementation: one Metal thread computes one
    full attention row. It avoids materializing QK^T, but it is not yet the
    optimized tiled/simdgroup version.
    """
    _, _, _, _, D = _validate_qkv(Q, K, V, require_same_sequence=True)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    backend_name = backend.lower()
    if backend_name == "auto":
        if os.environ.get("MLX_METAL_USE_SPECIALIZED", "0") == "1":
            if D == 64:
                backend_name = "baseline_d64"
            elif D == 128:
                backend_name = "baseline_d128"
            else:
                backend_name = "baseline"
        else:
            backend_name = "baseline"

    if backend_name == "reference":
        return reference_attention(Q, K, V, scale=scale, causal=causal)
    if backend_name == "baseline":
        return _fast_attention_baseline(Q, K, V, scale=scale, causal=causal)
    if backend_name in ("baseline_d64", "baseline_d128"):
        return _fast_attention_specialized(Q, K, V, scale=scale, causal=causal, backend_name=backend_name)
    if backend_name == "row_parallel":
        return _fast_attention_row_parallel(Q, K, V, scale=scale, causal=causal)
    if backend_name == "tiled_kv":
        return _fast_attention_tiled_kv(Q, K, V, scale=scale, causal=causal)
    raise ValueError(
        f"Unsupported backend={backend!r}. "
        "Expected one of: 'reference', 'baseline', 'baseline_d64', 'baseline_d128', "
        "'row_parallel', 'tiled_kv', 'auto'."
    )


def reference_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    scale: Optional[float] = None,
    *,
    causal: bool = False,
) -> mx.array:
    """MLX reference implementation used for correctness tests.

    This version materializes the attention scores and should not be used as
    the optimized path.
    """
    B, Sq, Sk, H, D = _validate_qkv(Q, K, V, require_same_sequence=False)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    dtype = Q.dtype
    Qp = Q.astype(mx.float32).transpose(0, 2, 1, 3)  # [B,H,S,D]
    Kp = K.astype(mx.float32).transpose(0, 2, 3, 1)  # [B,H,D,Sk]
    Vp = V.astype(mx.float32).transpose(0, 2, 1, 3)  # [B,H,Sk,D]

    scores = mx.matmul(Qp, Kp) * float(scale)        # [B,H,Sq,Sk]

    if causal:
        i = mx.arange(Sq)[:, None]
        j = mx.arange(Sk)[None, :]
        mask = j > i
        neg_inf = mx.array(-1.0e9, dtype=scores.dtype)
        scores = mx.where(mask[None, None, :, :], neg_inf, scores)

    probs = mx.softmax(scores, axis=-1)
    out = mx.matmul(probs, Vp).transpose(0, 2, 1, 3)  # [B,S,H,D]
    return out.astype(dtype)


def preprocess_v(V: mx.array) -> mx.array:
    """Backward-compatible no-op.

    The earlier scaffold padded V alone, which is invalid because K and V must
    remain aligned.  Keep this function as a no-op until a real K/V repack path
    exists.
    """
    return V


def decode_attention(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    scale: Optional[float] = None,
    *,
    backend: str = "auto",
) -> mx.array:
    """Decode attention scaffold for a single-token query.

    For now this routes through the MLX reference implementation with query
    sequence length 1. A future custom Metal decode kernel should replace this
    path once the paged-KV and split-KV decode work lands.
    """
    B, Sq, Sk, H, D = _validate_qkv(q, K_cache, V_cache, require_same_sequence=False)
    if Sq != 1:
        raise ValueError(f"decode_attention expects q.shape[1] == 1, got {q.shape}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # TODO: Add a dedicated custom Metal decode kernel for backend="baseline".
    # TODO: Add split-KV / paged-KV decode backends once cache formats exist.
    _ = backend  # Reserved for future backend-specific decode dispatch.
    return reference_attention(q, K_cache, V_cache, scale=scale, causal=False)


def fast_attention_with_split(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    scale: Optional[float] = None,
    num_splits: int = 4,
    causal: bool = False,
) -> mx.array:
    """Reference KV-split attention with log-sum-exp merge.

    This is a correctness/reference path implemented with MLX ops.  It is not
    yet a custom split Metal kernel.  It is useful as a template for adding a
    Flash-Decoding-style optimized split kernel later.
    """
    B, Sq, _, H, D = _validate_qkv(Q, K, V, require_same_sequence=True)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if num_splits <= 1:
        return fast_attention(Q, K, V, scale=scale, causal=causal)

    split_size = math.ceil(Sq / num_splits)
    partial_outputs: list[mx.array] = []
    partial_maxes: list[mx.array] = []
    partial_sums: list[mx.array] = []

    Qp = Q.astype(mx.float32).transpose(0, 2, 1, 3)  # [B,H,S,D]

    for g in range(num_splits):
        kv_start = g * split_size
        kv_end = min(kv_start + split_size, Sq)
        if kv_start >= kv_end:
            break

        K_g = K[:, kv_start:kv_end, :, :]
        V_g = V[:, kv_start:kv_end, :, :]
        Kp = K_g.astype(mx.float32).transpose(0, 2, 3, 1)
        Vp = V_g.astype(mx.float32).transpose(0, 2, 1, 3)

        scores = mx.matmul(Qp, Kp) * float(scale)  # [B,H,S,kv]

        if causal:
            q_idx = mx.arange(Sq)[:, None]
            k_idx = mx.arange(kv_start, kv_end)[None, :]
            mask = k_idx > q_idx
            scores = mx.where(mask[None, None, :, :], mx.array(-1.0e9, dtype=scores.dtype), scores)

        row_max = mx.max(scores, axis=-1)              # [B,H,S]
        exp_scores = mx.exp(scores - row_max[..., None])
        row_sum = mx.sum(exp_scores, axis=-1)          # [B,H,S]
        probs = exp_scores / mx.maximum(row_sum[..., None], mx.array(1e-20, dtype=row_sum.dtype))
        partial = mx.matmul(probs, Vp).transpose(0, 2, 1, 3)  # [B,S,H,D]

        partial_outputs.append(partial)
        partial_maxes.append(row_max.transpose(0, 2, 1))
        partial_sums.append(row_sum.transpose(0, 2, 1))

    return _softmax_merge(partial_outputs, partial_maxes, partial_sums).astype(Q.dtype)


def _softmax_merge(partial_outputs: list[mx.array], partial_maxes: list[mx.array], partial_sums: list[mx.array]) -> mx.array:
    if not partial_outputs:
        raise ValueError("No partial outputs to merge")

    global_max = partial_maxes[0]
    for m in partial_maxes[1:]:
        global_max = mx.maximum(global_max, m)

    denom = mx.zeros_like(partial_sums[0])
    out_acc = mx.zeros_like(partial_outputs[0]).astype(mx.float32)

    for out_p, max_p, sum_p in zip(partial_outputs, partial_maxes, partial_sums):
        weight = mx.exp(max_p - global_max)
        denom = denom + weight * sum_p
        out_acc = out_acc + weight[..., None] * out_p.astype(mx.float32)

    return out_acc / mx.maximum(denom[..., None], mx.array(1e-20, dtype=denom.dtype))


def optimal_num_splits(S: int, num_compute_units: int = 8, kv_tile: int = 64) -> int:
    """Simple heuristic for future Flash-Decoding tail split experiments."""
    if S <= 0:
        raise ValueError("S must be positive")
    nkb = math.ceil(S / kv_tile)
    tail = nkb % num_compute_units
    if tail == 0 or tail / num_compute_units >= 0.95:
        return 1

    merge_overhead = 2
    best_g, best_cost = 1, float("inf")
    for g in range(1, min(nkb + 1, 17)):
        rounds_per_part = math.ceil(nkb / g)
        rounds_with_g = math.ceil(max(tail, 1) * g / num_compute_units)
        cost = rounds_with_g * rounds_per_part + merge_overhead
        if cost < best_cost:
            best_cost, best_g = cost, g
    return best_g
