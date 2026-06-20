from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .decode_ops import decode_attention, reference_decode_attention
from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source
from .kv_cache_ops import kv_cache_update, normalize_positions, reference_kv_cache_update
from .layout_ops import qkv_split_rope, reference_qkv_split_rope, _validate_qkv_input, _validate_rope_inputs

_QKV_CACHE_KERNEL = KERNEL_DIR / "qkv_split_rope_cache_update.metal"
_RESIDUAL_ADD_KERNEL = KERNEL_DIR / "residual_add.metal"
_RMSNORM_RESIDUAL_KERNEL = KERNEL_DIR / "rmsnorm_residual.metal"
_THREADS = 256


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype)


@lru_cache(maxsize=8)
def _get_qkv_cache_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="qkv_split_rope_cache_update_forward",
        input_names=["qkv", "K_cache", "V_cache", "cos", "sin", "positions", "meta"],
        output_names=["q_rope", "updated_K_cache", "updated_V_cache"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_residual_add_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="residual_add_forward",
        input_names=["x", "residual", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_rmsnorm_residual_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="rmsnorm_residual_forward",
        input_names=["x", "residual", "weight", "meta", "eps"],
        output_names=["y", "z"],
        source=source,
        header=header,
    )


def _normalize_position_array(position, B: int, MAX_S: int) -> mx.array:
    return normalize_positions(position, B, MAX_S)


def _validate_cache_inputs(K_cache: mx.array, V_cache: mx.array) -> tuple[int, int, int, int]:
    if K_cache.ndim != 4 or V_cache.ndim != 4 or K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must be matching [B,MAX_S,H,D], got {K_cache.shape}, {V_cache.shape}")
    return K_cache.shape


def reference_qkv_rope_cache_update(
    qkv: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    H: int | None = None,
    D: int | None = None,
):
    B, S, H_val, D_val, _, _ = _validate_rope_inputs(qkv, cos, sin, H, D, 0)
    if S != 1:
        raise ValueError(f"qkv_rope_cache_update is decode-only and requires S == 1, got {qkv.shape}")
    cache_B, MAX_S, cache_H, cache_D = _validate_cache_inputs(K_cache, V_cache)
    if (B, H_val, D_val) != (cache_B, cache_H, cache_D):
        raise ValueError(
            f"qkv and cache tensors must agree on batch/heads/head_dim, got qkv={qkv.shape}, cache={K_cache.shape}"
        )

    pos_arr = _normalize_position_array(position, B, MAX_S)
    q, k, v = reference_qkv_split_rope(qkv, cos, sin, H=H_val, D=D_val, position_offset=0)
    # Per-batch positions need per-batch RoPE on q and k.
    q_rope_rows = []
    k_rope_rows = []
    for b in range(B):
        pos_b = int(pos_arr[b].item())
        q_b, k_b, v_b = reference_qkv_split_rope(qkv[b:b+1], cos, sin, H=H_val, D=D_val, position_offset=pos_b)
        q_rope_rows.append(q_b)
        k_rope_rows.append(k_b)
    q_rope = mx.concatenate(q_rope_rows, axis=0) if B > 1 else q_rope_rows[0]
    k_rope = mx.concatenate(k_rope_rows, axis=0) if B > 1 else k_rope_rows[0]
    updated_K, updated_V = reference_kv_cache_update(K_cache, V_cache, k_rope, v, pos_arr)
    return q_rope.astype(qkv.dtype), updated_K, updated_V


def qkv_rope_cache_update(
    qkv: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    H: int | None = None,
    D: int | None = None,
    *,
    backend: str = "auto",
):
    B, S, H_val, D_val, input_layout, cos_rows = _validate_rope_inputs(qkv, cos, sin, H, D, 0)
    if S != 1:
        raise ValueError(f"qkv_rope_cache_update is decode-only and requires S == 1, got {qkv.shape}")
    cache_B, MAX_S, cache_H, cache_D = _validate_cache_inputs(K_cache, V_cache)
    if (B, H_val, D_val) != (cache_B, cache_H, cache_D):
        raise ValueError(
            f"qkv and cache tensors must agree on batch/heads/head_dim, got qkv={qkv.shape}, cache={K_cache.shape}"
        )
    pos_arr = _normalize_position_array(position, B, MAX_S)

    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_qkv_rope_cache_update(qkv, K_cache, V_cache, cos, sin, pos_arr, H=H_val, D=D_val)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = qkv.dtype
    source = load_metal_source(_QKV_CACHE_KERNEL)
    header = _make_header(dtype)
    kernel = _get_qkv_cache_kernel(str(dtype), source, header)
    meta = mx.array([B, MAX_S, H_val, D_val, cos_rows, input_layout], dtype=mx.int32)
    q_shape = (B, 1, H_val, D_val)
    outputs = kernel(
        inputs=[qkv, K_cache.astype(dtype), V_cache.astype(dtype), cos.astype(mx.float32), sin.astype(mx.float32), pos_arr, meta],
        output_shapes=[q_shape, K_cache.shape, V_cache.shape],
        output_dtypes=[dtype, K_cache.dtype, V_cache.dtype],
        grid=(B * MAX_S * H_val * D_val, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return outputs[0], outputs[1], outputs[2]


def reference_residual_add(x: mx.array, residual: mx.array) -> mx.array:
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have identical shapes, got {x.shape}, {residual.shape}")
    return (x.astype(mx.float32) + residual.astype(mx.float32)).astype(x.dtype)


def residual_add(x: mx.array, residual: mx.array, *, backend: str = "auto") -> mx.array:
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have identical shapes, got {x.shape}, {residual.shape}")
    if x.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"x dtype must be float16 or bfloat16, got {x.dtype}")
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_residual_add(x, residual)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = x.dtype
    source = load_metal_source(_RESIDUAL_ADD_KERNEL)
    header = _make_header(dtype)
    kernel = _get_residual_add_kernel(str(dtype), source, header)
    meta = mx.array([x.size], dtype=mx.int32)
    return kernel(
        inputs=[x, residual.astype(dtype), meta],
        output_shapes=[x.shape],
        output_dtypes=[dtype],
        grid=(x.size, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]


def reference_rmsnorm_residual(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1e-5,
    *,
    return_residual: bool = False,
):
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have identical shapes, got {x.shape}, {residual.shape}")
    if weight.ndim != 1 or x.shape[-1] != weight.shape[0]:
        raise ValueError(f"weight must have shape [D] with D={x.shape[-1]}, got {weight.shape}")
    z = reference_residual_add(x, residual)
    variance = mx.mean(mx.square(z.astype(mx.float32)), axis=-1, keepdims=True)
    y = (z.astype(mx.float32) * mx.rsqrt(variance + float(eps)) * weight.astype(mx.float32)).astype(x.dtype)
    return (y, z) if return_residual else y


def rmsnorm_residual(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1e-5,
    *,
    return_residual: bool = False,
    backend: str = "auto",
):
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have identical shapes, got {x.shape}, {residual.shape}")
    if weight.ndim != 1 or x.shape[-1] != weight.shape[0]:
        raise ValueError(f"weight must have shape [D] with D={x.shape[-1]}, got {weight.shape}")
    if x.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"x dtype must be float16 or bfloat16, got {x.dtype}")
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_rmsnorm_residual(x, residual, weight, eps=eps, return_residual=return_residual)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    rows = x.size // x.shape[-1]
    D = x.shape[-1]
    dtype = x.dtype
    source = load_metal_source(_RMSNORM_RESIDUAL_KERNEL)
    header = _make_header(dtype)
    kernel = _get_rmsnorm_residual_kernel(str(dtype), source, header)
    meta = mx.array([rows, D], dtype=mx.int32)
    eps_arr = mx.array([float(eps)], dtype=mx.float32)
    outputs = kernel(
        inputs=[x, residual.astype(dtype), weight.astype(dtype), meta, eps_arr],
        output_shapes=[x.shape, x.shape],
        output_dtypes=[dtype, dtype],
        grid=(rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return (outputs[0], outputs[1]) if return_residual else outputs[0]


def fused_decode_step_from_qkv(
    qkv: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None = None,
    D: int | None = None,
    scale=None,
    backend: str = "auto",
):
    q_rope, K_cache, V_cache = qkv_rope_cache_update(
        qkv, K_cache, V_cache, cos, sin, position, H=H, D=D, backend=backend
    )
    if isinstance(position, int):
        lengths = position + 1
    else:
        lengths = _normalize_position_array(position, K_cache.shape[0], K_cache.shape[1]) + 1
    out = decode_attention(q_rope, K_cache, V_cache, lengths=lengths, scale=scale, backend=backend)
    return out, K_cache, V_cache
