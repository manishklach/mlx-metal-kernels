from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx

from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source

_KERNEL_PATH = KERNEL_DIR / "kv_cache_update.metal"
_THREADS = 256


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype)


@lru_cache(maxsize=4)
def _get_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="kv_cache_update_forward",
        input_names=["K_cache", "V_cache", "k_new", "v_new", "positions", "meta"],
        output_names=["updated_K_cache", "updated_V_cache"],
        source=source,
        header=header,
    )


def _normalize_token_shape(x: mx.array, name: str) -> mx.array:
    if x.ndim == 3:
        return x[:, None, :, :]
    if x.ndim == 4 and x.shape[1] == 1:
        return x
    raise ValueError(f"{name} must have shape [B,1,H,D] or [B,H,D], got {x.shape}")


def normalize_positions(positions, B: int, MAX_S: int) -> mx.array:
    if isinstance(positions, int):
        pos = mx.full((B,), positions, dtype=mx.int32)
    elif hasattr(positions, "shape") and hasattr(positions, "astype"):
        if positions.ndim == 0:
            pos = mx.full((B,), positions.astype(mx.int32).item(), dtype=mx.int32)
        elif positions.ndim == 1 and positions.shape[0] == B:
            pos = positions.astype(mx.int32)
        else:
            raise ValueError(f"positions must be scalar or shape [B], got {positions.shape}")
    else:
        pos_arr = mx.array(positions, dtype=mx.int32)
        if pos_arr.ndim == 0:
            pos = mx.full((B,), pos_arr.item(), dtype=mx.int32)
        elif pos_arr.ndim == 1 and pos_arr.shape[0] == B:
            pos = pos_arr.astype(mx.int32)
        else:
            raise ValueError(f"positions must be scalar or shape [B], got {pos_arr.shape}")
    pos_vals = pos.astype(mx.int32).tolist()
    if not isinstance(pos_vals, list):
        pos_vals = [pos_vals]
    for p in pos_vals:
        if p < 0 or p >= MAX_S:
            raise ValueError(f"positions must be in [0, {MAX_S}), got {p}")
    return pos


def _validate_inputs(
    K_cache: mx.array,
    V_cache: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    positions,
) -> tuple[mx.array, mx.array, mx.array, int, int, int, int]:
    if K_cache.ndim != 4 or V_cache.ndim != 4:
        raise ValueError(f"K_cache and V_cache must be 4-D [B,MAX_S,H,D], got {K_cache.shape}, {V_cache.shape}")
    if K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must have identical shapes, got {K_cache.shape}, {V_cache.shape}")
    k_new = _normalize_token_shape(k_new, "k_new")
    v_new = _normalize_token_shape(v_new, "v_new")
    B, MAX_S, H, D = K_cache.shape
    if k_new.shape != (B, 1, H, D) or v_new.shape != (B, 1, H, D):
        raise ValueError(
            "k_new and v_new must match cache batch/head/head_dim after normalization. "
            f"Got k_new={k_new.shape}, v_new={v_new.shape}, cache={K_cache.shape}."
        )
    if K_cache.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K_cache dtype must be float16 or bfloat16, got {K_cache.dtype}")
    if V_cache.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"V_cache dtype must be float16 or bfloat16, got {V_cache.dtype}")
    if k_new.dtype not in (mx.float16, mx.bfloat16) or v_new.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"k_new/v_new dtype must be float16 or bfloat16, got {k_new.dtype}, {v_new.dtype}")
    pos_arr = normalize_positions(positions, B, MAX_S)
    return k_new, v_new, pos_arr, B, MAX_S, H, D


def reference_kv_cache_update(
    K_cache: mx.array,
    V_cache: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    positions,
) -> tuple[mx.array, mx.array]:
    k_new, v_new, pos_arr, B, MAX_S, H, D = _validate_inputs(K_cache, V_cache, k_new, v_new, positions)
    idx = mx.arange(MAX_S).reshape(1, MAX_S, 1, 1)
    mask = idx == pos_arr.reshape(B, 1, 1, 1)
    k_broadcast = mx.broadcast_to(k_new.astype(K_cache.dtype), K_cache.shape)
    v_broadcast = mx.broadcast_to(v_new.astype(V_cache.dtype), V_cache.shape)
    K_updated = mx.where(mask, k_broadcast, K_cache)
    V_updated = mx.where(mask, v_broadcast, V_cache)
    return K_updated, V_updated


def kv_cache_update(
    K_cache: mx.array,
    V_cache: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    positions,
    *,
    backend: str = "auto",
) -> tuple[mx.array, mx.array]:
    k_new, v_new, pos_arr, B, MAX_S, H, D = _validate_inputs(K_cache, V_cache, k_new, v_new, positions)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if os.environ.get("MLX_METAL_CI_SAFE_MODE", "0") == "1":
        return reference_kv_cache_update(K_cache, V_cache, k_new, v_new, pos_arr)
    if backend_name == "reference":
        return reference_kv_cache_update(K_cache, V_cache, k_new, v_new, pos_arr)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = K_cache.dtype
    source = load_metal_source(_KERNEL_PATH)
    header = _make_header(dtype)
    kernel = _get_kernel(str(dtype), source, header)
    meta = mx.array([B, MAX_S, H, D], dtype=mx.int32)
    outputs = kernel(
        inputs=[K_cache, V_cache, k_new.astype(dtype), v_new.astype(dtype), pos_arr, meta],
        output_shapes=[K_cache.shape, V_cache.shape],
        output_dtypes=[dtype, dtype],
        grid=(B * MAX_S * H * D, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return outputs[0], outputs[1]
