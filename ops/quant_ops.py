from __future__ import annotations

import math
import os
from functools import lru_cache

import mlx.core as mx

from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source

_KERNEL_DIR = KERNEL_DIR
_DEQUANT_Q4_KERNEL = _KERNEL_DIR / "dequant_q4.metal"
_DEQUANT_Q8_KERNEL = _KERNEL_DIR / "dequant_q8.metal"
_GROUPWISE_DEQUANT_KERNEL = _KERNEL_DIR / "groupwise_dequant.metal"
_Q4_MATVEC_KERNEL = _KERNEL_DIR / "q4_matvec_decode.metal"
_Q8_MATVEC_KERNEL = _KERNEL_DIR / "q8_matvec_decode.metal"
_Q4_MATVEC_PARALLEL_KERNEL = _KERNEL_DIR / "q4_matvec_decode_parallel.metal"
_Q8_MATVEC_PARALLEL_KERNEL = _KERNEL_DIR / "q8_matvec_decode_parallel.metal"
_Q4_MATVEC_TILED_KERNEL = _KERNEL_DIR / "q4_matvec_decode_tiled.metal"
_Q8_MATVEC_TILED_KERNEL = _KERNEL_DIR / "q8_matvec_decode_tiled.metal"
_Q4_GATE_UP_MATVEC_TILED_KERNEL = _KERNEL_DIR / "q4_gate_up_matvec_tiled.metal"
_Q8_GATE_UP_MATVEC_TILED_KERNEL = _KERNEL_DIR / "q8_gate_up_matvec_tiled.metal"
_THREADS = 256
_MATVEC_PARALLEL_THREADS = 128
_MATVEC_TILED_THREADS = 128
_MATVEC_N_TILE = 4


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype, MATVEC_THREADS=_MATVEC_PARALLEL_THREADS, MATVEC_TILED_THREADS=_MATVEC_TILED_THREADS)


def _ci_safe_mode_enabled() -> bool:
    return os.environ.get("MLX_METAL_CI_SAFE_MODE", "0") == "1"


@lru_cache(maxsize=16)
def _get_dequant_q4_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="dequant_q4_forward",
        input_names=["packed", "scales", "zeros", "meta"],
        output_names=["out"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_dequant_q8_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="dequant_q8_forward",
        input_names=["q", "scales", "zeros", "meta"],
        output_names=["out"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q4_matvec_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q4_matvec_decode_forward",
        input_names=["x", "packed_w", "scales", "zeros", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q8_matvec_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q8_matvec_decode_forward",
        input_names=["x", "q_w", "scales", "zeros", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q4_matvec_parallel_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q4_matvec_decode_parallel_forward",
        input_names=["x", "packed_w", "scales", "zeros", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q8_matvec_parallel_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q8_matvec_decode_parallel_forward",
        input_names=["x", "q_w", "scales", "zeros", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q4_matvec_tiled_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q4_matvec_decode_tiled_forward",
        input_names=["x", "packed_w", "scales", "zeros", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q8_matvec_tiled_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q8_matvec_decode_tiled_forward",
        input_names=["x", "q_w", "scales", "zeros", "meta"],
        output_names=["y"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q4_gate_up_matvec_tiled_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q4_gate_up_matvec_tiled_forward",
        input_names=["x", "gate_packed", "up_packed", "gate_scales", "up_scales", "gate_zeros", "up_zeros", "meta"],
        output_names=["gate", "up"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=16)
def _get_q8_gate_up_matvec_tiled_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="q8_gate_up_matvec_tiled_forward",
        input_names=["x", "gate_q", "up_q", "gate_scales", "up_scales", "gate_zeros", "up_zeros", "meta"],
        output_names=["gate", "up"],
        source=source,
        header=header,
    )


def _normalize_out_dtype(out_dtype):
    if out_dtype is None:
        return mx.float16
    if out_dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"out_dtype must be float16 or bfloat16, got {out_dtype}")
    return out_dtype


def _validate_group_params(M: int, K: int, scales: mx.array, zeros: mx.array | None, group_size: int):
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = math.ceil(K / group_size)
    if scales.shape != (M, groups):
        raise ValueError(f"scales must have shape {(M, groups)}, got {scales.shape}")
    if zeros is not None and zeros.shape != (M, groups):
        raise ValueError(f"zeros must have shape {(M, groups)}, got {zeros.shape}")
    return groups


def pack_q4(q_unpacked: mx.array) -> mx.array:
    if q_unpacked.ndim != 2:
        raise ValueError(f"q_unpacked must be 2-D [M,K], got {q_unpacked.shape}")
    q_int = q_unpacked.astype(mx.uint8)
    if q_int.shape[1] % 2 != 0:
        pad = mx.zeros((q_int.shape[0], 1), dtype=mx.uint8)
        q_int = mx.concatenate([q_int, pad], axis=1)
    low = q_int[:, 0::2]
    high = q_int[:, 1::2]
    packed = low + (high << 4)
    return packed.astype(mx.uint8)


def unpack_q4_reference(packed: mx.array, K: int | None = None) -> mx.array:
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2-D [M,K_packed], got {packed.shape}")
    packed_u8 = packed.astype(mx.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    unpacked = mx.stack([low, high], axis=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    if K is not None:
        unpacked = unpacked[:, :K]
    return unpacked.astype(mx.uint8)


def reference_dequant_q4(
    packed: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
    out_dtype=None,
) -> mx.array:
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2-D [M,K_packed], got {packed.shape}")
    out_dtype = _normalize_out_dtype(out_dtype)
    M, K_packed = packed.shape
    K = K_packed * 2
    groups = _validate_group_params(M, K, scales, zeros, group_size)
    q = unpack_q4_reference(packed, K=K).astype(mx.float32)
    scale_full = mx.repeat(scales.astype(mx.float32), group_size, axis=1)[:, :K]
    if zeros is None:
        out = q * scale_full
    else:
        zero_full = mx.repeat(zeros.astype(mx.float32), group_size, axis=1)[:, :K]
        out = (q - zero_full) * scale_full
    return out.astype(out_dtype)


def dequant_q4(
    packed: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
    out_dtype=None,
    backend: str = "auto",
) -> mx.array:
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2-D [M,K_packed], got {packed.shape}")
    out_dtype = _normalize_out_dtype(out_dtype)
    M, K_packed = packed.shape
    K = K_packed * 2
    _validate_group_params(M, K, scales, zeros, group_size)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if _ci_safe_mode_enabled():
        return reference_dequant_q4(packed, scales, zeros, group_size=group_size, out_dtype=out_dtype)
    if backend_name == "reference":
        return reference_dequant_q4(packed, scales, zeros, group_size=group_size, out_dtype=out_dtype)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = out_dtype
    source = load_metal_source(_DEQUANT_Q4_KERNEL)
    header = _make_header(dtype)
    kernel = _get_dequant_q4_kernel(str(dtype), source, header)
    groups = math.ceil(K / group_size)
    has_zero = 1 if zeros is not None else 0
    zeros_arr = zeros if zeros is not None else mx.zeros((M, groups), dtype=scales.dtype)
    meta = mx.array([M, K_packed, K, group_size, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[packed.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(M, K)],
        output_dtypes=[dtype],
        grid=(M * K, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]


def reference_dequant_q8(
    q: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
    out_dtype=None,
) -> mx.array:
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D [M,K], got {q.shape}")
    out_dtype = _normalize_out_dtype(out_dtype)
    M, K = q.shape
    _validate_group_params(M, K, scales, zeros, group_size)
    q_f = q.astype(mx.float32)
    scale_full = mx.repeat(scales.astype(mx.float32), group_size, axis=1)[:, :K]
    if zeros is None:
        out = q_f * scale_full
    else:
        zero_full = mx.repeat(zeros.astype(mx.float32), group_size, axis=1)[:, :K]
        out = (q_f - zero_full) * scale_full
    return out.astype(out_dtype)


def dequant_q8(
    q: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
    out_dtype=None,
    backend: str = "auto",
) -> mx.array:
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D [M,K], got {q.shape}")
    out_dtype = _normalize_out_dtype(out_dtype)
    M, K = q.shape
    _validate_group_params(M, K, scales, zeros, group_size)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if _ci_safe_mode_enabled():
        return reference_dequant_q8(q, scales, zeros, group_size=group_size, out_dtype=out_dtype)
    if backend_name == "reference":
        return reference_dequant_q8(q, scales, zeros, group_size=group_size, out_dtype=out_dtype)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = out_dtype
    source = load_metal_source(_DEQUANT_Q8_KERNEL)
    header = _make_header(dtype)
    kernel = _get_dequant_q8_kernel(str(dtype), source, header)
    groups = math.ceil(K / group_size)
    has_zero = 1 if zeros is not None else 0
    zeros_arr = zeros if zeros is not None else mx.zeros((M, groups), dtype=scales.dtype)
    meta = mx.array([M, K, group_size, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[q.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(M, K)],
        output_dtypes=[dtype],
        grid=(M * K, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]


def groupwise_dequant(
    q,
    scales,
    zeros=None,
    *,
    bits,
    group_size: int = 32,
    out_dtype=None,
    backend: str = "auto",
):
    if bits == 4:
        return dequant_q4(q, scales, zeros, group_size=group_size, out_dtype=out_dtype, backend=backend)
    if bits == 8:
        return dequant_q8(q, scales, zeros, group_size=group_size, out_dtype=out_dtype, backend=backend)
    raise ValueError(f"bits must be 4 or 8, got {bits}")


def _normalize_x_2d(x: mx.array) -> tuple[mx.array, int, int]:
    if x.ndim == 2:
        return x, x.shape[0], x.shape[1]
    if x.ndim == 3 and x.shape[1] == 1:
        squeezed = x[:, 0, :]
        return squeezed, squeezed.shape[0], squeezed.shape[1]
    raise ValueError(f"x must have shape [B,K] or [B,1,K], got {x.shape}")


def _validate_matvec_input_dtype(x2d: mx.array):
    if x2d.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"x dtype must be float16 or bfloat16, got {x2d.dtype}")


def _normalize_hidden_rows_input(x: mx.array) -> tuple[mx.array, tuple[int, ...], int, int]:
    if x.ndim == 2:
        return x, x.shape, x.shape[0], x.shape[1]
    if x.ndim == 3:
        rows = x.shape[0] * x.shape[1]
        return x.reshape(rows, x.shape[2]), x.shape, rows, x.shape[2]
    raise ValueError(f"x must have shape [rows,K] or [B,S,K], got {x.shape}")


def _restore_hidden_rows_output(y2d: mx.array, original_shape: tuple[int, ...]) -> mx.array:
    if len(original_shape) == 2:
        return y2d
    return y2d.reshape(original_shape[:-1] + (y2d.shape[-1],))


def _zeros_arr_for_matvec(N: int, K: int, scales: mx.array, zeros: mx.array | None, group_size: int):
    groups = _validate_group_params(N, K, scales, zeros, group_size)
    has_zero = 1 if zeros is not None else 0
    zeros_arr = zeros if zeros is not None else mx.zeros((N, groups), dtype=scales.dtype)
    return zeros_arr, groups, has_zero


def _validate_gate_up_shapes(
    x2d: mx.array,
    gate_w: mx.array,
    up_w: mx.array,
    gate_scales: mx.array,
    up_scales: mx.array,
    gate_zeros: mx.array | None,
    up_zeros: mx.array | None,
    *,
    bits: int,
    group_size: int,
) -> tuple[int, int, int]:
    _validate_matvec_input_dtype(x2d)
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    rows, k_dim = x2d.shape
    if gate_w.ndim != 2 or up_w.ndim != 2:
        raise ValueError(f"gate_w and up_w must be 2-D, got {gate_w.shape}, {up_w.shape}")
    if gate_w.shape != up_w.shape:
        raise ValueError(f"gate_w and up_w must have identical shapes, got {gate_w.shape}, {up_w.shape}")
    out_dim = gate_w.shape[0]
    if bits == 4:
        expected_cols = math.ceil(k_dim / 2)
        if gate_w.shape[1] != expected_cols:
            raise ValueError(f"q4 gate/up weights must have shape [{out_dim},{expected_cols}], got {gate_w.shape}")
    else:
        if gate_w.shape[1] != k_dim:
            raise ValueError(f"q8 gate/up weights must have shape [{out_dim},{k_dim}], got {gate_w.shape}")
    groups = _validate_group_params(out_dim, k_dim, gate_scales, gate_zeros, group_size)
    up_groups = _validate_group_params(out_dim, k_dim, up_scales, up_zeros, group_size)
    if groups != up_groups:
        raise ValueError(f"gate and up scale group counts must match, got {groups} and {up_groups}")
    return rows, k_dim, out_dim


def _q4_matvec_decode_parallel(
    x2d: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    group_size: int,
) -> mx.array:
    B, K = x2d.shape
    N, K_packed = packed_w.shape
    zeros_arr, groups, has_zero = _zeros_arr_for_matvec(N, K, scales, zeros, group_size)
    dtype = x2d.dtype
    source = load_metal_source(_Q4_MATVEC_PARALLEL_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q4_matvec_parallel_kernel(str(dtype), source, header)
    meta = mx.array([B, N, K, K_packed, group_size, groups, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), packed_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * N * _MATVEC_PARALLEL_THREADS, 1, 1),
        threadgroup=(_MATVEC_PARALLEL_THREADS, 1, 1),
    )[0]


def _q8_matvec_decode_parallel(
    x2d: mx.array,
    q_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    group_size: int,
) -> mx.array:
    B, K = x2d.shape
    N = q_w.shape[0]
    zeros_arr, groups, has_zero = _zeros_arr_for_matvec(N, K, scales, zeros, group_size)
    dtype = x2d.dtype
    source = load_metal_source(_Q8_MATVEC_PARALLEL_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q8_matvec_parallel_kernel(str(dtype), source, header)
    meta = mx.array([B, N, K, group_size, groups, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), q_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * N * _MATVEC_PARALLEL_THREADS, 1, 1),
        threadgroup=(_MATVEC_PARALLEL_THREADS, 1, 1),
    )[0]


def _q4_matvec_decode_tiled(
    x2d: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    group_size: int,
) -> mx.array:
    B, K = x2d.shape
    N, K_packed = packed_w.shape
    zeros_arr, groups, has_zero = _zeros_arr_for_matvec(N, K, scales, zeros, group_size)
    dtype = x2d.dtype
    source = load_metal_source(_Q4_MATVEC_TILED_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q4_matvec_tiled_kernel(str(dtype), source, header)
    tiles_per_batch = math.ceil(N / _MATVEC_N_TILE)
    meta = mx.array([B, N, K, K_packed, group_size, groups, has_zero, _MATVEC_N_TILE], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), packed_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * tiles_per_batch * _MATVEC_TILED_THREADS, 1, 1),
        threadgroup=(_MATVEC_TILED_THREADS, 1, 1),
    )[0]


def _q8_matvec_decode_tiled(
    x2d: mx.array,
    q_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    group_size: int,
) -> mx.array:
    B, K = x2d.shape
    N = q_w.shape[0]
    zeros_arr, groups, has_zero = _zeros_arr_for_matvec(N, K, scales, zeros, group_size)
    dtype = x2d.dtype
    source = load_metal_source(_Q8_MATVEC_TILED_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q8_matvec_tiled_kernel(str(dtype), source, header)
    tiles_per_batch = math.ceil(N / _MATVEC_N_TILE)
    meta = mx.array([B, N, K, group_size, groups, has_zero, _MATVEC_N_TILE], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), q_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * tiles_per_batch * _MATVEC_TILED_THREADS, 1, 1),
        threadgroup=(_MATVEC_TILED_THREADS, 1, 1),
    )[0]


def reference_q4_matvec_decode(
    x: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
) -> mx.array:
    x2d, B, K = _normalize_x_2d(x)
    if packed_w.ndim != 2:
        raise ValueError(f"packed_w must be 2-D [N,K_packed], got {packed_w.shape}")
    N, K_packed = packed_w.shape
    if K_packed * 2 != K:
        raise ValueError(f"packed_w implies K={K_packed * 2}, but x has K={K}")
    W = reference_dequant_q4(packed_w, scales, zeros, group_size=group_size, out_dtype=x2d.dtype)
    return mx.matmul(x2d.astype(mx.float32), W.astype(mx.float32).transpose()).astype(x2d.dtype)


def q4_matvec_decode(
    x: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
    backend: str = "auto",
) -> mx.array:
    x2d, B, K = _normalize_x_2d(x)
    _validate_matvec_input_dtype(x2d)
    if packed_w.ndim != 2:
        raise ValueError(f"packed_w must be 2-D [N,K_packed], got {packed_w.shape}")
    N, K_packed = packed_w.shape
    if K_packed * 2 != K:
        raise ValueError(f"packed_w implies K={K_packed * 2}, but x has K={K}")
    _validate_group_params(N, K, scales, zeros, group_size)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal_tiled" if os.environ.get("MLX_METAL_USE_TILED_MATVEC", "0") == "1" else "metal"
    if backend_name == "reference":
        return reference_q4_matvec_decode(x2d, packed_w, scales, zeros, group_size=group_size)
    if _ci_safe_mode_enabled():
        return reference_q4_matvec_decode(x2d, packed_w, scales, zeros, group_size=group_size)
    if backend_name == "metal_parallel":
        return _q4_matvec_decode_parallel(x2d, packed_w, scales, zeros, group_size=group_size)
    if backend_name == "metal_tiled":
        return _q4_matvec_decode_tiled(x2d, packed_w, scales, zeros, group_size=group_size)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'metal_parallel', 'metal_tiled', 'auto'")
    dtype = x2d.dtype
    source = load_metal_source(_Q4_MATVEC_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q4_matvec_kernel(str(dtype), source, header)
    zeros_arr, groups, has_zero = _zeros_arr_for_matvec(N, K, scales, zeros, group_size)
    meta = mx.array([B, K, N, K_packed, group_size, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), packed_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * N, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def reference_q8_matvec_decode(
    x: mx.array,
    q_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
) -> mx.array:
    x2d, _, K = _normalize_x_2d(x)
    if q_w.ndim != 2 or q_w.shape[1] != K:
        raise ValueError(f"q_w must have shape [N,{K}], got {q_w.shape}")
    W = reference_dequant_q8(q_w, scales, zeros, group_size=group_size, out_dtype=x2d.dtype)
    return mx.matmul(x2d.astype(mx.float32), W.astype(mx.float32).transpose()).astype(x2d.dtype)


def q8_matvec_decode(
    x: mx.array,
    q_w: mx.array,
    scales: mx.array,
    zeros: mx.array | None = None,
    *,
    group_size: int = 32,
    backend: str = "auto",
) -> mx.array:
    x2d, B, K = _normalize_x_2d(x)
    _validate_matvec_input_dtype(x2d)
    if q_w.ndim != 2 or q_w.shape[1] != K:
        raise ValueError(f"q_w must have shape [N,{K}], got {q_w.shape}")
    N = q_w.shape[0]
    _validate_group_params(N, K, scales, zeros, group_size)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal_tiled" if os.environ.get("MLX_METAL_USE_TILED_MATVEC", "0") == "1" else "metal"
    if backend_name == "reference":
        return reference_q8_matvec_decode(x2d, q_w, scales, zeros, group_size=group_size)
    if _ci_safe_mode_enabled():
        return reference_q8_matvec_decode(x2d, q_w, scales, zeros, group_size=group_size)
    if backend_name == "metal_parallel":
        return _q8_matvec_decode_parallel(x2d, q_w, scales, zeros, group_size=group_size)
    if backend_name == "metal_tiled":
        return _q8_matvec_decode_tiled(x2d, q_w, scales, zeros, group_size=group_size)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'metal_parallel', 'metal_tiled', 'auto'")
    dtype = x2d.dtype
    source = load_metal_source(_Q8_MATVEC_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q8_matvec_kernel(str(dtype), source, header)
    zeros_arr, groups, has_zero = _zeros_arr_for_matvec(N, K, scales, zeros, group_size)
    meta = mx.array([B, K, N, group_size, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), q_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * N, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def q4_gate_up_matvec_tiled(
    x: mx.array,
    gate_packed: mx.array,
    up_packed: mx.array,
    gate_scales: mx.array,
    up_scales: mx.array,
    gate_zeros: mx.array | None = None,
    up_zeros: mx.array | None = None,
    *,
    group_size: int = 32,
):
    x2d, original_shape, rows, k_dim = _normalize_hidden_rows_input(x)
    _, _, out_dim = _validate_gate_up_shapes(
        x2d,
        gate_packed,
        up_packed,
        gate_scales,
        up_scales,
        gate_zeros,
        up_zeros,
        bits=4,
        group_size=group_size,
    )
    dtype = x2d.dtype
    source = load_metal_source(_Q4_GATE_UP_MATVEC_TILED_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q4_gate_up_matvec_tiled_kernel(str(dtype), source, header)
    gate_zeros_arr, groups, has_gate_zero = _zeros_arr_for_matvec(out_dim, k_dim, gate_scales, gate_zeros, group_size)
    up_zeros_arr, _, has_up_zero = _zeros_arr_for_matvec(out_dim, k_dim, up_scales, up_zeros, group_size)
    k_packed = gate_packed.shape[1]
    tiles_per_row = math.ceil(out_dim / _MATVEC_N_TILE)
    meta = mx.array(
        [rows, k_dim, out_dim, k_packed, group_size, groups, has_gate_zero, has_up_zero, _MATVEC_N_TILE],
        dtype=mx.int32,
    )
    gate2d, up2d = kernel(
        inputs=[
            x2d.astype(dtype),
            gate_packed.astype(mx.uint8),
            up_packed.astype(mx.uint8),
            gate_scales.astype(mx.float32),
            up_scales.astype(mx.float32),
            gate_zeros_arr.astype(mx.float32),
            up_zeros_arr.astype(mx.float32),
            meta,
        ],
        output_shapes=[(rows, out_dim), (rows, out_dim)],
        output_dtypes=[dtype, dtype],
        grid=(rows * tiles_per_row * _MATVEC_TILED_THREADS, 1, 1),
        threadgroup=(_MATVEC_TILED_THREADS, 1, 1),
    )
    return _restore_hidden_rows_output(gate2d, original_shape), _restore_hidden_rows_output(up2d, original_shape)


def q8_gate_up_matvec_tiled(
    x: mx.array,
    gate_q: mx.array,
    up_q: mx.array,
    gate_scales: mx.array,
    up_scales: mx.array,
    gate_zeros: mx.array | None = None,
    up_zeros: mx.array | None = None,
    *,
    group_size: int = 32,
):
    x2d, original_shape, rows, k_dim = _normalize_hidden_rows_input(x)
    _, _, out_dim = _validate_gate_up_shapes(
        x2d,
        gate_q,
        up_q,
        gate_scales,
        up_scales,
        gate_zeros,
        up_zeros,
        bits=8,
        group_size=group_size,
    )
    dtype = x2d.dtype
    source = load_metal_source(_Q8_GATE_UP_MATVEC_TILED_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q8_gate_up_matvec_tiled_kernel(str(dtype), source, header)
    gate_zeros_arr, groups, has_gate_zero = _zeros_arr_for_matvec(out_dim, k_dim, gate_scales, gate_zeros, group_size)
    up_zeros_arr, _, has_up_zero = _zeros_arr_for_matvec(out_dim, k_dim, up_scales, up_zeros, group_size)
    tiles_per_row = math.ceil(out_dim / _MATVEC_N_TILE)
    meta = mx.array(
        [rows, k_dim, out_dim, group_size, groups, has_gate_zero, has_up_zero, _MATVEC_N_TILE],
        dtype=mx.int32,
    )
    gate2d, up2d = kernel(
        inputs=[
            x2d.astype(dtype),
            gate_q.astype(mx.uint8),
            up_q.astype(mx.uint8),
            gate_scales.astype(mx.float32),
            up_scales.astype(mx.float32),
            gate_zeros_arr.astype(mx.float32),
            up_zeros_arr.astype(mx.float32),
            meta,
        ],
        output_shapes=[(rows, out_dim), (rows, out_dim)],
        output_dtypes=[dtype, dtype],
        grid=(rows * tiles_per_row * _MATVEC_TILED_THREADS, 1, 1),
        threadgroup=(_MATVEC_TILED_THREADS, 1, 1),
    )
    return _restore_hidden_rows_output(gate2d, original_shape), _restore_hidden_rows_output(up2d, original_shape)
