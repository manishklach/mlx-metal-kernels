from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
_DEQUANT_Q4_KERNEL = _KERNEL_DIR / "dequant_q4.metal"
_DEQUANT_Q8_KERNEL = _KERNEL_DIR / "dequant_q8.metal"
_GROUPWISE_DEQUANT_KERNEL = _KERNEL_DIR / "groupwise_dequant.metal"
_Q4_MATVEC_KERNEL = _KERNEL_DIR / "q4_matvec_decode.metal"
_Q8_MATVEC_KERNEL = _KERNEL_DIR / "q8_matvec_decode.metal"
_THREADS = 256


def _make_header(dtype: mx.Dtype) -> str:
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"quant ops support only float16/bfloat16 outputs, got {dtype}")
    return f"""
#include <metal_stdlib>
using namespace metal;
#define ELEM_TYPE {elem_type}
"""


def _load_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing Metal kernel source: {path}")
    return path.read_text()


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
    if backend_name == "reference":
        return reference_dequant_q4(packed, scales, zeros, group_size=group_size, out_dtype=out_dtype)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = out_dtype
    source = _load_source(_DEQUANT_Q4_KERNEL)
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
    if backend_name == "reference":
        return reference_dequant_q8(q, scales, zeros, group_size=group_size, out_dtype=out_dtype)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = out_dtype
    source = _load_source(_DEQUANT_Q8_KERNEL)
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
    if packed_w.ndim != 2:
        raise ValueError(f"packed_w must be 2-D [N,K_packed], got {packed_w.shape}")
    N, K_packed = packed_w.shape
    if K_packed * 2 != K:
        raise ValueError(f"packed_w implies K={K_packed * 2}, but x has K={K}")
    _validate_group_params(N, K, scales, zeros, group_size)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_q4_matvec_decode(x2d, packed_w, scales, zeros, group_size=group_size)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = x2d.dtype
    source = _load_source(_Q4_MATVEC_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q4_matvec_kernel(str(dtype), source, header)
    groups = math.ceil(K / group_size)
    has_zero = 1 if zeros is not None else 0
    zeros_arr = zeros if zeros is not None else mx.zeros((N, groups), dtype=scales.dtype)
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
    if q_w.ndim != 2 or q_w.shape[1] != K:
        raise ValueError(f"q_w must have shape [N,{K}], got {q_w.shape}")
    N = q_w.shape[0]
    _validate_group_params(N, K, scales, zeros, group_size)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_q8_matvec_decode(x2d, q_w, scales, zeros, group_size=group_size)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = x2d.dtype
    source = _load_source(_Q8_MATVEC_KERNEL)
    header = _make_header(dtype)
    kernel = _get_q8_matvec_kernel(str(dtype), source, header)
    groups = math.ceil(K / group_size)
    has_zero = 1 if zeros is not None else 0
    zeros_arr = zeros if zeros is not None else mx.zeros((N, groups), dtype=scales.dtype)
    meta = mx.array([B, K, N, group_size, has_zero], dtype=mx.int32)
    return kernel(
        inputs=[x2d.astype(dtype), q_w.astype(mx.uint8), scales.astype(mx.float32), zeros_arr.astype(mx.float32), meta],
        output_shapes=[(B, N)],
        output_dtypes=[dtype],
        grid=(B * N, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]
