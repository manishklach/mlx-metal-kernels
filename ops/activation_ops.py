from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx

from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source

_KERNEL_PATH = KERNEL_DIR / "swiglu.metal"
_FUSED_KERNEL_PATH = KERNEL_DIR / "fused_swiglu.metal"
_THREADS = 256


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype)


@lru_cache(maxsize=4)
def _get_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="swiglu_forward",
        input_names=["gate", "up", "meta"],
        output_names=["out"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=4)
def _get_fused_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="fused_swiglu_forward",
        input_names=["gate", "up", "meta"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _validate_inputs(gate: mx.array, up: mx.array) -> tuple[int, int, int]:
    if gate.ndim != 3 or up.ndim != 3:
        raise ValueError(f"gate and up must be 3-D [B,S,D], got {gate.shape}, {up.shape}")
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have identical shapes, got {gate.shape}, {up.shape}")
    B, S, D = gate.shape
    if gate.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"gate dtype must be float16 or bfloat16, got {gate.dtype}")
    if up.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"up dtype must be float16 or bfloat16, got {up.dtype}")
    return B, S, D


def _normalize_fused_inputs(gate: mx.array, up: mx.array) -> tuple[mx.array, mx.array, tuple[int, ...], int, int]:
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have identical shapes, got {gate.shape}, {up.shape}")
    if gate.ndim == 2:
        rows, dim = gate.shape
        original_shape = gate.shape
        gate2d = gate
        up2d = up
    elif gate.ndim == 3:
        rows = gate.shape[0] * gate.shape[1]
        dim = gate.shape[2]
        original_shape = gate.shape
        gate2d = gate.reshape(rows, dim)
        up2d = up.reshape(rows, dim)
    else:
        raise ValueError(f"gate and up must have shape [rows,D] or [B,S,D], got {gate.shape}, {up.shape}")
    if gate.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"gate dtype must be float16 or bfloat16, got {gate.dtype}")
    if up.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"up dtype must be float16 or bfloat16, got {up.dtype}")
    return gate2d, up2d, original_shape, rows, dim


def _restore_fused_output(out2d: mx.array, original_shape: tuple[int, ...]) -> mx.array:
    if len(original_shape) == 2:
        return out2d
    return out2d.reshape(original_shape)


def reference_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    _validate_inputs(gate, up)
    gate_f32 = gate.astype(mx.float32)
    up_f32 = up.astype(mx.float32)
    silu = gate_f32 / (1.0 + mx.exp(-gate_f32))
    return (silu * up_f32).astype(gate.dtype)


def swiglu(gate: mx.array, up: mx.array, *, backend: str = "auto") -> mx.array:
    B, S, D = _validate_inputs(gate, up)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if os.environ.get("MLX_METAL_CI_SAFE_MODE", "0") == "1":
        return reference_swiglu(gate, up)
    if backend_name == "reference":
        return reference_swiglu(gate, up)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = gate.dtype
    source = load_metal_source(_KERNEL_PATH)
    header = _make_header(dtype)
    kernel = _get_kernel(str(dtype), source, header)
    meta = mx.array([B, S, D], dtype=mx.int32)
    return kernel(
        inputs=[gate, up.astype(dtype), meta],
        output_shapes=[gate.shape],
        output_dtypes=[dtype],
        grid=(B * S * D, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]


def fused_swiglu(gate: mx.array, up: mx.array, *, backend: str = "metal_fused") -> mx.array:
    gate2d, up2d, original_shape, rows, dim = _normalize_fused_inputs(gate, up)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal_fused"
    if os.environ.get("MLX_METAL_CI_SAFE_MODE", "0") == "1":
        gate_f32 = gate2d.astype(mx.float32)
        up_f32 = up2d.astype(mx.float32)
        silu = gate_f32 / (1.0 + mx.exp(-gate_f32))
        out2d = (silu * up_f32).astype(gate2d.dtype)
        return _restore_fused_output(out2d, original_shape)
    if backend_name == "reference":
        gate_f32 = gate2d.astype(mx.float32)
        up_f32 = up2d.astype(mx.float32)
        silu = gate_f32 / (1.0 + mx.exp(-gate_f32))
        out2d = (silu * up_f32).astype(gate2d.dtype)
        return _restore_fused_output(out2d, original_shape)
    if backend_name == "metal":
        if len(original_shape) != 3:
            raise ValueError("backend='metal' for fused_swiglu requires [B,S,D] inputs; use 'metal_fused' for flattened shapes")
        return swiglu(gate, up, backend="metal")
    if backend_name != "metal_fused":
        raise ValueError("backend must be one of 'reference', 'metal', 'metal_fused', 'auto'")

    dtype = gate2d.dtype
    source = load_metal_source(_FUSED_KERNEL_PATH)
    header = _make_header(dtype)
    kernel = _get_fused_kernel(str(dtype), source, header)
    meta = mx.array([rows, dim], dtype=mx.int32)
    out2d = kernel(
        inputs=[gate2d.astype(dtype), up2d.astype(dtype), meta],
        output_shapes=[(rows, dim)],
        output_dtypes=[dtype],
        grid=(rows * dim, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )[0]
    return _restore_fused_output(out2d, original_shape)
