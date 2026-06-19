from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "swiglu.metal"
_THREADS = 256


def _make_header(dtype: mx.Dtype) -> str:
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"swiglu supports only float16/bfloat16, got {dtype}")
    return f"""
#include <metal_stdlib>
using namespace metal;
#define ELEM_TYPE {elem_type}
"""


def _load_source() -> str:
    if not _KERNEL_PATH.exists():
        raise FileNotFoundError(f"Missing Metal kernel source: {_KERNEL_PATH}")
    return _KERNEL_PATH.read_text()


@lru_cache(maxsize=4)
def _get_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="swiglu_forward",
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
    if backend_name == "reference":
        return reference_swiglu(gate, up)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = gate.dtype
    source = _load_source()
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
