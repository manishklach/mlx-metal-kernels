from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .rope_ops import reference_apply_rope

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
_QKV_SPLIT_KERNEL = _KERNEL_DIR / "qkv_split.metal"
_QKV_SPLIT_ROPE_KERNEL = _KERNEL_DIR / "qkv_split_rope.metal"
_THREADS = 256


def _make_header(dtype: mx.Dtype) -> str:
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"layout ops support only float16/bfloat16, got {dtype}")
    return f"""
#include <metal_stdlib>
using namespace metal;
#define ELEM_TYPE {elem_type}
"""


def _load_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing Metal kernel source: {path}")
    return path.read_text()


@lru_cache(maxsize=8)
def _get_split_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="qkv_split_forward",
        input_names=["qkv", "meta"],
        output_names=["q", "k", "v"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_split_rope_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="qkv_split_rope_forward",
        input_names=["qkv", "cos", "sin", "meta"],
        output_names=["q_rope", "k_rope", "v"],
        source=source,
        header=header,
    )


def _validate_qkv_input(qkv: mx.array, H: int | None, D: int | None) -> tuple[int, int, int, int, int]:
    if qkv.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"qkv dtype must be float16 or bfloat16, got {qkv.dtype}")
    if qkv.ndim == 5:
        if qkv.shape[2] != 3:
            raise ValueError(f"explicit qkv layout must have shape [B,S,3,H,D], got {qkv.shape}")
        B, S, _, H_infer, D_infer = qkv.shape
        return B, S, H_infer, D_infer, 1
    if qkv.ndim == 3:
        if H is None or D is None:
            raise ValueError("H and D are required for packed qkv layout [B,S,3*H*D]")
        B, S, packed = qkv.shape
        if packed != 3 * H * D:
            raise ValueError(
                f"packed qkv last dimension must equal 3*H*D={3 * H * D}, got {packed}"
            )
        return B, S, H, D, 0
    raise ValueError(f"qkv must have shape [B,S,3*H*D] or [B,S,3,H,D], got {qkv.shape}")


def _qkv_output_shape(B: int, S: int, H: int, D: int) -> tuple[int, int, int, int]:
    return (B, S, H, D)


def _extract_qkv_components(qkv: mx.array, H: int, D: int, input_layout: int) -> tuple[mx.array, mx.array, mx.array]:
    if input_layout == 1:
        return qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
    qkv_reshaped = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, H, D)
    return qkv_reshaped[:, :, 0, :, :], qkv_reshaped[:, :, 1, :, :], qkv_reshaped[:, :, 2, :, :]


def reference_qkv_split(qkv: mx.array, H: int | None = None, D: int | None = None):
    B, S, H_val, D_val, input_layout = _validate_qkv_input(qkv, H, D)
    q, k, v = _extract_qkv_components(qkv, H_val, D_val, input_layout)
    return q.astype(qkv.dtype), k.astype(qkv.dtype), v.astype(qkv.dtype)


def qkv_split(qkv: mx.array, H: int | None = None, D: int | None = None, *, backend: str = "auto"):
    B, S, H_val, D_val, input_layout = _validate_qkv_input(qkv, H, D)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_qkv_split(qkv, H=H_val, D=D_val)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = qkv.dtype
    source = _load_source(_QKV_SPLIT_KERNEL)
    header = _make_header(dtype)
    kernel = _get_split_kernel(str(dtype), source, header)
    meta = mx.array([B, S, H_val, D_val, input_layout], dtype=mx.int32)
    out_shape = _qkv_output_shape(B, S, H_val, D_val)
    outputs = kernel(
        inputs=[qkv, meta],
        output_shapes=[out_shape, out_shape, out_shape],
        output_dtypes=[dtype, dtype, dtype],
        grid=(B * S * H_val * D_val, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return outputs[0], outputs[1], outputs[2]


def _validate_rope_inputs(
    qkv: mx.array,
    cos: mx.array,
    sin: mx.array,
    H: int | None,
    D: int | None,
    position_offset: int,
) -> tuple[int, int, int, int, int, int]:
    B, S, H_val, D_val, input_layout = _validate_qkv_input(qkv, H, D)
    if D_val % 2 != 0:
        raise ValueError(f"RoPE requires even D, got D={D_val}")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError(f"cos and sin must be matching 2-D arrays, got {cos.shape}, {sin.shape}")
    if cos.shape[1] != D_val // 2:
        raise ValueError(f"cos/sin last dim must be D/2={D_val // 2}, got {cos.shape[1]}")
    if position_offset < 0 or position_offset + S > cos.shape[0]:
        raise ValueError(
            f"position_offset + S must fit in cos/sin rows, got offset={position_offset}, S={S}, rows={cos.shape[0]}"
        )
    return B, S, H_val, D_val, input_layout, cos.shape[0]


def reference_qkv_split_rope(
    qkv: mx.array,
    cos: mx.array,
    sin: mx.array,
    H: int | None = None,
    D: int | None = None,
    *,
    position_offset: int = 0,
):
    B, S, H_val, D_val, input_layout, _ = _validate_rope_inputs(qkv, cos, sin, H, D, position_offset)
    q, k, v = reference_qkv_split(qkv, H=H_val, D=D_val)
    q_rope = reference_apply_rope(q, cos, sin, position_offset=position_offset)
    k_rope = reference_apply_rope(k, cos, sin, position_offset=position_offset)
    return q_rope, k_rope, v


def qkv_split_rope(
    qkv: mx.array,
    cos: mx.array,
    sin: mx.array,
    H: int | None = None,
    D: int | None = None,
    *,
    position_offset: int = 0,
    backend: str = "auto",
):
    B, S, H_val, D_val, input_layout, cos_rows = _validate_rope_inputs(qkv, cos, sin, H, D, position_offset)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_qkv_split_rope(qkv, cos, sin, H=H_val, D=D_val, position_offset=position_offset)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    dtype = qkv.dtype
    source = _load_source(_QKV_SPLIT_ROPE_KERNEL)
    header = _make_header(dtype)
    kernel = _get_split_rope_kernel(str(dtype), source, header)
    meta = mx.array([B, S, H_val, D_val, cos_rows, position_offset, input_layout], dtype=mx.int32)
    out_shape = _qkv_output_shape(B, S, H_val, D_val)
    pair_count = B * S * H_val * (D_val // 2)
    outputs = kernel(
        inputs=[qkv, cos.astype(mx.float32), sin.astype(mx.float32), meta],
        output_shapes=[out_shape, out_shape, out_shape],
        output_dtypes=[dtype, dtype, dtype],
        grid=(pair_count, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return outputs[0], outputs[1], outputs[2]
