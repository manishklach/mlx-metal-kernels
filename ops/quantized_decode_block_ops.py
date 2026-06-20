from __future__ import annotations

import math

import mlx.core as mx

from .decode_block_ops import (
    decode_block_from_qkv,
    paged_decode_block_from_qkv,
)
from .quant_ops import (
    q4_matvec_decode,
    q8_matvec_decode,
    reference_q4_matvec_decode,
    reference_q8_matvec_decode,
)


def validate_bits(bits: int) -> int:
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    return bits


def normalize_hidden_input(x: mx.array) -> tuple[mx.array, int, tuple[int, ...]]:
    if x.ndim == 2:
        return x, x.shape[0], x.shape
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :], x.shape[0], x.shape
    if x.ndim == 3:
        B = x.shape[0]
        return x.reshape(B, x.shape[1] * x.shape[2]), B, x.shape
    if x.ndim == 4 and x.shape[1] == 1:
        B = x.shape[0]
        return x.reshape(B, x.shape[2] * x.shape[3]), B, x.shape
    raise ValueError(f"x must have shape [B,K], [B,1,K], [B,H,D], or [B,1,H,D], got {x.shape}")


def infer_qkv_dims(qkv_w: mx.array, H: int | None, D: int | None, bits: int) -> tuple[int, int]:
    validate_bits(bits)
    if qkv_w.ndim != 2:
        raise ValueError(f"qkv_w must be 2-D, got {qkv_w.shape}")
    qkv_out_dim = qkv_w.shape[0]
    if qkv_out_dim % 3 != 0:
        raise ValueError(f"qkv_w output rows must be divisible by 3, got {qkv_out_dim}")

    if H is not None and D is not None:
        if 3 * H * D != qkv_out_dim:
            raise ValueError(f"qkv_w output rows must equal 3*H*D={3 * H * D}, got {qkv_out_dim}")
        return H, D

    per_proj = qkv_out_dim // 3
    if H is not None:
        if per_proj % H != 0:
            raise ValueError(f"qkv_w output rows imply per-projection dim {per_proj}, not divisible by H={H}")
        return H, per_proj // H
    if D is not None:
        if per_proj % D != 0:
            raise ValueError(f"qkv_w output rows imply per-projection dim {per_proj}, not divisible by D={D}")
        return per_proj // D, D
    raise ValueError("H and D must be provided when qkv dimensions cannot be inferred uniquely")


def validate_quantized_weight_shapes(
    x2d: mx.array,
    w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    bits: int,
    group_size: int,
    expected_in_dim: int | None = None,
    expected_out_dim: int | None = None,
    name: str,
) -> tuple[int, int]:
    validate_bits(bits)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if x2d.ndim != 2:
        raise ValueError(f"{name} input must normalize to [B,K], got {x2d.shape}")
    if w.ndim != 2:
        raise ValueError(f"{name} weights must be 2-D, got {w.shape}")

    in_dim = x2d.shape[1]
    if expected_in_dim is not None and in_dim != expected_in_dim:
        raise ValueError(f"{name} input dim must be {expected_in_dim}, got {in_dim}")

    out_dim = w.shape[0]
    if expected_out_dim is not None and out_dim != expected_out_dim:
        raise ValueError(f"{name} output dim must be {expected_out_dim}, got {out_dim}")

    if bits == 4:
        expected_packed = math.ceil(in_dim / 2)
        if w.shape[1] != expected_packed:
            raise ValueError(f"{name} q4 weights must have shape [N,{expected_packed}], got {w.shape}")
    else:
        if w.shape[1] != in_dim:
            raise ValueError(f"{name} q8 weights must have shape [N,{in_dim}], got {w.shape}")

    groups = math.ceil(in_dim / group_size)
    if scales.shape != (out_dim, groups):
        raise ValueError(f"{name} scales must have shape {(out_dim, groups)}, got {scales.shape}")
    if zeros is not None and zeros.shape != (out_dim, groups):
        raise ValueError(f"{name} zeros must have shape {(out_dim, groups)}, got {zeros.shape}")
    return in_dim, out_dim


def _matvec_dispatch(
    x2d: mx.array,
    w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    bits: int,
    group_size: int,
    backend: str,
    reference: bool,
) -> mx.array:
    validate_bits(bits)
    if reference:
        if bits == 4:
            return reference_q4_matvec_decode(x2d, w, scales, zeros, group_size=group_size)
        return reference_q8_matvec_decode(x2d, w, scales, zeros, group_size=group_size)
    if bits == 4:
        return q4_matvec_decode(x2d, w, scales, zeros, group_size=group_size, backend=backend)
    return q8_matvec_decode(x2d, w, scales, zeros, group_size=group_size, backend=backend)


def reference_quantized_qkv_projection(
    x,
    qkv_w,
    qkv_scales,
    qkv_zeros=None,
    *,
    bits=4,
    group_size=32,
):
    return quantized_qkv_projection(
        x,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        backend="reference",
    )


def quantized_qkv_projection(
    x,
    qkv_w,
    qkv_scales,
    qkv_zeros=None,
    *,
    bits=4,
    group_size=32,
    backend="auto",
):
    x2d, B, _ = normalize_hidden_input(x)
    validate_quantized_weight_shapes(
        x2d,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        name="qkv_projection",
    )
    qkv = _matvec_dispatch(
        x2d,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        backend=backend,
        reference=backend.lower() == "reference",
    )
    return qkv.reshape(B, 1, qkv.shape[1])


def reference_quantized_output_projection(
    x,
    out_w,
    out_scales,
    out_zeros=None,
    *,
    bits=4,
    group_size=32,
):
    return quantized_output_projection(
        x,
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        backend="reference",
    )


def quantized_output_projection(
    x,
    out_w,
    out_scales,
    out_zeros=None,
    *,
    bits=4,
    group_size=32,
    backend="auto",
):
    x2d, B, _ = normalize_hidden_input(x)
    _, out_dim = validate_quantized_weight_shapes(
        x2d,
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        name="output_projection",
    )
    y = _matvec_dispatch(
        x2d,
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        backend=backend,
        reference=backend.lower() == "reference",
    )
    return y.reshape(B, 1, out_dim)


def reference_quantized_decode_block(
    x,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    K_cache,
    V_cache,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    return_intermediates=False,
):
    return quantized_decode_block(
        x,
        qkv_w,
        qkv_scales,
        out_w,
        out_scales,
        K_cache,
        V_cache,
        cos,
        sin,
        position,
        qkv_zeros=qkv_zeros,
        out_zeros=out_zeros,
        bits=bits,
        group_size=group_size,
        H=H,
        D=D,
        scale=scale,
        matvec_backend="reference",
        block_backend="reference",
        return_intermediates=return_intermediates,
    )


def quantized_decode_block(
    x,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    K_cache,
    V_cache,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    matvec_backend="metal_parallel",
    block_backend="auto",
    return_intermediates=False,
):
    x2d, _, _ = normalize_hidden_input(x)
    H, D = infer_qkv_dims(qkv_w, H, D, bits)
    validate_quantized_weight_shapes(
        x2d,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        expected_out_dim=3 * H * D,
        name="qkv_projection",
    )
    validate_quantized_weight_shapes(
        mx.zeros((x2d.shape[0], H * D), dtype=x2d.dtype),
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        expected_in_dim=H * D,
        name="output_projection",
    )

    qkv = quantized_qkv_projection(
        x2d,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        backend=matvec_backend,
    )
    attn_out, updated_K, updated_V = decode_block_from_qkv(
        qkv,
        K_cache,
        V_cache,
        cos,
        sin,
        position,
        H=H,
        D=D,
        scale=scale,
        backend=block_backend,
    )
    projected = quantized_output_projection(
        attn_out,
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        backend=matvec_backend,
    )
    if return_intermediates:
        return projected, updated_K, updated_V, qkv, attn_out
    return projected, updated_K, updated_V


def reference_paged_quantized_decode_block(
    x,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    K_pages,
    V_pages,
    block_table,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    return_intermediates=False,
):
    return paged_quantized_decode_block(
        x,
        qkv_w,
        qkv_scales,
        out_w,
        out_scales,
        K_pages,
        V_pages,
        block_table,
        cos,
        sin,
        position,
        qkv_zeros=qkv_zeros,
        out_zeros=out_zeros,
        bits=bits,
        group_size=group_size,
        H=H,
        D=D,
        scale=scale,
        matvec_backend="reference",
        block_backend="reference",
        return_intermediates=return_intermediates,
    )


def paged_quantized_decode_block(
    x,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    K_pages,
    V_pages,
    block_table,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    matvec_backend="metal_parallel",
    block_backend="auto",
    return_intermediates=False,
):
    x2d, _, _ = normalize_hidden_input(x)
    H, D = infer_qkv_dims(qkv_w, H, D, bits)
    validate_quantized_weight_shapes(
        x2d,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        expected_out_dim=3 * H * D,
        name="qkv_projection",
    )
    validate_quantized_weight_shapes(
        mx.zeros((x2d.shape[0], H * D), dtype=x2d.dtype),
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        expected_in_dim=H * D,
        name="output_projection",
    )

    qkv = quantized_qkv_projection(
        x2d,
        qkv_w,
        qkv_scales,
        qkv_zeros,
        bits=bits,
        group_size=group_size,
        backend=matvec_backend,
    )
    attn_out, updated_K, updated_V = paged_decode_block_from_qkv(
        qkv,
        K_pages,
        V_pages,
        block_table,
        cos,
        sin,
        position,
        H=H,
        D=D,
        scale=scale,
        backend=block_backend,
    )
    projected = quantized_output_projection(
        attn_out,
        out_w,
        out_scales,
        out_zeros,
        bits=bits,
        group_size=group_size,
        backend=matvec_backend,
    )
    if return_intermediates:
        return projected, updated_K, updated_V, qkv, attn_out
    return projected, updated_K, updated_V
