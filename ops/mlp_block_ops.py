from __future__ import annotations

import math
import os

import mlx.core as mx

from .activation_ops import fused_swiglu, reference_swiglu, swiglu
from .fused_ops import reference_residual_add, residual_add
from .norm_ops import reference_rms_norm, rms_norm
from .quant_ops import (
    q4_gate_up_matvec_tiled,
    q4_matvec_decode,
    q8_gate_up_matvec_tiled,
    q8_matvec_decode,
    reference_q4_matvec_decode,
    reference_q8_matvec_decode,
)


def _validate_bits(bits: int) -> int:
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    return bits


def _normalize_hidden_input(x: mx.array) -> tuple[mx.array, tuple[int, ...]]:
    if x.ndim == 2:
        return x, x.shape
    if x.ndim == 3:
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2]), x.shape
    raise ValueError(f"x must have shape [B,K], [B,1,K], or [B,S,K], got {x.shape}")


def _restore_hidden_output(y2d: mx.array, original_shape: tuple[int, ...]) -> mx.array:
    if len(original_shape) == 2:
        return y2d
    return y2d.reshape(original_shape[:-1] + (y2d.shape[-1],))


def _validate_quantized_linear_shapes(
    x2d: mx.array,
    w: mx.array,
    scales: mx.array,
    zeros: mx.array | None,
    *,
    bits: int,
    group_size: int,
    name: str,
) -> tuple[int, int]:
    _validate_bits(bits)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if x2d.ndim != 2:
        raise ValueError(f"{name} input must normalize to [rows,K], got {x2d.shape}")
    if w.ndim != 2:
        raise ValueError(f"{name} weights must be 2-D, got {w.shape}")

    in_dim = x2d.shape[1]
    out_dim = w.shape[0]
    if bits == 4:
        expected_cols = math.ceil(in_dim / 2)
        if w.shape[1] != expected_cols:
            raise ValueError(f"{name} q4 weights must have shape [{out_dim},{expected_cols}], got {w.shape}")
    else:
        if w.shape[1] != in_dim:
            raise ValueError(f"{name} q8 weights must have shape [{out_dim},{in_dim}], got {w.shape}")

    groups = math.ceil(in_dim / group_size)
    if scales.shape != (out_dim, groups):
        raise ValueError(f"{name} scales must have shape {(out_dim, groups)}, got {scales.shape}")
    if zeros is not None and zeros.shape != (out_dim, groups):
        raise ValueError(f"{name} zeros must have shape {(out_dim, groups)}, got {zeros.shape}")
    return in_dim, out_dim


def quantized_linear(
    x,
    w,
    scales,
    zeros=None,
    *,
    bits=4,
    group_size=32,
    backend="auto",
):
    x2d, original_shape = _normalize_hidden_input(x)
    _validate_quantized_linear_shapes(x2d, w, scales, zeros, bits=bits, group_size=group_size, name="quantized_linear")
    if bits == 4:
        y2d = (
            reference_q4_matvec_decode(x2d, w, scales, zeros, group_size=group_size)
            if backend == "reference"
            else q4_matvec_decode(x2d, w, scales, zeros, group_size=group_size, backend=backend)
        )
    else:
        y2d = (
            reference_q8_matvec_decode(x2d, w, scales, zeros, group_size=group_size)
            if backend == "reference"
            else q8_matvec_decode(x2d, w, scales, zeros, group_size=group_size, backend=backend)
        )
    return _restore_hidden_output(y2d, original_shape)


def swiglu_down_project(
    gate,
    up,
    down_w,
    down_scales,
    down_zeros=None,
    *,
    bits=4,
    group_size=32,
    activation_backend="metal",
    matvec_backend="metal_tiled",
):
    hidden = reference_swiglu(gate, up) if activation_backend == "reference" else swiglu(gate, up, backend=activation_backend)
    return quantized_linear(
        hidden,
        down_w,
        down_scales,
        down_zeros,
        bits=bits,
        group_size=group_size,
        backend=matvec_backend,
    )


def quantized_gate_up_projection(
    x,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    *,
    gate_zeros=None,
    up_zeros=None,
    bits=4,
    group_size=32,
    backend="metal_tiled",
):
    _validate_bits(bits)
    if backend == "reference":
        gate = quantized_linear(
            x,
            gate_w,
            gate_scales,
            gate_zeros,
            bits=bits,
            group_size=group_size,
            backend="reference",
        )
        up = quantized_linear(
            x,
            up_w,
            up_scales,
            up_zeros,
            bits=bits,
            group_size=group_size,
            backend="reference",
        )
        return gate, up
    if os.environ.get("MLX_METAL_CI_SAFE_MODE", "0") == "1":
        gate = quantized_linear(
            x,
            gate_w,
            gate_scales,
            gate_zeros,
            bits=bits,
            group_size=group_size,
            backend="reference",
        )
        up = quantized_linear(
            x,
            up_w,
            up_scales,
            up_zeros,
            bits=bits,
            group_size=group_size,
            backend="reference",
        )
        return gate, up
    if backend == "metal_tiled":
        gate = quantized_linear(
            x,
            gate_w,
            gate_scales,
            gate_zeros,
            bits=bits,
            group_size=group_size,
            backend="metal_tiled",
        )
        up = quantized_linear(
            x,
            up_w,
            up_scales,
            up_zeros,
            bits=bits,
            group_size=group_size,
            backend="metal_tiled",
        )
        return gate, up
    if backend != "metal_gate_up_tiled":
        raise ValueError(
            "backend must be one of 'reference', 'metal_tiled', 'metal_gate_up_tiled'"
        )
    if bits == 4:
        return q4_gate_up_matvec_tiled(
            x,
            gate_w,
            up_w,
            gate_scales,
            up_scales,
            gate_zeros,
            up_zeros,
            group_size=group_size,
        )
    if bits == 8:
        return q8_gate_up_matvec_tiled(
            x,
            gate_w,
            up_w,
            gate_scales,
            up_scales,
            gate_zeros,
            up_zeros,
            group_size=group_size,
        )
    raise ValueError(f"metal_gate_up_tiled supports only bits in {{4, 8}}, got {bits}")


def _resolve_mlp_backends(
    *,
    backend_preset=None,
    norm_backend="metal",
    matvec_backend="metal_tiled",
    activation_backend="metal",
    residual_backend="metal",
):
    if backend_preset is None:
        return norm_backend, matvec_backend, activation_backend, residual_backend
    mapping = {
        "reference": ("reference", "reference", "reference", "reference"),
        "metal": ("metal", "metal_tiled", "metal", "metal"),
        "parallel": ("metal", "metal_parallel", "metal", "metal"),
        "tiled": ("metal", "metal_tiled", "metal", "metal"),
        "fused_experimental": ("metal", "metal_gate_up_tiled", "metal_fused", "metal"),
    }
    if backend_preset not in mapping:
        raise ValueError(f"backend_preset must be one of {tuple(mapping)}, got {backend_preset}")
    return mapping[backend_preset]


def reference_quantized_mlp_block(
    x,
    residual,
    norm_weight,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    *,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    eps=1e-5,
    return_intermediates=False,
):
    return quantized_mlp_block(
        x,
        residual,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        down_zeros=down_zeros,
        bits=bits,
        group_size=group_size,
        eps=eps,
        norm_backend="reference",
        matvec_backend="reference",
        activation_backend="reference",
        residual_backend="reference",
        return_intermediates=return_intermediates,
    )


def quantized_mlp_block(
    x,
    residual,
    norm_weight,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    *,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    eps=1e-5,
    backend_preset=None,
    norm_backend="metal",
    matvec_backend="metal_tiled",
    activation_backend="metal",
    residual_backend="metal",
    return_intermediates=False,
):
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have identical shapes, got {x.shape}, {residual.shape}")
    if x.ndim != 3:
        raise ValueError(f"x and residual must have shape [B,S,D], got {x.shape}")
    if norm_weight.shape != (x.shape[-1],):
        raise ValueError(f"norm_weight must have shape {(x.shape[-1],)}, got {norm_weight.shape}")
    norm_backend, matvec_backend, activation_backend, residual_backend = _resolve_mlp_backends(
        backend_preset=backend_preset,
        norm_backend=norm_backend,
        matvec_backend=matvec_backend,
        activation_backend=activation_backend,
        residual_backend=residual_backend,
    )

    z = reference_residual_add(x, residual) if residual_backend == "reference" else residual_add(x, residual, backend=residual_backend)
    normed = reference_rms_norm(z, norm_weight, eps=eps) if norm_backend == "reference" else rms_norm(z, norm_weight, eps=eps, backend=norm_backend)
    gate, up = quantized_gate_up_projection(
        normed,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        bits=bits,
        group_size=group_size,
        backend=matvec_backend,
    )
    if activation_backend == "reference":
        mlp = reference_swiglu(gate, up)
    elif activation_backend == "metal_fused":
        mlp = fused_swiglu(gate, up, backend="metal_fused")
    else:
        mlp = swiglu(gate, up, backend=activation_backend)
    down_backend = "metal_tiled" if backend_preset == "fused_experimental" else matvec_backend
    down = quantized_linear(mlp, down_w, down_scales, down_zeros, bits=bits, group_size=group_size, backend=down_backend)
    out = reference_residual_add(z, down) if residual_backend == "reference" else residual_add(z, down, backend=residual_backend)
    if not return_intermediates:
        return out
    return out, {"z": z, "normed": normed, "gate": gate, "up": up, "mlp": mlp, "down": down}


def quantized_mlp_decode_step(
    x,
    norm_weight,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    *,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    eps=1e-5,
    backend_preset="tiled",
):
    if backend_preset == "parallel":
        backend_preset = "tiled"
    return quantized_mlp_block(
        x,
        x,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        down_zeros=down_zeros,
        bits=bits,
        group_size=group_size,
        eps=eps,
        backend_preset=backend_preset,
    )
