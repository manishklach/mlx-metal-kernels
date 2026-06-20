from __future__ import annotations

import mlx.core as mx

from .activation_ops import reference_swiglu, swiglu
from .fused_ops import reference_residual_add, residual_add
from .norm_ops import reference_rms_norm, rms_norm
from .paged_kv_ops import allocate_paged_kv_cache
from .quantized_decode_block_ops import (
    normalize_hidden_input,
    paged_quantized_decode_block,
    quantized_decode_block,
    quantized_output_projection,
    reference_paged_quantized_decode_block,
    reference_quantized_decode_block,
    reference_quantized_output_projection,
    validate_quantized_weight_shapes,
)


def _normalize_token_input(x: mx.array) -> tuple[mx.array, int, int]:
    x2d, B, _ = normalize_hidden_input(x)
    return x2d.reshape(B, 1, x2d.shape[1]), B, x2d.shape[1]


def _validate_mlp_weights(
    x3d: mx.array,
    gate_w: mx.array,
    gate_scales: mx.array,
    gate_zeros: mx.array | None,
    up_w: mx.array,
    up_scales: mx.array,
    up_zeros: mx.array | None,
    down_w: mx.array,
    down_scales: mx.array,
    down_zeros: mx.array | None,
    *,
    bits: int,
    group_size: int,
) -> int:
    x2d = x3d[:, 0, :]
    _, gate_out = validate_quantized_weight_shapes(
        x2d, gate_w, gate_scales, gate_zeros, bits=bits, group_size=group_size, name="gate_projection"
    )
    _, up_out = validate_quantized_weight_shapes(
        x2d, up_w, up_scales, up_zeros, bits=bits, group_size=group_size, name="up_projection"
    )
    if gate_out != up_out:
        raise ValueError(f"gate and up projection output dims must match, got {gate_out}, {up_out}")
    validate_quantized_weight_shapes(
        mx.zeros((x2d.shape[0], gate_out), dtype=x2d.dtype),
        down_w,
        down_scales,
        down_zeros,
        bits=bits,
        group_size=group_size,
        expected_in_dim=gate_out,
        expected_out_dim=x2d.shape[1],
        name="down_projection",
    )
    return gate_out


def reference_toy_transformer_decode_layer(
    x,
    attn_norm_weight,
    ffn_norm_weight,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    K_cache,
    V_cache,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    eps: float = 1.0e-5,
):
    return toy_transformer_decode_layer(
        x,
        attn_norm_weight,
        ffn_norm_weight,
        qkv_w,
        qkv_scales,
        out_w,
        out_scales,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        K_cache,
        V_cache,
        cos,
        sin,
        position,
        qkv_zeros=qkv_zeros,
        out_zeros=out_zeros,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        down_zeros=down_zeros,
        bits=bits,
        group_size=group_size,
        H=H,
        D=D,
        scale=scale,
        eps=eps,
        matvec_backend="reference",
        block_backend="reference",
        norm_backend="reference",
        activation_backend="reference",
        residual_backend="reference",
    )


def toy_transformer_decode_layer(
    x,
    attn_norm_weight,
    ffn_norm_weight,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    K_cache,
    V_cache,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    eps: float = 1.0e-5,
    matvec_backend="metal_parallel",
    block_backend="auto",
    norm_backend="metal",
    activation_backend="metal",
    residual_backend="metal",
):
    x3d, _, hidden_dim = _normalize_token_input(x)
    if attn_norm_weight.shape != (hidden_dim,):
        raise ValueError(f"attn_norm_weight must have shape {(hidden_dim,)}, got {attn_norm_weight.shape}")
    if ffn_norm_weight.shape != (hidden_dim,):
        raise ValueError(f"ffn_norm_weight must have shape {(hidden_dim,)}, got {ffn_norm_weight.shape}")
    _validate_mlp_weights(
        x3d,
        gate_w,
        gate_scales,
        gate_zeros,
        up_w,
        up_scales,
        up_zeros,
        down_w,
        down_scales,
        down_zeros,
        bits=bits,
        group_size=group_size,
    )

    attn_in = (
        reference_rms_norm(x3d, attn_norm_weight, eps=eps)
        if norm_backend == "reference"
        else rms_norm(x3d, attn_norm_weight, eps=eps, backend=norm_backend)
    )
    if matvec_backend == "reference" and block_backend == "reference":
        attn_out, updated_K, updated_V = reference_quantized_decode_block(
            attn_in,
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
        )
    else:
        attn_out, updated_K, updated_V = quantized_decode_block(
            attn_in,
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
            matvec_backend=matvec_backend,
            block_backend=block_backend,
        )
    post_attn = (
        reference_residual_add(attn_out, x3d)
        if residual_backend == "reference"
        else residual_add(attn_out, x3d, backend=residual_backend)
    )
    ffn_in = (
        reference_rms_norm(post_attn, ffn_norm_weight, eps=eps)
        if norm_backend == "reference"
        else rms_norm(post_attn, ffn_norm_weight, eps=eps, backend=norm_backend)
    )
    if matvec_backend == "reference":
        gate = reference_quantized_output_projection(ffn_in, gate_w, gate_scales, gate_zeros, bits=bits, group_size=group_size)
        up = reference_quantized_output_projection(ffn_in, up_w, up_scales, up_zeros, bits=bits, group_size=group_size)
    else:
        gate = quantized_output_projection(ffn_in, gate_w, gate_scales, gate_zeros, bits=bits, group_size=group_size, backend=matvec_backend)
        up = quantized_output_projection(ffn_in, up_w, up_scales, up_zeros, bits=bits, group_size=group_size, backend=matvec_backend)
    hidden = reference_swiglu(gate, up) if activation_backend == "reference" else swiglu(gate, up, backend=activation_backend)
    down = (
        reference_quantized_output_projection(hidden, down_w, down_scales, down_zeros, bits=bits, group_size=group_size)
        if matvec_backend == "reference"
        else quantized_output_projection(hidden, down_w, down_scales, down_zeros, bits=bits, group_size=group_size, backend=matvec_backend)
    )
    out = (
        reference_residual_add(down, post_attn)
        if residual_backend == "reference"
        else residual_add(down, post_attn, backend=residual_backend)
    )
    return out, updated_K, updated_V


def reference_paged_toy_transformer_decode_layer(
    x,
    attn_norm_weight,
    ffn_norm_weight,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    K_pages,
    V_pages,
    block_table,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    eps: float = 1.0e-5,
):
    return paged_toy_transformer_decode_layer(
        x,
        attn_norm_weight,
        ffn_norm_weight,
        qkv_w,
        qkv_scales,
        out_w,
        out_scales,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        K_pages,
        V_pages,
        block_table,
        cos,
        sin,
        position,
        qkv_zeros=qkv_zeros,
        out_zeros=out_zeros,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        down_zeros=down_zeros,
        bits=bits,
        group_size=group_size,
        H=H,
        D=D,
        scale=scale,
        eps=eps,
        matvec_backend="reference",
        block_backend="reference",
        norm_backend="reference",
        activation_backend="reference",
        residual_backend="reference",
    )


def paged_toy_transformer_decode_layer(
    x,
    attn_norm_weight,
    ffn_norm_weight,
    qkv_w,
    qkv_scales,
    out_w,
    out_scales,
    gate_w,
    gate_scales,
    up_w,
    up_scales,
    down_w,
    down_scales,
    K_pages,
    V_pages,
    block_table,
    cos,
    sin,
    position,
    *,
    qkv_zeros=None,
    out_zeros=None,
    gate_zeros=None,
    up_zeros=None,
    down_zeros=None,
    bits=4,
    group_size=32,
    H=None,
    D=None,
    scale=None,
    eps: float = 1.0e-5,
    matvec_backend="metal_parallel",
    block_backend="auto",
    norm_backend="metal",
    activation_backend="metal",
    residual_backend="metal",
):
    x3d, _, hidden_dim = _normalize_token_input(x)
    if attn_norm_weight.shape != (hidden_dim,):
        raise ValueError(f"attn_norm_weight must have shape {(hidden_dim,)}, got {attn_norm_weight.shape}")
    if ffn_norm_weight.shape != (hidden_dim,):
        raise ValueError(f"ffn_norm_weight must have shape {(hidden_dim,)}, got {ffn_norm_weight.shape}")
    _validate_mlp_weights(
        x3d,
        gate_w,
        gate_scales,
        gate_zeros,
        up_w,
        up_scales,
        up_zeros,
        down_w,
        down_scales,
        down_zeros,
        bits=bits,
        group_size=group_size,
    )

    attn_in = (
        reference_rms_norm(x3d, attn_norm_weight, eps=eps)
        if norm_backend == "reference"
        else rms_norm(x3d, attn_norm_weight, eps=eps, backend=norm_backend)
    )
    if matvec_backend == "reference" and block_backend == "reference":
        attn_out, updated_K, updated_V = reference_paged_quantized_decode_block(
            attn_in,
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
        )
    else:
        attn_out, updated_K, updated_V = paged_quantized_decode_block(
            attn_in,
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
            matvec_backend=matvec_backend,
            block_backend=block_backend,
        )
    post_attn = (
        reference_residual_add(attn_out, x3d)
        if residual_backend == "reference"
        else residual_add(attn_out, x3d, backend=residual_backend)
    )
    ffn_in = (
        reference_rms_norm(post_attn, ffn_norm_weight, eps=eps)
        if norm_backend == "reference"
        else rms_norm(post_attn, ffn_norm_weight, eps=eps, backend=norm_backend)
    )
    if matvec_backend == "reference":
        gate = reference_quantized_output_projection(ffn_in, gate_w, gate_scales, gate_zeros, bits=bits, group_size=group_size)
        up = reference_quantized_output_projection(ffn_in, up_w, up_scales, up_zeros, bits=bits, group_size=group_size)
    else:
        gate = quantized_output_projection(ffn_in, gate_w, gate_scales, gate_zeros, bits=bits, group_size=group_size, backend=matvec_backend)
        up = quantized_output_projection(ffn_in, up_w, up_scales, up_zeros, bits=bits, group_size=group_size, backend=matvec_backend)
    hidden = reference_swiglu(gate, up) if activation_backend == "reference" else swiglu(gate, up, backend=activation_backend)
    down = (
        reference_quantized_output_projection(hidden, down_w, down_scales, down_zeros, bits=bits, group_size=group_size)
        if matvec_backend == "reference"
        else quantized_output_projection(hidden, down_w, down_scales, down_zeros, bits=bits, group_size=group_size, backend=matvec_backend)
    )
    out = (
        reference_residual_add(down, post_attn)
        if residual_backend == "reference"
        else residual_add(down, post_attn, backend=residual_backend)
    )
    return out, updated_K, updated_V


def make_toy_layer_weights(hidden_dim: int, intermediate_dim: int, *, bits: int, group_size: int):
    groups_hidden = (hidden_dim + group_size - 1) // group_size
    groups_intermediate = (intermediate_dim + group_size - 1) // group_size
    qkv_out = hidden_dim * 3
    q_range = 16 if bits == 4 else 255

    def _q(shape):
        return (mx.random.uniform(shape) * q_range).astype(mx.uint8)

    def _maybe_pack(q):
        from .quant_ops import pack_q4

        return pack_q4(q) if bits == 4 else q

    return {
        "attn_norm_weight": mx.ones((hidden_dim,), dtype=mx.float16),
        "ffn_norm_weight": mx.ones((hidden_dim,), dtype=mx.float16),
        "qkv_w": _maybe_pack(_q((qkv_out, hidden_dim))),
        "qkv_scales": mx.random.normal((qkv_out, groups_hidden)).astype(mx.float32),
        "out_w": _maybe_pack(_q((hidden_dim, hidden_dim))),
        "out_scales": mx.random.normal((hidden_dim, groups_hidden)).astype(mx.float32),
        "gate_w": _maybe_pack(_q((intermediate_dim, hidden_dim))),
        "gate_scales": mx.random.normal((intermediate_dim, groups_hidden)).astype(mx.float32),
        "up_w": _maybe_pack(_q((intermediate_dim, hidden_dim))),
        "up_scales": mx.random.normal((intermediate_dim, groups_hidden)).astype(mx.float32),
        "down_w": _maybe_pack(_q((hidden_dim, intermediate_dim))),
        "down_scales": mx.random.normal((hidden_dim, groups_intermediate)).astype(mx.float32),
    }


def allocate_toy_paged_cache(B: int, MAX_S: int, H: int, D: int, PAGE_SIZE: int, dtype):
    return allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
