import mlx.core as mx
import pytest

from ops.quant_ops import pack_q4
from ops.quantized_decode_block_ops import (
    quantized_decode_block,
    quantized_output_projection,
    quantized_qkv_projection,
    reference_quantized_decode_block,
    reference_quantized_output_projection,
    reference_quantized_qkv_projection,
)


def _helper_tolerances(dtype):
    if dtype == mx.bfloat16:
        return 6e-2, 6e-2
    return 4e-2, 4e-2


def _block_tolerances(dtype):
    if dtype == mx.bfloat16:
        return 7e-2, 7e-2
    return 5e-2, 5e-2


def _groups(K, group_size):
    return (K + group_size - 1) // group_size


def _make_quantized_weights(bits, out_dim, in_dim, group_size):
    groups = _groups(in_dim, group_size)
    scales = mx.random.normal((out_dim, groups)).astype(mx.float32)
    if bits == 4:
        q = (mx.random.uniform((out_dim, in_dim)) * 16).astype(mx.uint8)
        return pack_q4(q), scales
    q = (mx.random.uniform((out_dim, in_dim)) * 255).astype(mx.uint8)
    return q, scales


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_quantized_qkv_projection_matches_reference(bits, dtype):
    mx.random.seed(201)
    B, K, H, D, group_size = 1, 64, 2, 16, 32
    x = mx.random.normal((B, 1, K)).astype(dtype)
    qkv_w, qkv_scales = _make_quantized_weights(bits, 3 * H * D, K, group_size)

    got = quantized_qkv_projection(
        x,
        qkv_w,
        qkv_scales,
        bits=bits,
        group_size=group_size,
        backend="metal_parallel",
    )
    ref = reference_quantized_qkv_projection(x, qkv_w, qkv_scales, bits=bits, group_size=group_size)
    mx.eval(got, ref)
    atol, rtol = _helper_tolerances(dtype)
    assert got.shape == (B, 1, 3 * H * D)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_quantized_output_projection_matches_reference(bits, dtype):
    mx.random.seed(202)
    B, K, H, D, group_size = 1, 64, 2, 16, 32
    attn_out = mx.random.normal((B, 1, H, D)).astype(dtype)
    out_w, out_scales = _make_quantized_weights(bits, K, H * D, group_size)

    got = quantized_output_projection(
        attn_out,
        out_w,
        out_scales,
        bits=bits,
        group_size=group_size,
        backend="metal_parallel",
    )
    ref = reference_quantized_output_projection(attn_out, out_w, out_scales, bits=bits, group_size=group_size)
    mx.eval(got, ref)
    atol, rtol = _helper_tolerances(dtype)
    assert got.shape == (B, 1, K)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize(
    ("bits", "B", "K", "H", "D", "MAX_S", "T", "dtype"),
    [
        (4, 1, 64, 2, 16, 8, 4, mx.float16),
        (8, 1, 64, 2, 16, 8, 4, mx.float16),
        (4, 2, 128, 4, 16, 8, 4, mx.float16),
        (4, 1, 64, 2, 16, 8, 4, mx.bfloat16),
    ],
)
def test_quantized_decode_block_matches_reference(bits, B, K, H, D, MAX_S, T, dtype):
    mx.random.seed(203)
    group_size = 32
    qkv_w, qkv_scales = _make_quantized_weights(bits, 3 * H * D, K, group_size)
    out_w, out_scales = _make_quantized_weights(bits, K, H * D, group_size)
    K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    ref_K = K_cache
    ref_V = V_cache
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    atol, rtol = _block_tolerances(dtype)

    for pos in range(T):
        x = mx.random.normal((B, 1, K)).astype(dtype)
        got_y, K_cache, V_cache = quantized_decode_block(
            x,
            qkv_w,
            qkv_scales,
            out_w,
            out_scales,
            K_cache,
            V_cache,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=group_size,
            H=H,
            D=D,
            matvec_backend="metal_parallel",
            block_backend="metal",
        )
        ref_y, ref_K, ref_V = reference_quantized_decode_block(
            x,
            qkv_w,
            qkv_scales,
            out_w,
            out_scales,
            ref_K,
            ref_V,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=group_size,
            H=H,
            D=D,
        )
        mx.eval(got_y, K_cache, V_cache, ref_y, ref_K, ref_V)
        assert mx.allclose(got_y, ref_y, atol=atol, rtol=rtol).item()
        assert mx.allclose(K_cache, ref_K, atol=atol, rtol=rtol).item()
        assert mx.allclose(V_cache, ref_V, atol=atol, rtol=rtol).item()
