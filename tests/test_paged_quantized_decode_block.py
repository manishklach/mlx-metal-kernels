import mlx.core as mx
import pytest

from ops.paged_kv_ops import allocate_paged_kv_cache
from ops.quant_ops import pack_q4
from ops.quantized_decode_block_ops import (
    paged_quantized_decode_block,
    reference_paged_quantized_decode_block,
)


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


@pytest.mark.parametrize(
    ("bits", "B", "K", "H", "D", "MAX_S", "PAGE_SIZE", "T", "dtype"),
    [
        (4, 1, 64, 2, 16, 8, 4, 4, mx.float16),
        (8, 1, 64, 2, 16, 8, 4, 4, mx.float16),
        (4, 2, 128, 4, 16, 8, 4, 4, mx.float16),
    ],
)
def test_paged_quantized_decode_block_matches_reference(bits, B, K, H, D, MAX_S, PAGE_SIZE, T, dtype):
    mx.random.seed(204)
    group_size = 32
    qkv_w, qkv_scales = _make_quantized_weights(bits, 3 * H * D, K, group_size)
    out_w, out_scales = _make_quantized_weights(bits, K, H * D, group_size)
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    ref_K = K_pages
    ref_V = V_pages
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    atol, rtol = _block_tolerances(dtype)

    for pos in range(T):
        x = mx.random.normal((B, 1, K)).astype(dtype)
        got_y, K_pages, V_pages = paged_quantized_decode_block(
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
            pos,
            bits=bits,
            group_size=group_size,
            H=H,
            D=D,
            matvec_backend="metal_parallel",
            block_backend="metal",
        )
        ref_y, ref_K, ref_V = reference_paged_quantized_decode_block(
            x,
            qkv_w,
            qkv_scales,
            out_w,
            out_scales,
            ref_K,
            ref_V,
            block_table,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=group_size,
            H=H,
            D=D,
        )
        mx.eval(got_y, K_pages, V_pages, ref_y, ref_K, ref_V)
        assert mx.allclose(got_y, ref_y, atol=atol, rtol=rtol).item()
        assert mx.allclose(K_pages, ref_K, atol=atol, rtol=rtol).item()
        assert mx.allclose(V_pages, ref_V, atol=atol, rtol=rtol).item()
