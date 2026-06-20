import mlx.core as mx
import pytest

from ops.paged_kv_ops import allocate_paged_kv_cache
from ops.toy_transformer_ops import (
    make_toy_layer_weights,
    paged_toy_transformer_decode_layer,
    reference_paged_toy_transformer_decode_layer,
    reference_toy_transformer_decode_layer,
    toy_transformer_decode_layer,
)


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 8e-2, 8e-2
    return 6e-2, 6e-2


@pytest.mark.parametrize(("bits", "B", "K", "H", "D", "MAX_S", "T", "dtype"), [(4, 1, 64, 2, 16, 8, 4, mx.float16), (4, 1, 64, 2, 16, 8, 4, mx.bfloat16), (8, 1, 64, 2, 16, 8, 4, mx.float16)])
def test_toy_transformer_decode_layer_matches_reference(bits, B, K, H, D, MAX_S, T, dtype):
    mx.random.seed(213)
    weights = make_toy_layer_weights(K, K * 2, bits=bits, group_size=32, num_attention_heads=H, head_dim=D)
    K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    ref_K = K_cache
    ref_V = V_cache
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    atol, rtol = _tol(dtype)

    for pos in range(T):
        x = mx.random.normal((B, 1, K)).astype(dtype)
        got, K_cache, V_cache = toy_transformer_decode_layer(
            x,
            weights["attn_norm_weight"].astype(dtype),
            weights["ffn_norm_weight"].astype(dtype),
            weights["qkv_w"],
            weights["qkv_scales"],
            weights["out_w"],
            weights["out_scales"],
            weights["gate_w"],
            weights["gate_scales"],
            weights["up_w"],
            weights["up_scales"],
            weights["down_w"],
            weights["down_scales"],
            K_cache,
            V_cache,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=32,
            H=H,
            D=D,
            matvec_backend="metal_parallel",
            block_backend="metal",
        )
        ref, ref_K, ref_V = reference_toy_transformer_decode_layer(
            x,
            weights["attn_norm_weight"].astype(dtype),
            weights["ffn_norm_weight"].astype(dtype),
            weights["qkv_w"],
            weights["qkv_scales"],
            weights["out_w"],
            weights["out_scales"],
            weights["gate_w"],
            weights["gate_scales"],
            weights["up_w"],
            weights["up_scales"],
            weights["down_w"],
            weights["down_scales"],
            ref_K,
            ref_V,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=32,
            H=H,
            D=D,
        )
        mx.eval(got, K_cache, V_cache, ref, ref_K, ref_V)
        assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
        assert mx.allclose(K_cache, ref_K, atol=atol, rtol=rtol).item()
        assert mx.allclose(V_cache, ref_V, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize(("bits", "B", "K", "H", "D", "MAX_S", "PAGE_SIZE", "T", "dtype"), [(4, 1, 64, 2, 16, 8, 4, 4, mx.float16), (8, 1, 64, 2, 16, 8, 4, 4, mx.float16)])
def test_paged_toy_transformer_decode_layer_matches_reference(bits, B, K, H, D, MAX_S, PAGE_SIZE, T, dtype):
    mx.random.seed(214)
    weights = make_toy_layer_weights(K, K * 2, bits=bits, group_size=32, num_attention_heads=H, head_dim=D)
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    ref_K = K_pages
    ref_V = V_pages
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    atol, rtol = _tol(dtype)

    for pos in range(T):
        x = mx.random.normal((B, 1, K)).astype(dtype)
        got, K_pages, V_pages = paged_toy_transformer_decode_layer(
            x,
            weights["attn_norm_weight"].astype(dtype),
            weights["ffn_norm_weight"].astype(dtype),
            weights["qkv_w"],
            weights["qkv_scales"],
            weights["out_w"],
            weights["out_scales"],
            weights["gate_w"],
            weights["gate_scales"],
            weights["up_w"],
            weights["up_scales"],
            weights["down_w"],
            weights["down_scales"],
            K_pages,
            V_pages,
            block_table,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=32,
            H=H,
            D=D,
            matvec_backend="metal_parallel",
            block_backend="metal",
        )
        ref, ref_K, ref_V = reference_paged_toy_transformer_decode_layer(
            x,
            weights["attn_norm_weight"].astype(dtype),
            weights["ffn_norm_weight"].astype(dtype),
            weights["qkv_w"],
            weights["qkv_scales"],
            weights["out_w"],
            weights["out_scales"],
            weights["gate_w"],
            weights["gate_scales"],
            weights["up_w"],
            weights["up_scales"],
            weights["down_w"],
            weights["down_scales"],
            ref_K,
            ref_V,
            block_table,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=32,
            H=H,
            D=D,
        )
        mx.eval(got, K_pages, V_pages, ref, ref_K, ref_V)
        assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
        assert mx.allclose(K_pages, ref_K, atol=atol, rtol=rtol).item()
        assert mx.allclose(V_pages, ref_V, atol=atol, rtol=rtol).item()
