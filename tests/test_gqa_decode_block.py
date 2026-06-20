import mlx.core as mx

from models.llama_config import tiny_gqa_debug_config
from models.model_adapter import KernelBackendConfig, LlamaLikeKernelAdapter
from ops.gqa_ops import (
    gqa_decode_block_from_qkv,
    paged_gqa_decode_block_from_qkv,
    reference_gqa_decode_block_from_qkv,
    reference_paged_gqa_decode_block_from_qkv,
)
from ops.paged_kv_ops import allocate_paged_kv_cache


def test_gqa_decode_block_matches_reference():
    mx.random.seed(244)
    B, T, MAX_S, Hq, Hkv, D = 1, 4, 8, 4, 2, 16
    K_cache = mx.zeros((B, MAX_S, Hkv, D), dtype=mx.float16)
    V_cache = mx.zeros((B, MAX_S, Hkv, D), dtype=mx.float16)
    ref_K = K_cache
    ref_V = V_cache
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    for pos in range(T):
        qkv = mx.random.normal((B, 1, Hq * D + 2 * Hkv * D)).astype(mx.float16)
        got, K_cache, V_cache = gqa_decode_block_from_qkv(
            qkv, K_cache, V_cache, cos, sin, pos, num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=D, backend="reference"
        )
        ref, ref_K, ref_V = reference_gqa_decode_block_from_qkv(
            qkv, ref_K, ref_V, cos, sin, pos, num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=D
        )
        mx.eval(got, K_cache, V_cache, ref, ref_K, ref_V)
        assert mx.allclose(got, ref, atol=6e-2, rtol=6e-2).item()
        assert mx.allclose(K_cache, ref_K, atol=6e-2, rtol=6e-2).item()
        assert mx.allclose(V_cache, ref_V, atol=6e-2, rtol=6e-2).item()


def test_paged_gqa_decode_block_matches_reference():
    mx.random.seed(245)
    B, T, MAX_S, PAGE_SIZE, Hq, Hkv, D = 1, 4, 8, 4, 4, 2, 16
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, Hkv, D, PAGE_SIZE, mx.float16)
    ref_K = K_pages
    ref_V = V_pages
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    for pos in range(T):
        qkv = mx.random.normal((B, 1, Hq * D + 2 * Hkv * D)).astype(mx.float16)
        got, K_pages, V_pages = paged_gqa_decode_block_from_qkv(
            qkv, K_pages, V_pages, block_table, cos, sin, pos, num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=D, backend="reference"
        )
        ref, ref_K, ref_V = reference_paged_gqa_decode_block_from_qkv(
            qkv, ref_K, ref_V, block_table, cos, sin, pos, num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=D
        )
        mx.eval(got, K_pages, V_pages, ref, ref_K, ref_V)
        assert mx.allclose(got, ref, atol=6e-2, rtol=6e-2).item()
        assert mx.allclose(K_pages, ref_K, atol=6e-2, rtol=6e-2).item()
        assert mx.allclose(V_pages, ref_V, atol=6e-2, rtol=6e-2).item()


def test_model_adapter_tiny_gqa_config_supported():
    adapter = LlamaLikeKernelAdapter(tiny_gqa_debug_config(), KernelBackendConfig(use_autotune=False))
    adapter.validate_supported()
    desc = adapter.describe()
    assert desc["gqa_supported"] is True
    assert desc["num_key_value_heads"] == 2
