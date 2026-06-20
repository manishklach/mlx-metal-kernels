import pytest

mx = pytest.importorskip("mlx.core")

from models.llama_config import LlamaLikeConfig
from ops.llama_layer_ops import create_random_quantized_llama_layer_weights, init_llama_layer_cache, llama_layer_decode_loop


def _make_config(num_kv_heads):
    return LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=8,
    ).validate()


@pytest.mark.parametrize("num_kv_heads", [2, 1])
def test_llama_layer_decode_gqa_and_mqa_match_reference(num_kv_heads):
    mx.random.seed(902 + num_kv_heads)
    cfg = _make_config(num_kv_heads)
    weights = create_random_quantized_llama_layer_weights(cfg, bits=4, dtype=mx.float16, seed=77 + num_kv_heads)
    cache_ref = init_llama_layer_cache(cfg, 1, cfg.max_position_embeddings, dtype=mx.float16)
    cache_opt = init_llama_layer_cache(cfg, 1, cfg.max_position_embeddings, dtype=mx.float16)
    cos = mx.random.normal((cfg.max_position_embeddings + 4, cfg.head_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((cfg.max_position_embeddings + 4, cfg.head_dim // 2)).astype(mx.float32)
    inputs = mx.random.normal((1, 4, cfg.hidden_size)).astype(mx.float16)
    got, got_cache = llama_layer_decode_loop(inputs, weights, cache_opt, cos, sin, cfg, backend_preset="fused_experimental")
    ref, ref_cache = llama_layer_decode_loop(inputs, weights, cache_ref, cos, sin, cfg, backend_preset="reference")
    mx.eval(got, ref, *got_cache, *ref_cache)
    assert mx.allclose(got, ref, atol=1.2e-1, rtol=1.2e-1).item()
    assert mx.allclose(got_cache[0], ref_cache[0], atol=1.2e-1, rtol=1.2e-1).item()
    assert mx.allclose(got_cache[1], ref_cache[1], atol=1.2e-1, rtol=1.2e-1).item()
