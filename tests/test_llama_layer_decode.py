import pytest

mx = pytest.importorskip("mlx.core")

from models.llama_config import LlamaLikeConfig
from ops.llama_layer_ops import (
    create_random_quantized_llama_layer_weights,
    init_llama_layer_cache,
    llama_layer_decode_loop,
)


def _make_config(*, num_kv_heads=4):
    return LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=8,
    ).validate()


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 1.5e-1, 1.5e-1
    return 1.2e-1, 1.2e-1


@pytest.mark.parametrize(
    ("bits", "B", "T", "backend_preset", "dtype"),
    [
        (4, 1, 4, "fused_experimental", mx.float16),
        (8, 1, 4, "tiled", mx.float16),
        (4, 2, 4, "fused_experimental", mx.float16),
    ],
)
def test_llama_layer_decode_matches_reference(bits, B, T, backend_preset, dtype):
    mx.random.seed(901)
    cfg = _make_config(num_kv_heads=4)
    weights = create_random_quantized_llama_layer_weights(cfg, bits=bits, dtype=dtype, seed=bits + B + T)
    cache_ref = init_llama_layer_cache(cfg, B, cfg.max_position_embeddings, dtype=dtype)
    cache_opt = init_llama_layer_cache(cfg, B, cfg.max_position_embeddings, dtype=dtype)
    cos = mx.random.normal((cfg.max_position_embeddings + 4, cfg.head_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((cfg.max_position_embeddings + 4, cfg.head_dim // 2)).astype(mx.float32)
    inputs = mx.random.normal((B, T, cfg.hidden_size)).astype(dtype)
    got, got_cache = llama_layer_decode_loop(inputs, weights, cache_opt, cos, sin, cfg, backend_preset=backend_preset)
    ref, ref_cache = llama_layer_decode_loop(inputs, weights, cache_ref, cos, sin, cfg, backend_preset="reference")
    mx.eval(got, ref, *got_cache, *ref_cache)
    atol, rtol = _tol(dtype)
    assert got.shape == ref.shape
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
    assert mx.allclose(got_cache[0], ref_cache[0], atol=atol, rtol=rtol).item()
    assert mx.allclose(got_cache[1], ref_cache[1], atol=atol, rtol=rtol).item()


def test_llama_layer_decode_invalid_weight_shape_raises():
    cfg = _make_config(num_kv_heads=4)
    weights = create_random_quantized_llama_layer_weights(cfg, bits=4, dtype=mx.float16, seed=11)
    bad = weights
    bad.qkv_w = bad.qkv_w[:1, :]
    cache = init_llama_layer_cache(cfg, 1, cfg.max_position_embeddings, dtype=mx.float16)
    cos = mx.random.normal((cfg.max_position_embeddings + 4, cfg.head_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((cfg.max_position_embeddings + 4, cfg.head_dim // 2)).astype(mx.float32)
    x = mx.random.normal((1, 1, cfg.hidden_size)).astype(mx.float16)
    with pytest.raises(ValueError, match="qkv_w"):
        llama_layer_decode_loop(x, bad, cache, cos, sin, cfg, backend_preset="reference")
