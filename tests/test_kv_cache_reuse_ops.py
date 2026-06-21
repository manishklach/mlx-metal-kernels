from __future__ import annotations

import numpy as np

from models.llama_config import LlamaLikeConfig

import pytest


def _get_cache_ops():
    try:
        from ops.kv_cache_reuse_ops import (
            cache_prefix_equal,
            clone_layer_cache,
            clone_stack_cache,
            copy_prefix_cache_into,
            slice_layer_cache,
            paged_cache_reuse_not_implemented,
        )
        return {
            "cache_prefix_equal": cache_prefix_equal,
            "clone_layer_cache": clone_layer_cache,
            "clone_stack_cache": clone_stack_cache,
            "copy_prefix_cache_into": copy_prefix_cache_into,
            "slice_layer_cache": slice_layer_cache,
            "paged_cache_reuse_not_implemented": paged_cache_reuse_not_implemented,
        }
    except ImportError:
        pytest.skip("Prefix KV-cache reuse ops require mlx (not available in this environment)")


def _make_cache(config=None, B=1, max_seq_len=16, dtype=np.float32):
    try:
        from ops.llama_stack_ops import init_llama_stack_cache
    except ImportError:
        pytest.skip("llama_stack_ops require mlx")
    if config is None:
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=max_seq_len,
        ).validate()
    return init_llama_stack_cache(config, B, max_seq_len, cache_layout="contiguous", dtype=dtype)


def test_clone_layer_cache():
    ops = _get_cache_ops()
    config = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=1,
        max_position_embeddings=16,
    ).validate()
    stack_cache = _make_cache(config, 1, 16)
    original = stack_cache.layer_caches[0]
    cloned = ops["clone_layer_cache"](original)
    K_orig, V_orig = original
    K_clone, V_clone = cloned
    assert K_clone.shape == K_orig.shape
    assert V_clone.shape == V_orig.shape
    assert np.array_equal(K_clone, K_orig)
    assert np.array_equal(V_clone, V_orig)
    K_clone[0, 0, 0, 0] = 99.0
    assert K_orig[0, 0, 0, 0] != 99.0


def test_clone_stack_cache():
    ops = _get_cache_ops()
    stack_cache = _make_cache()
    cloned = ops["clone_stack_cache"](stack_cache)
    assert len(cloned.layer_caches) == len(stack_cache.layer_caches)
    assert cloned.cache_layout == stack_cache.cache_layout
    assert cloned.max_seq_len == stack_cache.max_seq_len
    for orig_lc, clone_lc in zip(stack_cache.layer_caches, cloned.layer_caches):
        assert np.array_equal(orig_lc[0], clone_lc[0])
        assert np.array_equal(orig_lc[1], clone_lc[1])
    cloned.layer_caches[0][0][0, 0, 0, 0] = 99.0
    assert stack_cache.layer_caches[0][0][0, 0, 0, 0] != 99.0


def test_slice_layer_cache():
    ops = _get_cache_ops()
    config = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=1,
        max_position_embeddings=16,
    ).validate()
    stack_cache = _make_cache(config, B=1, max_seq_len=16)
    layer_cache = stack_cache.layer_caches[0]
    K, V = layer_cache
    K[0, :5] = 1.0
    V[0, :5] = 2.0
    sliced = ops["slice_layer_cache"](layer_cache, 3)
    K_s, V_s = sliced
    assert K_s.shape == (1, 3, 2, 16)
    assert V_s.shape == (1, 3, 2, 16)
    assert np.array_equal(K_s, K[:, :3])
    assert np.array_equal(V_s, V[:, :3])


def test_copy_prefix_cache_into():
    ops = _get_cache_ops()
    config = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=32,
    ).validate()
    src = _make_cache(config, B=1, max_seq_len=32)
    dst = _make_cache(config, B=1, max_seq_len=32)
    for lc in src.layer_caches:
        lc[0][:, :5] = 42.0
        lc[1][:, :5] = 7.0
    result = ops["copy_prefix_cache_into"](src, dst, length=5)
    for r_lc in result.layer_caches:
        assert np.all(r_lc[0][:, :5] == 42.0)
        assert np.all(r_lc[1][:, :5] == 7.0)
        assert np.all(r_lc[0][:, 5:] == 0.0)
        assert np.all(r_lc[1][:, 5:] == 0.0)


def test_copy_prefix_cache_into_mismatch():
    ops = _get_cache_ops()
    config_src = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=32,
    ).validate()
    config_dst = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=3,
        max_position_embeddings=32,
    ).validate()
    src = _make_cache(config_src, B=1, max_seq_len=32)
    dst = _make_cache(config_dst, B=1, max_seq_len=32)
    with pytest.raises(ValueError, match="layer count mismatch"):
        ops["copy_prefix_cache_into"](src, dst, length=5)


def test_cache_prefix_equal():
    ops = _get_cache_ops()
    stack_cache = _make_cache()
    cloned = ops["clone_stack_cache"](stack_cache)
    assert ops["cache_prefix_equal"](stack_cache, cloned)
    assert ops["cache_prefix_equal"](stack_cache, cloned, length=8)
    stack_cache.layer_caches[0][0][0, 0, 0, 0] = 1.0
    assert not ops["cache_prefix_equal"](stack_cache, cloned)
    assert not ops["cache_prefix_equal"](stack_cache, cloned, length=1)


def test_cache_prefix_equal_layers_mismatch():
    ops = _get_cache_ops()
    config_2 = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=16,
    ).validate()
    config_3 = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=3,
        max_position_embeddings=16,
    ).validate()
    a = _make_cache(config_2, B=1, max_seq_len=16)
    b = _make_cache(config_3, B=1, max_seq_len=16)
    assert not ops["cache_prefix_equal"](a, b)


def test_cache_prefix_equal_layout_mismatch():
    ops = _get_cache_ops()
    config = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=16,
    ).validate()
    a = _make_cache(config, B=1, max_seq_len=16)
    try:
        from ops.llama_stack_ops import LlamaStackCache
    except ImportError:
        pytest.skip("llama_stack_ops require mlx")
    b = LlamaStackCache(
        layer_caches=a.layer_caches,
        cache_layout="paged",
        max_seq_len=a.max_seq_len,
        page_size=16,
    )
    assert not ops["cache_prefix_equal"](a, b)


def test_cache_prefix_equal_too_short():
    ops = _get_cache_ops()
    config = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=8,
    ).validate()
    a = _make_cache(config, B=1, max_seq_len=8)
    b = ops["clone_stack_cache"](a)
    assert not ops["cache_prefix_equal"](a, b, length=16)


def test_paged_cache_reuse_not_implemented():
    ops = _get_cache_ops()
    with pytest.raises(NotImplementedError, match="contiguous"):
        ops["paged_cache_reuse_not_implemented"]()
