from __future__ import annotations

import numpy as np

from models.llama_config import LlamaLikeConfig, tiny_debug_config
from models.prefix_cache import (
    InMemoryPrefixCache,
    PrefixCacheEntry,
    PrefixCacheMatch,
    compute_fingerprint,
    prefill_with_prefix_reuse,
)

import pytest


def _init_stack_cache(config, B, max_seq_len, cache_layout="contiguous", dtype=np.float32):
    try:
        from ops.llama_stack_ops import init_llama_stack_cache
    except ImportError:
        pytest.skip("llama_stack_ops require mlx (not available in this environment)")
    return init_llama_stack_cache(config, B, max_seq_len, cache_layout=cache_layout, dtype=dtype)


def test_compute_fingerprint_consistent():
    config = tiny_debug_config()
    fp1 = compute_fingerprint(config)
    fp2 = compute_fingerprint(config)
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 16


def test_compute_fingerprint_differs():
    config_a = tiny_debug_config()
    config_b = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=64,
    ).validate()
    fp_a = compute_fingerprint(config_a)
    fp_b = compute_fingerprint(config_b)
    assert fp_a != fp_b


def test_in_memory_cache_store_and_lookup():
    cache = InMemoryPrefixCache(max_size=16)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    entry = PrefixCacheEntry(fingerprint=fp, token_ids=[1, 2, 3], stack_cache=stack_cache)
    cache.store(entry)
    assert cache.size == 1
    match = cache.lookup([1, 2, 3, 4], fp)
    assert match.matched
    assert match.matched_length == 3
    assert match.suffix_token_ids == [4]


def test_in_memory_cache_longest_prefix():
    cache = InMemoryPrefixCache(max_size=16)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    cache.store(PrefixCacheEntry(fingerprint=fp, token_ids=[1, 2], stack_cache=stack_cache))
    cache.store(PrefixCacheEntry(fingerprint=fp, token_ids=[1, 2, 3, 5], stack_cache=stack_cache))
    match = cache.lookup([1, 2, 3, 4], fp)
    assert match.matched
    assert match.matched_length == 3


def test_in_memory_cache_no_match():
    cache = InMemoryPrefixCache(max_size=16)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    cache.store(PrefixCacheEntry(fingerprint=fp, token_ids=[1, 2, 3], stack_cache=stack_cache))
    match = cache.lookup([4, 5, 6], fp)
    assert not match.matched
    assert match.matched_length == 0


def test_in_memory_cache_different_fingerprint():
    cache = InMemoryPrefixCache(max_size=16)
    config_a = tiny_debug_config()
    config_b = LlamaLikeConfig(
        hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=64,
    ).validate()
    fp_a = compute_fingerprint(config_a)
    fp_b = compute_fingerprint(config_b)
    stack_cache = _init_stack_cache(config_a, 1, 16)
    cache.store(PrefixCacheEntry(fingerprint=fp_a, token_ids=[1, 2, 3], stack_cache=stack_cache))
    match = cache.lookup([1, 2, 3], fp_b)
    assert not match.matched


def test_in_memory_cache_eviction():
    cache = InMemoryPrefixCache(max_size=3)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    for i in range(4):
        entry = PrefixCacheEntry(fingerprint=fp, token_ids=[i], stack_cache=stack_cache)
        cache.store(entry)
    assert cache.size == 3


def test_in_memory_cache_clear():
    cache = InMemoryPrefixCache(max_size=16)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    for i in range(5):
        cache.store(PrefixCacheEntry(fingerprint=fp, token_ids=[i], stack_cache=stack_cache))
    assert cache.size == 5
    cache.clear()
    assert cache.size == 0


def test_in_memory_cache_stats():
    cache = InMemoryPrefixCache(max_size=64)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    cache.store(PrefixCacheEntry(fingerprint=fp, token_ids=[1, 2, 3], stack_cache=stack_cache))
    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["max_size"] == 64
    assert fp in stats["fingerprints"]


def test_in_memory_cache_empty_lookup():
    cache = InMemoryPrefixCache(max_size=16)
    match = cache.lookup([1, 2, 3], "test_fp")
    assert not match.matched


def test_in_memory_cache_empty_token_ids():
    cache = InMemoryPrefixCache(max_size=16)
    config = tiny_debug_config()
    fp = compute_fingerprint(config)
    stack_cache = _init_stack_cache(config, 1, 16)
    cache.store(PrefixCacheEntry(fingerprint=fp, token_ids=[1, 2], stack_cache=stack_cache))
    match = cache.lookup([], fp)
    assert not match.matched


def test_in_memory_cache_invalid_max_size():
    with pytest.raises(ValueError, match="max_size"):
        InMemoryPrefixCache(max_size=0)


def test_prefix_cache_match_dataclass():
    match = PrefixCacheMatch(matched=True, matched_length=3, suffix_token_ids=[4, 5], entry=None)
    assert match.matched
    assert match.matched_length == 3
    assert match.suffix_token_ids == [4, 5]
