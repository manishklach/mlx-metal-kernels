from __future__ import annotations

import numpy as np
import pytest

from models.llama_config import LlamaLikeConfig


def _get_cache_ops():
    try:
        from ops.kv_cache_reuse_ops import cache_prefix_equal, clone_stack_cache
        from ops.speculative_cache_ops import commit_accepted_cache, discard_suffix
        return {
            "cache_prefix_equal": cache_prefix_equal,
            "clone_stack_cache": clone_stack_cache,
            "commit_accepted_cache": commit_accepted_cache,
            "discard_suffix": discard_suffix,
        }
    except ImportError:
        pytest.skip("Speculative cache ops require mlx (not available in this environment)")


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


def _fill_cache(cache, fill_value: float = 1.0):
    for i, (K, V) in enumerate(cache.layer_caches):
        K[:] = (i + 1) * fill_value
        V[:] = -(i + 1) * fill_value


class TestCommitAcceptedCache:
    def test_commit_zero_accepted(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=32,
        ).validate()
        draft = _make_cache(config, B=1, max_seq_len=32)
        committed = _make_cache(config, B=1, max_seq_len=32)
        _fill_cache(draft, 2.0)
        _fill_cache(committed, 1.0)
        result = ops["commit_accepted_cache"](draft, committed, accepted_count=0)
        assert ops["cache_prefix_equal"](result, committed)

    def test_commit_all_accepted(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=32,
        ).validate()
        draft = _make_cache(config, B=1, max_seq_len=32)
        committed = _make_cache(config, B=1, max_seq_len=32)
        _fill_cache(draft, 2.0)
        _fill_cache(committed, 1.0)
        result = ops["commit_accepted_cache"](draft, committed, accepted_count=16)
        assert ops["cache_prefix_equal"](result, draft, length=16)

    def test_commit_partial(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=32,
        ).validate()
        draft = _make_cache(config, B=1, max_seq_len=32)
        committed = _make_cache(config, B=1, max_seq_len=32)
        _fill_cache(draft, 2.0)
        _fill_cache(committed, 1.0)
        result = ops["commit_accepted_cache"](draft, committed, accepted_count=4)
        assert ops["cache_prefix_equal"](result, draft, length=4)
        assert ops["cache_prefix_equal"](result, committed, length=16)

    def test_layout_mismatch_raises(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=32,
        ).validate()
        draft = _make_cache(config)
        committed = _make_cache(config)
        draft.cache_layout = "paged"
        with pytest.raises(ValueError, match="cache_layout"):
            ops["commit_accepted_cache"](draft, committed, accepted_count=0)

    def test_layer_count_mismatch_raises(self):
        ops = _get_cache_ops()
        config_a = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=32,
        ).validate()
        config_b = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=3,
            max_position_embeddings=32,
        ).validate()
        draft = _make_cache(config_a)
        committed = _make_cache(config_b)
        with pytest.raises(ValueError, match="layer count"):
            ops["commit_accepted_cache"](draft, committed, accepted_count=0)

    def test_accepted_count_exceeds_width_raises(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=8,
        ).validate()
        draft = _make_cache(config, max_seq_len=8)
        committed = _make_cache(config, max_seq_len=8)
        with pytest.raises(ValueError, match="accepted_count"):
            ops["commit_accepted_cache"](draft, committed, accepted_count=16)

    def test_does_not_mutate_inputs(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=32,
        ).validate()
        draft = _make_cache(config)
        committed = _make_cache(config)
        _fill_cache(draft, 2.0)
        _fill_cache(committed, 1.0)
        draft_copy = ops["clone_stack_cache"](draft)
        committed_copy = ops["clone_stack_cache"](committed)
        result = ops["commit_accepted_cache"](draft, committed, accepted_count=3)
        assert ops["cache_prefix_equal"](draft, draft_copy)
        assert ops["cache_prefix_equal"](committed, committed_copy)


class TestDiscardSuffix:
    def test_discard_first_half(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=16,
        ).validate()
        cache = _make_cache(config, max_seq_len=16)
        _fill_cache(cache, 1.0)
        sliced = ops["discard_suffix"](cache, 8)
        assert sliced.layer_caches[0][0].shape[1] == 8

    def test_discard_all(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=16,
        ).validate()
        cache = _make_cache(config, max_seq_len=16)
        sliced = ops["discard_suffix"](cache, 0)
        assert sliced.layer_caches[0][0].shape[1] == 0

    def test_does_not_mutate_input(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=16,
        ).validate()
        cache = _make_cache(config, max_seq_len=16)
        _fill_cache(cache, 1.0)
        original_shape = cache.layer_caches[0][0].shape[1]
        ops["discard_suffix"](cache, 4)
        assert cache.layer_caches[0][0].shape[1] == original_shape

    def test_paged_raises(self):
        ops = _get_cache_ops()
        config = LlamaLikeConfig(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=2,
            max_position_embeddings=16,
        ).validate()
        try:
            from ops.llama_stack_ops import init_llama_stack_cache
            paged = init_llama_stack_cache(config, 1, 16, cache_layout="paged", dtype=np.float32)
        except ImportError:
            pytest.skip("llama_stack_ops require mlx")
        with pytest.raises(NotImplementedError):
            ops["discard_suffix"](paged, 4)
