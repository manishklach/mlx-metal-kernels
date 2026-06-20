from __future__ import annotations

import numpy as np

from models.generation import _optional_llama_prefill_ops, _optional_llama_stack_ops
from models.llama_config import LlamaLikeConfig


def _prefill_module():
    ops = _optional_llama_prefill_ops()
    assert ops is not None
    return ops["module"]


def _stack_ops():
    ops = _optional_llama_stack_ops()
    assert ops is not None
    return ops


def test_llama_stack_prefill_matches_reference_gqa_with_logits():
    config = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=16,
        vocab_size=64,
    ).validate()
    stack_ops = _stack_ops()
    prefill_module = _prefill_module()
    weights = stack_ops["create_random_quantized_llama_stack_weights"](
        config,
        vocab_size=64,
        bits=4,
        group_size=32,
        dtype=None,
        seed=13,
        include_embedding=True,
        include_lm_head=True,
    )
    cache = stack_ops["init_llama_stack_cache"](config, 1, 16, cache_layout="contiguous", dtype=None)
    x = np.random.default_rng(19).normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    cos, sin = prefill_module._build_rope_tables_numpy(config, 16)
    ref_logits, ref_hidden, ref_cache = prefill_module.reference_llama_stack_prefill(x, weights, cache, cos, sin, config)
    opt_logits, opt_hidden, opt_cache = prefill_module.llama_stack_prefill(
        x,
        weights,
        cache,
        cos,
        sin,
        config,
        backend_config=prefill_module.fused_experimental_prefill_backend_config(),
    )
    assert ref_logits.shape == (1, 4, 64)
    assert ref_hidden.shape == (1, 4, 64)
    assert np.allclose(ref_logits, opt_logits, atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_hidden, opt_hidden, atol=1.5e-1, rtol=1.5e-1)
    assert len(opt_cache.layer_caches) == 2
    assert opt_cache.layer_caches[0][0].shape == (1, 16, 2, 16)


def test_llama_stack_prefill_without_lm_head_returns_hidden():
    config = LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=16,
        vocab_size=32,
    ).validate()
    stack_ops = _stack_ops()
    prefill_module = _prefill_module()
    weights = stack_ops["create_random_quantized_llama_stack_weights"](
        config,
        vocab_size=32,
        bits=4,
        group_size=32,
        dtype=None,
        seed=5,
        include_embedding=True,
        include_lm_head=False,
    )
    cache = stack_ops["init_llama_stack_cache"](config, 1, 16, cache_layout="contiguous", dtype=None)
    x = np.random.default_rng(23).normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    cos, sin = prefill_module._build_rope_tables_numpy(config, 16)
    hidden, updated_cache = prefill_module.llama_stack_prefill(
        x,
        weights,
        cache,
        cos,
        sin,
        config,
        backend_config=prefill_module.reference_prefill_backend_config(),
        return_logits=False,
    )
    assert hidden.shape == (1, 4, 32)
    assert len(updated_cache.layer_caches) == 2
