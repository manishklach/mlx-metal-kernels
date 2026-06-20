from __future__ import annotations

import numpy as np
import pytest

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


def _layer_case(config: LlamaLikeConfig, *, bits: int = 4):
    stack_ops = _stack_ops()
    prefill_module = _prefill_module()
    weights = stack_ops["create_random_quantized_llama_stack_weights"](
        config,
        vocab_size=64,
        bits=bits,
        group_size=32,
        dtype=None,
        seed=7,
        include_embedding=False,
        include_lm_head=False,
    ).layers[0]
    cache = (
        np.zeros((1, 8, config.num_key_value_heads, config.head_dim), dtype=np.float32),
        np.zeros((1, 8, config.num_key_value_heads, config.head_dim), dtype=np.float32),
    )
    x = np.random.default_rng(11).normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    cos, sin = prefill_module._build_rope_tables_numpy(config, 16)
    ref_out, ref_cache = prefill_module.reference_llama_layer_prefill(x, weights, cache, cos, sin, config)
    opt_out, opt_cache = prefill_module.llama_layer_prefill(
        x,
        weights,
        cache,
        cos,
        sin,
        config,
        backend_config=prefill_module.fused_experimental_prefill_backend_config(),
    )
    return ref_out, ref_cache, opt_out, opt_cache


def test_llama_layer_prefill_matches_reference_mha():
    config = LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=16,
        vocab_size=64,
    ).validate()
    ref_out, ref_cache, opt_out, opt_cache = _layer_case(config)
    assert np.allclose(ref_out, opt_out, atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_cache[0], opt_cache[0], atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_cache[1], opt_cache[1], atol=1.5e-1, rtol=1.5e-1)


def test_llama_layer_prefill_matches_reference_gqa():
    config = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=16,
        vocab_size=64,
    ).validate()
    ref_out, ref_cache, opt_out, opt_cache = _layer_case(config)
    assert np.allclose(ref_out, opt_out, atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_cache[0], opt_cache[0], atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_cache[1], opt_cache[1], atol=1.5e-1, rtol=1.5e-1)


def test_llama_layer_prefill_matches_reference_mqa():
    config = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=16,
        vocab_size=64,
    ).validate()
    ref_out, ref_cache, opt_out, opt_cache = _layer_case(config)
    assert np.allclose(ref_out, opt_out, atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_cache[0], opt_cache[0], atol=1.5e-1, rtol=1.5e-1)
    assert np.allclose(ref_cache[1], opt_cache[1], atol=1.5e-1, rtol=1.5e-1)


def test_llama_layer_prefill_continuation_not_implemented():
    config = LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
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
        seed=9,
        include_embedding=False,
        include_lm_head=False,
    ).layers[0]
    cache = (
        np.zeros((1, 8, config.num_key_value_heads, config.head_dim), dtype=np.float32),
        np.zeros((1, 8, config.num_key_value_heads, config.head_dim), dtype=np.float32),
    )
    x = np.random.default_rng(3).normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    cos, sin = prefill_module._build_rope_tables_numpy(config, 16)
    with pytest.raises(NotImplementedError, match="Continuation prefill"):
        prefill_module.reference_llama_layer_prefill(x, weights, cache, cos, sin, config, start_position=1)
