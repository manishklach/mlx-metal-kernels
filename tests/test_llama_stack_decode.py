import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from models import LlamaLikeConfig


def _load_stack_ops():
    root = Path(__file__).resolve().parents[1]
    path = root / "ops" / "llama_stack_ops.py"
    spec = importlib.util.spec_from_file_location("llama_stack_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stack_ops = _load_stack_ops()


def _mha_config():
    return LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=16,
        vocab_size=48,
        model_type="stack_mha_test",
    ).validate()


def _gqa_config():
    return LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=16,
        vocab_size=64,
        model_type="stack_gqa_test",
    ).validate()


def test_llama_stack_decode_matches_reference_for_mha():
    cfg = _mha_config()
    weights = stack_ops.create_random_quantized_llama_stack_weights(cfg, vocab_size=48, bits=4, seed=1)
    cache_ref = stack_ops.init_llama_stack_cache(cfg, 1, cfg.max_position_embeddings)
    cache_opt = stack_ops.init_llama_stack_cache(cfg, 1, cfg.max_position_embeddings)
    cos, sin = stack_ops._build_rope_tables_numpy(cfg, cfg.max_position_embeddings + 1)
    inputs = np.random.default_rng(2).normal(size=(1, 4, cfg.hidden_size)).astype(np.float32)
    ref, _ = stack_ops.reference_llama_stack_decode_loop(inputs, weights, cache_ref, cos, sin, cfg, return_logits=True)
    got, _ = stack_ops.llama_stack_decode_loop(inputs, weights, cache_opt, cos, sin, cfg, backend_preset="fused_experimental", return_logits=True)
    assert ref.shape == got.shape
    assert np.allclose(got, ref, atol=2e-1, rtol=2e-1)


def test_llama_stack_decode_matches_reference_for_gqa():
    cfg = _gqa_config()
    weights = stack_ops.create_random_quantized_llama_stack_weights(cfg, vocab_size=64, bits=4, seed=3)
    cache_ref = stack_ops.init_llama_stack_cache(cfg, 1, cfg.max_position_embeddings)
    cache_opt = stack_ops.init_llama_stack_cache(cfg, 1, cfg.max_position_embeddings)
    cos, sin = stack_ops._build_rope_tables_numpy(cfg, cfg.max_position_embeddings + 1)
    inputs = np.random.default_rng(4).normal(size=(1, 4, cfg.hidden_size)).astype(np.float32)
    ref, _ = stack_ops.reference_llama_stack_decode_loop(inputs, weights, cache_ref, cos, sin, cfg, return_logits=True)
    got, _ = stack_ops.llama_stack_decode_loop(inputs, weights, cache_opt, cos, sin, cfg, backend_preset="fused_experimental", return_logits=True)
    assert np.allclose(got, ref, atol=2e-1, rtol=2e-1)


def test_llama_stack_decode_three_layers_output_shape():
    cfg = LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=3,
        max_position_embeddings=8,
        vocab_size=32,
        model_type="stack_3layer_test",
    ).validate()
    weights = stack_ops.create_random_quantized_llama_stack_weights(cfg, vocab_size=32, bits=4, seed=5)
    cache = stack_ops.init_llama_stack_cache(cfg, 1, cfg.max_position_embeddings)
    cos, sin = stack_ops._build_rope_tables_numpy(cfg, cfg.max_position_embeddings + 1)
    inputs = np.random.default_rng(6).normal(size=(1, 3, cfg.hidden_size)).astype(np.float32)
    outputs, _ = stack_ops.llama_stack_decode_loop(inputs, weights, cache, cos, sin, cfg, backend_preset="fused_experimental", return_logits=True)
    assert outputs.shape == (1, 3, 32)


def test_llama_stack_decode_with_lm_head_shape():
    cfg = _mha_config()
    weights = stack_ops.create_random_quantized_llama_stack_weights(cfg, vocab_size=48, bits=4, seed=7)
    cache = stack_ops.init_llama_stack_cache(cfg, 1, cfg.max_position_embeddings)
    cos, sin = stack_ops._build_rope_tables_numpy(cfg, cfg.max_position_embeddings + 1)
    inputs = np.random.default_rng(8).normal(size=(1, 2, cfg.hidden_size)).astype(np.float32)
    outputs, _ = stack_ops.llama_stack_decode_loop(inputs, weights, cache, cos, sin, cfg, backend_preset="reference", return_logits=True)
    assert outputs.shape == (1, 2, 48)


def test_invalid_stack_layer_count_raises():
    cfg = _mha_config()
    weights = stack_ops.create_random_quantized_llama_stack_weights(cfg, vocab_size=48, bits=4, seed=9)
    bad = stack_ops.LlamaStackWeights(
        layers=weights.layers[:1],
        final_norm_weight=weights.final_norm_weight,
        lm_head=weights.lm_head,
        embedding=weights.embedding,
    )
    with pytest.raises(ValueError, match="layers must have length"):
        bad.validate(cfg)
