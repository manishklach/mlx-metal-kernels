import mlx.core as mx
import pytest

from models.llama_config import LlamaLikeConfig, tiny_debug_config
from models.weight_layouts import fused_qkv_spec, llama_layer_weight_specs, validate_weight_shapes


def test_llama_layer_weight_specs_dims():
    cfg = tiny_debug_config()
    specs = llama_layer_weight_specs(cfg)
    first = specs[0]
    assert first.q_proj.out_dim == cfg.hidden_size
    assert first.k_proj.out_dim == cfg.hidden_size
    assert first.down_proj.in_dim == cfg.intermediate_size


def test_fused_qkv_spec_shape_for_mha():
    cfg = tiny_debug_config()
    spec = fused_qkv_spec(cfg)
    assert spec.expected_shape() == (3 * cfg.hidden_size, cfg.hidden_size)


def test_fused_qkv_spec_shape_for_gqa():
    cfg = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=64,
    ).validate()
    spec = fused_qkv_spec(cfg)
    assert spec.expected_shape() == (cfg.fused_qkv_output_dim(), cfg.hidden_size)
    assert spec.expected_shape()[0] < 3 * cfg.hidden_size


def test_quantized_specs_include_bits_and_group_size():
    cfg = tiny_debug_config()
    spec = fused_qkv_spec(cfg, quantized=True, bits=4, group_size=32)
    assert spec.quantized is True
    assert spec.bits == 4
    assert spec.group_size == 32


def test_validate_weight_shapes_catches_wrong_shape():
    cfg = tiny_debug_config()
    specs = llama_layer_weight_specs(cfg)
    first = specs[0]
    weights = {
        first.q_proj.name: mx.zeros((first.q_proj.out_dim, first.q_proj.in_dim), dtype=mx.float16),
        first.k_proj.name: mx.zeros((first.k_proj.out_dim, first.k_proj.in_dim), dtype=mx.float16),
        first.v_proj.name: mx.zeros((first.v_proj.out_dim, first.v_proj.in_dim), dtype=mx.float16),
        first.o_proj.name: mx.zeros((first.o_proj.out_dim, first.o_proj.in_dim), dtype=mx.float16),
        first.gate_proj.name: mx.zeros((first.gate_proj.out_dim, first.gate_proj.in_dim), dtype=mx.float16),
        first.up_proj.name: mx.zeros((first.up_proj.out_dim, first.up_proj.in_dim), dtype=mx.float16),
        first.down_proj.name: mx.zeros((first.down_proj.out_dim + 1, first.down_proj.in_dim), dtype=mx.float16),
        first.input_layernorm[0]: mx.zeros((first.input_layernorm[1],), dtype=mx.float16),
        first.post_attention_layernorm[0]: mx.zeros((first.post_attention_layernorm[1],), dtype=mx.float16),
    }
    with pytest.raises(ValueError, match="down_proj"):
        validate_weight_shapes(first, weights)


def test_gqa_config_has_smaller_kv_dims():
    cfg = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=64,
    ).validate()
    specs = llama_layer_weight_specs(cfg)
    assert specs[0].k_proj.out_dim == cfg.kv_output_dim()
    assert specs[0].k_proj.out_dim < specs[0].q_proj.out_dim
