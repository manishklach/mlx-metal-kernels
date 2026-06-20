import pytest

from models.llama_config import tiny_debug_config, tiny_gqa_debug_config
from models.quant_packaging import (
    llama_quantized_layer_specs,
    q4_packed_shape,
    q8_packed_shape,
    quantized_linear_spec,
)


def test_q4_packed_shape():
    assert q4_packed_shape((32, 64)) == (32, 32)
    assert q4_packed_shape((32, 65)) == (32, 33)


def test_q8_packed_shape():
    assert q8_packed_shape((32, 64)) == (32, 64)


def test_quantized_linear_spec_bits_4():
    spec = quantized_linear_spec("foo", (32, 64), bits=4, group_size=32)
    assert spec.packed_shape == (32, 32)
    assert spec.scales_shape == (32, 2)
    assert spec.zeros_shape is None


def test_quantized_linear_spec_bits_8_and_zeros():
    spec = quantized_linear_spec("foo", (32, 64), bits=8, group_size=32, with_zeros=True)
    assert spec.packed_shape == (32, 64)
    assert spec.zeros_shape == (32, 2)


def test_invalid_bits_and_group_size_raise():
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        quantized_linear_spec("foo", (32, 64), bits=3)
    with pytest.raises(ValueError, match="group_size must be positive"):
        quantized_linear_spec("foo", (32, 64), group_size=0)


def test_llama_quantized_layer_specs_keys_and_gqa_dims():
    specs = llama_quantized_layer_specs(tiny_debug_config(), layer_idx=0)
    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert key in specs
    gqa_specs = llama_quantized_layer_specs(tiny_gqa_debug_config(), layer_idx=0)
    assert gqa_specs["q_proj"].original_shape[0] > gqa_specs["k_proj"].original_shape[0]
