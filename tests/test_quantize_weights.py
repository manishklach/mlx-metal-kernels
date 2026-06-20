import pytest

from models.quantize_weights import (
    QuantizationConfig,
    QuantizedWeight,
    dequantize_quantized_weight,
    quantization_error,
    quantize_weight_groupwise,
)

np = pytest.importorskip("numpy")


def _weight(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def test_q4_symmetric_quantization_shapes_and_error():
    weight = _weight((16, 64), seed=10)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=4, group_size=32))
    dequantized = dequantize_quantized_weight(quantized.packed_weight, quantized.scales, quantized.zeros, bits=4, group_size=32)
    metrics = quantization_error(weight, dequantized)
    assert quantized.packed_weight.shape == (16, 32)
    assert quantized.scales.shape == (16, 2)
    assert dequantized.shape == (16, 64)
    assert metrics["rmse"] < 0.25
    assert metrics["relative_rmse"] < 0.25


def test_q4_odd_input_dim_shapes():
    weight = _weight((16, 65), seed=11)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=4, group_size=32))
    assert quantized.packed_weight.shape == (16, 33)
    assert quantized.scales.shape == (16, 3)


def test_q8_symmetric_quantization_shapes():
    weight = _weight((16, 64), seed=12)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=8, group_size=32))
    dequantized = dequantize_quantized_weight(quantized.packed_weight, quantized.scales, bits=8, group_size=32)
    assert quantized.packed_weight.shape == (16, 64)
    assert quantized.scales.shape == (16, 2)
    assert dequantized.shape == (16, 64)


def test_zeros_absent_for_default_symmetric_path():
    weight = _weight((8, 32), seed=13)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=4, group_size=32))
    assert quantized.zeros is None


def test_invalid_bits_raises():
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        QuantizationConfig(bits=3).validate()


def test_invalid_group_size_raises():
    with pytest.raises(ValueError, match="group_size must be positive"):
        QuantizationConfig(group_size=0).validate()


def test_quantized_weight_validate_works():
    weight = _weight((8, 64), seed=14)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=4, group_size=32))
    validated = QuantizedWeight(
        name="foo",
        bits=quantized.bits,
        group_size=quantized.group_size,
        packed_weight=quantized.packed_weight,
        scales=quantized.scales,
        zeros=quantized.zeros,
        original_shape=quantized.original_shape,
        packed_shape=quantized.packed_shape,
        scale_shape=quantized.scale_shape,
        symmetric=True,
    ).validate()
    assert validated.original_shape == (8, 64)


def test_quantization_error_keys():
    weight = _weight((4, 8), seed=15)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=8, group_size=4))
    metrics = quantization_error(weight, dequantize_quantized_weight(quantized.packed_weight, quantized.scales, bits=8, group_size=4))
    assert set(metrics) == {"max_abs_error", "mean_abs_error", "rmse", "relative_rmse"}
