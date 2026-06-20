import mlx.core as mx
import pytest

from ops.mlp_block_ops import quantized_linear
from ops.quant_ops import pack_q4


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 6e-2, 6e-2
    return 4e-2, 4e-2


def _groups(k, group_size):
    return (k + group_size - 1) // group_size


def _make_quantized_weights(bits, out_dim, in_dim, group_size, *, with_zeros=False):
    groups = _groups(in_dim, group_size)
    scales = mx.random.normal((out_dim, groups)).astype(mx.float32)
    zeros = mx.random.normal((out_dim, groups)).astype(mx.float32) if with_zeros else None
    if bits == 4:
        q = (mx.random.uniform((out_dim, in_dim)) * 16).astype(mx.uint8)
        return pack_q4(q), scales, zeros
    q = (mx.random.uniform((out_dim, in_dim)) * 255).astype(mx.uint8)
    return q, scales, zeros


@pytest.mark.parametrize(("dtype", "shape"), [(mx.float16, (1, 1, 64)), (mx.bfloat16, (1, 1, 64))])
def test_quantized_linear_q4_matches_reference(dtype, shape):
    mx.random.seed(221)
    B, S, K = shape
    out_dim = 128
    x = mx.random.normal((B, S, K)).astype(dtype)
    w, scales, _ = _make_quantized_weights(4, out_dim, K, 32)
    got = quantized_linear(x, w, scales, bits=4, group_size=32, backend="metal_tiled")
    ref = quantized_linear(x, w, scales, bits=4, group_size=32, backend="reference")
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert got.shape == (B, S, out_dim)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_quantized_linear_q8_matches_reference():
    mx.random.seed(222)
    B, S, K, out_dim = 2, 4, 64, 128
    x = mx.random.normal((B, S, K)).astype(mx.float16)
    w, scales, _ = _make_quantized_weights(8, out_dim, K, 32)
    got = quantized_linear(x, w, scales, bits=8, group_size=32, backend="metal_tiled")
    ref = quantized_linear(x, w, scales, bits=8, group_size=32, backend="reference")
    mx.eval(got, ref)
    atol, rtol = _tol(mx.float16)
    assert got.shape == (B, S, out_dim)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_quantized_linear_q4_with_zeros_matches_reference():
    mx.random.seed(223)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    w, scales, zeros = _make_quantized_weights(4, 128, 64, 32, with_zeros=True)
    got = quantized_linear(x, w, scales, zeros, bits=4, group_size=32, backend="metal_tiled")
    ref = quantized_linear(x, w, scales, zeros, bits=4, group_size=32, backend="reference")
    mx.eval(got, ref)
    atol, rtol = _tol(mx.float16)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_quantized_linear_invalid_bits_raises():
    x = mx.zeros((1, 1, 64), dtype=mx.float16)
    w = mx.zeros((128, 32), dtype=mx.uint8)
    scales = mx.zeros((128, 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        quantized_linear(x, w, scales, bits=3)


def test_quantized_linear_wrong_scale_shape_raises():
    x = mx.zeros((1, 1, 64), dtype=mx.float16)
    w, _, _ = _make_quantized_weights(4, 128, 64, 32)
    bad_scales = mx.zeros((128, 3), dtype=mx.float32)
    with pytest.raises(ValueError, match="scales must have shape"):
        quantized_linear(x, w, bad_scales, bits=4, group_size=32, backend="reference")
