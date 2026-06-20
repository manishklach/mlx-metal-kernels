import pytest

mx = pytest.importorskip("mlx.core")

from ops.mlp_block_ops import quantized_gate_up_projection
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


@pytest.mark.parametrize(
    ("bits", "shape", "out_dim", "with_zeros", "dtype"),
    [
        (4, (1, 1, 64), 128, False, "float16"),
        (4, (2, 4, 64), 128, False, "float16"),
        (4, (1, 3, 96), 50, True, "float16"),
        (8, (1, 1, 64), 128, False, "float16"),
        (8, (2, 4, 64), 128, False, "float16"),
        pytest.param(4, (1, 1, 64), 128, False, "bfloat16", marks=pytest.mark.skipif(not hasattr(mx, "bfloat16"), reason="bf16 unavailable")),
    ],
)
def test_quantized_gate_up_projection_matches_reference(bits, shape, out_dim, with_zeros, dtype):
    mx.random.seed(501)
    dtype_obj = mx.float16 if dtype == "float16" else mx.bfloat16
    group_size = 32
    x = mx.random.normal(shape).astype(dtype_obj)
    gate_w, gate_scales, gate_zeros = _make_quantized_weights(bits, out_dim, shape[-1], group_size, with_zeros=with_zeros)
    up_w, up_scales, up_zeros = _make_quantized_weights(bits, out_dim, shape[-1], group_size, with_zeros=with_zeros)

    got_gate, got_up = quantized_gate_up_projection(
        x,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        bits=bits,
        group_size=group_size,
        backend="metal_gate_up_tiled",
    )
    ref_gate, ref_up = quantized_gate_up_projection(
        x,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        gate_zeros=gate_zeros,
        up_zeros=up_zeros,
        bits=bits,
        group_size=group_size,
        backend="reference",
    )
    mx.eval(got_gate, got_up, ref_gate, ref_up)
    atol, rtol = _tol(dtype_obj)
    assert got_gate.shape == ref_gate.shape
    assert got_up.shape == ref_up.shape
    assert mx.allclose(got_gate, ref_gate, atol=atol, rtol=rtol).item()
    assert mx.allclose(got_up, ref_up, atol=atol, rtol=rtol).item()


def test_quantized_gate_up_projection_requires_explicit_backend():
    mx.random.seed(502)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    gate_w, gate_scales, _ = _make_quantized_weights(4, 128, 64, 32)
    up_w, up_scales, _ = _make_quantized_weights(4, 128, 64, 32)
    with pytest.raises(ValueError, match="backend must be one of"):
        quantized_gate_up_projection(
            x,
            gate_w,
            gate_scales,
            up_w,
            up_scales,
            bits=4,
            group_size=32,
            backend="metal_gate_up_auto_fallback",
        )
