import pytest

mx = pytest.importorskip("mlx.core")

from ops.mlp_block_ops import quantized_mlp_block, reference_quantized_mlp_block
from ops.quant_ops import pack_q4


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 1.2e-1, 1.2e-1
    return 1e-1, 1e-1


def _groups(k, group_size):
    return (k + group_size - 1) // group_size


def _make_quantized_weights(bits, out_dim, in_dim, group_size):
    groups = _groups(in_dim, group_size)
    if bits == 4:
        scales = mx.random.uniform((out_dim, groups), low=0.01, high=0.1).astype(mx.float32)
        q = (mx.random.uniform((out_dim, in_dim)) * 16).astype(mx.uint8)
        return pack_q4(q), scales
    scales = mx.random.uniform((out_dim, groups), low=0.001, high=0.005).astype(mx.float32)
    q = (mx.random.uniform((out_dim, in_dim)) * 255).astype(mx.uint8)
    return q, scales


@pytest.mark.parametrize(
    ("bits", "B", "S", "hidden", "intermediate", "dtype"),
    [
        (4, 1, 1, 64, 128, mx.float16),
        (8, 1, 1, 64, 128, mx.float16),
        (4, 2, 4, 64, 128, mx.float16),
        (4, 1, 1, 64, 130, mx.float16),
        pytest.param(4, 1, 1, 64, 128, mx.bfloat16, marks=pytest.mark.skipif(not hasattr(mx, "bfloat16"), reason="bf16 unavailable")),
    ],
)
def test_quantized_mlp_fused_backend_matches_reference(bits, B, S, hidden, intermediate, dtype):
    mx.random.seed(701)
    group_size = 32
    x = mx.random.normal((B, S, hidden)).astype(dtype)
    residual = mx.random.normal((B, S, hidden)).astype(dtype)
    norm_weight = mx.random.normal((hidden,)).astype(dtype)
    gate_w, gate_scales = _make_quantized_weights(bits, intermediate, hidden, group_size)
    up_w, up_scales = _make_quantized_weights(bits, intermediate, hidden, group_size)
    down_w, down_scales = _make_quantized_weights(bits, hidden, intermediate, group_size)

    got = quantized_mlp_block(
        x,
        residual,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        bits=bits,
        group_size=group_size,
        backend_preset="fused_experimental",
    )
    ref = reference_quantized_mlp_block(
        x,
        residual,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        bits=bits,
        group_size=group_size,
    )
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert got.shape == x.shape
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_quantized_mlp_fused_backend_return_intermediates():
    mx.random.seed(702)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    residual = mx.random.normal((1, 1, 64)).astype(mx.float16)
    norm_weight = mx.random.normal((64,)).astype(mx.float16)
    gate_w, gate_scales = _make_quantized_weights(4, 128, 64, 32)
    up_w, up_scales = _make_quantized_weights(4, 128, 64, 32)
    down_w, down_scales = _make_quantized_weights(4, 64, 128, 32)
    out, intermediates = quantized_mlp_block(
        x,
        residual,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        bits=4,
        group_size=32,
        backend_preset="fused_experimental",
        return_intermediates=True,
    )
    mx.eval(out, *intermediates.values())
    assert out.shape == x.shape
    assert set(intermediates) == {"z", "normed", "gate", "up", "mlp", "down"}


def test_quantized_mlp_fused_backend_unsupported_bits_raise():
    mx.random.seed(703)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    residual = mx.random.normal((1, 1, 64)).astype(mx.float16)
    norm_weight = mx.random.normal((64,)).astype(mx.float16)
    gate_w, gate_scales = _make_quantized_weights(4, 128, 64, 32)
    up_w, up_scales = _make_quantized_weights(4, 128, 64, 32)
    down_w, down_scales = _make_quantized_weights(4, 64, 128, 32)
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        quantized_mlp_block(
            x,
            residual,
            norm_weight,
            gate_w,
            gate_scales,
            up_w,
            up_scales,
            down_w,
            down_scales,
            bits=3,
            group_size=32,
            backend_preset="fused_experimental",
        )
