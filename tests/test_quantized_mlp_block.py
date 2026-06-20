import mlx.core as mx
import pytest

from ops.mlp_block_ops import quantized_mlp_block, quantized_mlp_decode_step, reference_quantized_mlp_block
from ops.quant_ops import pack_q4


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 1e-1, 1e-1
    return 8e-2, 8e-2


def _groups(k, group_size):
    return (k + group_size - 1) // group_size


def _make_quantized_weights(bits, out_dim, in_dim, group_size):
    groups = _groups(in_dim, group_size)
    scales = mx.random.normal((out_dim, groups)).astype(mx.float32)
    if bits == 4:
        q = (mx.random.uniform((out_dim, in_dim)) * 16).astype(mx.uint8)
        return pack_q4(q), scales
    q = (mx.random.uniform((out_dim, in_dim)) * 255).astype(mx.uint8)
    return q, scales


@pytest.mark.parametrize(
    ("bits", "B", "S", "hidden", "intermediate", "dtype"),
    [
        (4, 1, 1, 64, 128, mx.float16),
        (8, 1, 1, 64, 128, mx.float16),
        (4, 2, 4, 64, 128, mx.float16),
        (4, 1, 1, 64, 128, mx.bfloat16),
    ],
)
def test_quantized_mlp_block_matches_reference(bits, B, S, hidden, intermediate, dtype):
    mx.random.seed(231)
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
        norm_backend="metal",
        matvec_backend="metal_tiled",
        activation_backend="metal",
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


def test_quantized_mlp_block_return_intermediates():
    mx.random.seed(232)
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
        return_intermediates=True,
    )
    mx.eval(out, *intermediates.values())
    assert out.shape == x.shape
    assert set(intermediates) == {"z", "normed", "gate", "up", "mlp", "down"}


@pytest.mark.parametrize("backend_preset", ["reference", "metal", "parallel", "tiled", "fused_experimental"])
def test_quantized_mlp_decode_step_presets_work(backend_preset):
    mx.random.seed(233)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    norm_weight = mx.random.normal((64,)).astype(mx.float16)
    gate_w, gate_scales = _make_quantized_weights(4, 128, 64, 32)
    up_w, up_scales = _make_quantized_weights(4, 128, 64, 32)
    down_w, down_scales = _make_quantized_weights(4, 64, 128, 32)
    out = quantized_mlp_decode_step(
        x,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        bits=4,
        group_size=32,
        backend_preset=backend_preset,
    )
    mx.eval(out)
    assert out.shape == x.shape
