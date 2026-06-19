import mlx.core as mx
import pytest

from ops.quant_ops import (
    pack_q4,
    q4_matvec_decode,
    q8_matvec_decode,
    reference_q4_matvec_decode,
    reference_q8_matvec_decode,
)


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 5e-2, 5e-2
    return 3e-2, 3e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "K", "N", "group_size", "use_zero", "dtype"),
    [
        (1, 64, 32, 32, False, mx.float16),
        (1, 64, 32, 32, True, mx.bfloat16),
        (2, 128, 64, 32, False, mx.float16),
        (2, 128, 64, 32, True, mx.bfloat16),
    ],
)
def test_q4_matvec_decode(B, K, N, group_size, use_zero, dtype, backend):
    mx.random.seed(65)
    x = mx.random.normal((B, K)).astype(dtype)
    q = (mx.random.uniform((N, K)) * 16).astype(mx.uint8)
    packed = pack_q4(q)
    scales = mx.random.normal((N, K // group_size)).astype(mx.float32)
    zeros = (mx.random.uniform((N, K // group_size)) * 8).astype(mx.float32) if use_zero else None
    got = q4_matvec_decode(x, packed, scales, zeros, group_size=group_size, backend=backend)
    ref = reference_q4_matvec_decode(x, packed, scales, zeros, group_size=group_size)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "K", "N", "group_size", "use_zero", "dtype"),
    [
        (1, 64, 32, 32, False, mx.float16),
        (1, 64, 32, 32, True, mx.bfloat16),
        (2, 128, 64, 32, False, mx.float16),
        (2, 128, 64, 32, True, mx.bfloat16),
    ],
)
def test_q8_matvec_decode(B, K, N, group_size, use_zero, dtype, backend):
    mx.random.seed(66)
    x = mx.random.normal((B, K)).astype(dtype)
    q_w = (mx.random.uniform((N, K)) * 255).astype(mx.uint8)
    scales = mx.random.normal((N, K // group_size)).astype(mx.float32)
    zeros = (mx.random.uniform((N, K // group_size)) * 16).astype(mx.float32) if use_zero else None
    got = q8_matvec_decode(x, q_w, scales, zeros, group_size=group_size, backend=backend)
    ref = reference_q8_matvec_decode(x, q_w, scales, zeros, group_size=group_size)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
