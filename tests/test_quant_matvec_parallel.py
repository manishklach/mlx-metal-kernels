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


@pytest.mark.parametrize(
    ("B", "K", "N", "group_size", "use_zero", "dtype", "use_token_shape"),
    [
        (1, 64, 32, 32, False, mx.float16, False),
        (2, 128, 64, 32, True, mx.float16, False),
        (1, 96, 48, 16, False, mx.float16, False),
        (1, 128, 64, 32, True, mx.bfloat16, False),
        (1, 64, 32, 32, False, mx.float16, True),
    ],
)
def test_q4_matvec_decode_parallel(B, K, N, group_size, use_zero, dtype, use_token_shape):
    mx.random.seed(121)
    x = mx.random.normal((B, 1, K)).astype(dtype) if use_token_shape else mx.random.normal((B, K)).astype(dtype)
    q = (mx.random.uniform((N, K)) * 16).astype(mx.uint8)
    packed = pack_q4(q)
    groups = (K + group_size - 1) // group_size
    scales = mx.random.normal((N, groups)).astype(mx.float32)
    zeros = (mx.random.uniform((N, groups)) * 8).astype(mx.float32) if use_zero else None
    ref = reference_q4_matvec_decode(x, packed, scales, zeros, group_size=group_size)
    got_parallel = q4_matvec_decode(x, packed, scales, zeros, group_size=group_size, backend="metal_parallel")
    got_metal = q4_matvec_decode(x, packed, scales, zeros, group_size=group_size, backend="metal")
    mx.eval(ref, got_parallel, got_metal)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got_parallel, ref, atol=atol, rtol=rtol).item()
    assert mx.allclose(got_parallel, got_metal, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize(
    ("B", "K", "N", "group_size", "use_zero", "dtype", "use_token_shape"),
    [
        (1, 64, 32, 32, False, mx.float16, False),
        (2, 128, 64, 32, True, mx.float16, False),
        (1, 96, 48, 16, False, mx.float16, False),
        (1, 128, 64, 32, True, mx.bfloat16, False),
        (1, 64, 32, 32, False, mx.float16, True),
    ],
)
def test_q8_matvec_decode_parallel(B, K, N, group_size, use_zero, dtype, use_token_shape):
    mx.random.seed(122)
    x = mx.random.normal((B, 1, K)).astype(dtype) if use_token_shape else mx.random.normal((B, K)).astype(dtype)
    q_w = (mx.random.uniform((N, K)) * 255).astype(mx.uint8)
    groups = (K + group_size - 1) // group_size
    scales = mx.random.normal((N, groups)).astype(mx.float32)
    zeros = (mx.random.uniform((N, groups)) * 16).astype(mx.float32) if use_zero else None
    ref = reference_q8_matvec_decode(x, q_w, scales, zeros, group_size=group_size)
    got_parallel = q8_matvec_decode(x, q_w, scales, zeros, group_size=group_size, backend="metal_parallel")
    got_metal = q8_matvec_decode(x, q_w, scales, zeros, group_size=group_size, backend="metal")
    mx.eval(ref, got_parallel, got_metal)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got_parallel, ref, atol=atol, rtol=rtol).item()
    assert mx.allclose(got_parallel, got_metal, atol=atol, rtol=rtol).item()
