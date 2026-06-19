import mlx.core as mx
import pytest

from ops.quant_ops import (
    dequant_q4,
    dequant_q8,
    pack_q4,
    reference_dequant_q4,
    reference_dequant_q8,
    unpack_q4_reference,
)


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 5e-2, 5e-2
    return 3e-2, 3e-2


def test_pack_unpack_q4():
    mx.random.seed(60)
    q = (mx.random.uniform((4, 64)) * 16).astype(mx.uint8)
    packed = pack_q4(q)
    unpacked = unpack_q4_reference(packed, K=64)
    mx.eval(packed, unpacked)
    assert mx.all(unpacked == q).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dequant_q4_no_zero(dtype, backend):
    mx.random.seed(61)
    M, K, group_size = 4, 64, 32
    q = (mx.random.uniform((M, K)) * 16).astype(mx.uint8)
    packed = pack_q4(q)
    scales = mx.random.normal((M, K // group_size)).astype(mx.float32)
    got = dequant_q4(packed, scales, group_size=group_size, out_dtype=dtype, backend=backend)
    ref = reference_dequant_q4(packed, scales, group_size=group_size, out_dtype=dtype)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
def test_dequant_q4_with_zero(backend):
    mx.random.seed(62)
    M, K, group_size = 4, 64, 32
    q = (mx.random.uniform((M, K)) * 16).astype(mx.uint8)
    packed = pack_q4(q)
    scales = mx.random.normal((M, K // group_size)).astype(mx.float32)
    zeros = (mx.random.uniform((M, K // group_size)) * 8).astype(mx.float32)
    got = dequant_q4(packed, scales, zeros, group_size=group_size, out_dtype=mx.float16, backend=backend)
    ref = reference_dequant_q4(packed, scales, zeros, group_size=group_size, out_dtype=mx.float16)
    mx.eval(got, ref)
    assert mx.allclose(got, ref, atol=3e-2, rtol=3e-2).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dequant_q8_no_zero(dtype, backend):
    mx.random.seed(63)
    M, K, group_size = 4, 64, 32
    q = (mx.random.uniform((M, K)) * 255).astype(mx.uint8)
    scales = mx.random.normal((M, K // group_size)).astype(mx.float32)
    got = dequant_q8(q, scales, group_size=group_size, out_dtype=dtype, backend=backend)
    ref = reference_dequant_q8(q, scales, group_size=group_size, out_dtype=dtype)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
def test_dequant_q8_with_zero(backend):
    mx.random.seed(64)
    M, K, group_size = 4, 64, 32
    q = (mx.random.uniform((M, K)) * 255).astype(mx.uint8)
    scales = mx.random.normal((M, K // group_size)).astype(mx.float32)
    zeros = (mx.random.uniform((M, K // group_size)) * 16).astype(mx.float32)
    got = dequant_q8(q, scales, zeros, group_size=group_size, out_dtype=mx.float16, backend=backend)
    ref = reference_dequant_q8(q, scales, zeros, group_size=group_size, out_dtype=mx.float16)
    mx.eval(got, ref)
    assert mx.allclose(got, ref, atol=3e-2, rtol=3e-2).item()
