import mlx.core as mx
import pytest

from ops.fused_ops import (
    qkv_rope_cache_update,
    reference_qkv_rope_cache_update,
    reference_residual_add,
    reference_rmsnorm_residual,
    residual_add,
    rmsnorm_residual,
)


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_residual_add(dtype, backend):
    mx.random.seed(40)
    x = mx.random.normal((2, 4, 64)).astype(dtype)
    residual = mx.random.normal((2, 4, 64)).astype(dtype)
    got = residual_add(x, residual, backend=backend)
    ref = reference_residual_add(x, residual)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(("shape", "return_residual"), [((2, 4, 128), False), ((1, 8, 1024), True)])
def test_rmsnorm_residual(dtype, backend, shape, return_residual):
    mx.random.seed(41)
    x = mx.random.normal(shape).astype(dtype)
    residual = mx.random.normal(shape).astype(dtype)
    weight = mx.random.normal((shape[-1],)).astype(dtype)
    got = rmsnorm_residual(x, residual, weight, return_residual=return_residual, backend=backend)
    ref = reference_rmsnorm_residual(x, residual, weight, return_residual=return_residual)
    atol, rtol = _tolerances(dtype)
    if return_residual:
        mx.eval(*got, *ref)
        for g, r in zip(got, ref):
            assert mx.allclose(g, r, atol=atol, rtol=rtol).item()
    else:
        mx.eval(got, ref)
        assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "MAX_S", "H", "D", "positions"),
    [
        (1, 8, 2, 8, 3),
        (2, 16, 4, 16, [2, 7]),
    ],
)
def test_qkv_rope_cache_update(B, MAX_S, H, D, positions, backend):
    mx.random.seed(42)
    dtype = mx.float16
    qkv = mx.random.normal((B, 1, 3 * H * D)).astype(dtype)
    K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    got = qkv_rope_cache_update(qkv, K_cache, V_cache, cos, sin, positions, H=H, D=D, backend=backend)
    ref = reference_qkv_rope_cache_update(qkv, K_cache, V_cache, cos, sin, positions, H=H, D=D)
    mx.eval(*got, *ref)
    for g, r in zip(got, ref):
        assert mx.allclose(g, r, atol=2e-2, rtol=2e-2).item()
