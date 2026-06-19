import mlx.core as mx
import pytest

from ops.decode_block_ops import (
    reference_residual_rmsnorm_block,
    residual_rmsnorm_block,
)
from ops.fused_ops import reference_rmsnorm_residual, rmsnorm_residual


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("shape", [(2, 4, 128), (1, 8, 1024)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("return_residual", [False, True])
def test_residual_rmsnorm_block_matches_existing_helper(shape, dtype, return_residual):
    mx.random.seed(93)
    x = mx.random.normal(shape).astype(dtype)
    residual = mx.random.normal(shape).astype(dtype)
    weight = mx.random.normal((shape[-1],)).astype(dtype)

    got = residual_rmsnorm_block(
        x, residual, weight, return_residual=return_residual, backend="metal"
    )
    ref = rmsnorm_residual(
        x, residual, weight, return_residual=return_residual, backend="metal"
    )
    golden = reference_residual_rmsnorm_block(
        x, residual, weight, return_residual=return_residual
    )
    baseline = reference_rmsnorm_residual(
        x, residual, weight, return_residual=return_residual
    )

    if return_residual:
        mx.eval(got[0], got[1], ref[0], ref[1], golden[0], golden[1], baseline[0], baseline[1])
        atol, rtol = _tol(dtype)
        assert mx.allclose(got[0], ref[0], atol=atol, rtol=rtol).item()
        assert mx.allclose(got[1], ref[1], atol=atol, rtol=rtol).item()
        assert mx.allclose(golden[0], baseline[0], atol=atol, rtol=rtol).item()
        assert mx.allclose(golden[1], baseline[1], atol=atol, rtol=rtol).item()
    else:
        mx.eval(got, ref, golden, baseline)
        atol, rtol = _tol(dtype)
        assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
        assert mx.allclose(golden, baseline, atol=atol, rtol=rtol).item()
