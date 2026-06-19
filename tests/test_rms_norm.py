import mlx.core as mx
import pytest

from ops.norm_ops import reference_rms_norm, rms_norm


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "S", "D", "dtype"),
    [
        (1, 4, 64, mx.float16),
        (1, 4, 64, mx.bfloat16),
        (2, 8, 128, mx.float16),
        (2, 8, 128, mx.bfloat16),
        (1, 16, 1024, mx.float16),
        (1, 16, 1024, mx.bfloat16),
    ],
)
def test_rms_norm_matches_reference(B, S, D, dtype, backend):
    mx.random.seed(0)
    x = mx.random.normal((B, S, D)).astype(dtype)
    weight = mx.random.normal((D,)).astype(dtype)
    got = rms_norm(x, weight, backend=backend)
    ref = reference_rms_norm(x, weight)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
