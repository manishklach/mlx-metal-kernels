import mlx.core as mx
import pytest

from ops.activation_ops import reference_swiglu, swiglu


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "S", "D", "dtype"),
    [
        (1, 8, 64, mx.float16),
        (1, 8, 64, mx.bfloat16),
        (2, 16, 256, mx.float16),
        (2, 16, 256, mx.bfloat16),
    ],
)
def test_swiglu_matches_reference(B, S, D, dtype, backend):
    mx.random.seed(2)
    gate = mx.random.normal((B, S, D)).astype(dtype)
    up = mx.random.normal((B, S, D)).astype(dtype)
    got = swiglu(gate, up, backend=backend)
    ref = reference_swiglu(gate, up)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
