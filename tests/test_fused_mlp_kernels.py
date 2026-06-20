import pytest

mx = pytest.importorskip("mlx.core")

from ops.activation_ops import fused_swiglu, reference_swiglu


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 5e-2, 5e-2
    return 3e-2, 3e-2


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 1, 128), mx.float16),
        ((2, 4, 128), mx.float16),
        pytest.param((1, 2, 64), mx.bfloat16, marks=pytest.mark.skipif(not hasattr(mx, "bfloat16"), reason="bf16 unavailable")),
    ],
)
def test_fused_swiglu_matches_reference(shape, dtype):
    mx.random.seed(601)
    gate = mx.random.normal(shape).astype(dtype)
    up = mx.random.normal(shape).astype(dtype)
    got = fused_swiglu(gate, up, backend="metal_fused")
    ref = reference_swiglu(gate, up)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert got.shape == gate.shape
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_fused_swiglu_supports_flattened_rows():
    mx.random.seed(602)
    gate = mx.random.normal((8, 96)).astype(mx.float16)
    up = mx.random.normal((8, 96)).astype(mx.float16)
    got = fused_swiglu(gate, up, backend="metal_fused")
    ref = fused_swiglu(gate, up, backend="reference")
    mx.eval(got, ref)
    assert got.shape == gate.shape
    assert mx.allclose(got, ref, atol=3e-2, rtol=3e-2).item()

