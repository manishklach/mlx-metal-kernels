import mlx.core as mx
import pytest

from ops.layout_ops import qkv_split, qkv_split_rope, reference_qkv_split, reference_qkv_split_rope


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "S", "H", "D", "dtype"),
    [
        (1, 4, 2, 8, mx.float16),
        (1, 4, 2, 8, mx.bfloat16),
        (2, 3, 4, 16, mx.float16),
        (2, 3, 4, 16, mx.bfloat16),
    ],
)
def test_qkv_split_packed_matches_reference(B, S, H, D, dtype, backend):
    mx.random.seed(30)
    qkv = mx.random.normal((B, S, 3 * H * D)).astype(dtype)
    got = qkv_split(qkv, H=H, D=D, backend=backend)
    ref = reference_qkv_split(qkv, H=H, D=D)
    mx.eval(*got, *ref)
    atol, rtol = _tolerances(dtype)
    for g, r in zip(got, ref):
        assert mx.allclose(g, r, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
def test_qkv_split_explicit_matches_reference(backend):
    mx.random.seed(31)
    qkv = mx.random.normal((1, 4, 3, 2, 8)).astype(mx.float16)
    got = qkv_split(qkv, backend=backend)
    ref = reference_qkv_split(qkv)
    mx.eval(*got, *ref)
    for g, r in zip(got, ref):
        assert mx.allclose(g, r, atol=2e-2, rtol=2e-2).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "S", "H", "D", "position_offset", "dtype"),
    [
        (1, 4, 2, 8, 0, mx.float16),
        (1, 4, 2, 8, 3, mx.bfloat16),
        (2, 5, 4, 16, 0, mx.float16),
        (2, 5, 4, 16, 3, mx.bfloat16),
    ],
)
def test_qkv_split_rope_matches_reference(B, S, H, D, position_offset, dtype, backend):
    mx.random.seed(32)
    qkv = mx.random.normal((B, S, 3 * H * D)).astype(dtype)
    cos = mx.random.normal((S + position_offset + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((S + position_offset + 4, D // 2)).astype(mx.float32)
    got = qkv_split_rope(qkv, cos, sin, H=H, D=D, position_offset=position_offset, backend=backend)
    ref = reference_qkv_split_rope(qkv, cos, sin, H=H, D=D, position_offset=position_offset)
    mx.eval(*got, *ref)
    atol, rtol = _tolerances(dtype)
    for g, r in zip(got, ref):
        assert mx.allclose(g, r, atol=atol, rtol=rtol).item()
