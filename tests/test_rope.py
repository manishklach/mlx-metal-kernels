import mlx.core as mx
import pytest

from ops.rope_ops import apply_rope, reference_apply_rope


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "S", "H", "D", "position_offset", "dtype"),
    [
        (1, 8, 4, 64, 0, mx.float16),
        (1, 8, 4, 64, 5, mx.bfloat16),
        (2, 16, 8, 128, 0, mx.float16),
        (2, 16, 8, 128, 5, mx.bfloat16),
    ],
)
def test_rope_matches_reference(B, S, H, D, position_offset, dtype, backend):
    mx.random.seed(1)
    x = mx.random.normal((B, S, H, D)).astype(dtype)
    total_positions = S + position_offset + 4
    cos = mx.random.normal((total_positions, D // 2)).astype(mx.float32)
    sin = mx.random.normal((total_positions, D // 2)).astype(mx.float32)
    got = apply_rope(x, cos, sin, position_offset=position_offset, backend=backend)
    ref = reference_apply_rope(x, cos, sin, position_offset=position_offset)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
