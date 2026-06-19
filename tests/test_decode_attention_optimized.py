import mlx.core as mx
import pytest

from ops.decode_ops import decode_attention, reference_decode_attention


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "MAX_S", "H", "D", "lengths", "dtype"),
    [
        (1, 16, 2, 32, 8, mx.float16),
        (1, 32, 4, 64, 32, mx.float16),
        (2, 32, 4, 64, [12, 20], mx.float16),
        (1, 16, 2, 32, 10, mx.bfloat16),
        (1, 8, 1, 128, 8, mx.float16),
    ],
)
def test_decode_attention_matches_reference(B, MAX_S, H, D, lengths, dtype, backend):
    mx.random.seed(12)
    q = mx.random.normal((B, 1, H, D)).astype(dtype)
    K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    got = decode_attention(q, K_cache, V_cache, lengths=lengths, backend=backend)
    ref = reference_decode_attention(q, K_cache, V_cache, lengths=lengths)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
