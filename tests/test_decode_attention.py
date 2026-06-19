import mlx.core as mx
import pytest

from ops.decode_ops import decode_attention, reference_decode_attention


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "S", "H", "D", "dtype"),
    [
        (1, 16, 4, 64, mx.float16),
        (1, 16, 4, 64, mx.bfloat16),
        (2, 32, 8, 64, mx.float16),
        (2, 32, 8, 64, mx.bfloat16),
    ],
)
def test_decode_attention_matches_reference(B, S, H, D, dtype, backend):
    mx.random.seed(3)
    q = mx.random.normal((B, 1, H, D)).astype(dtype)
    K_cache = mx.random.normal((B, S, H, D)).astype(dtype)
    V_cache = mx.random.normal((B, S, H, D)).astype(dtype)
    got = decode_attention(q, K_cache, V_cache, backend=backend)
    ref = reference_decode_attention(q, K_cache, V_cache)
    mx.eval(got, ref)
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_decode_attention_rejects_non_singleton_query():
    mx.random.seed(4)
    q = mx.random.normal((1, 2, 4, 64)).astype(mx.float16)
    K_cache = mx.random.normal((1, 16, 4, 64)).astype(mx.float16)
    V_cache = mx.random.normal((1, 16, 4, 64)).astype(mx.float16)
    with pytest.raises(ValueError, match="q.shape\\[1\\] == 1"):
        decode_attention(q, K_cache, V_cache)
