import mlx.core as mx
import pytest

from ops.attention_ops import fast_attention, reference_attention


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(
    ("B", "S", "H", "D", "dtype", "causal"),
    [
        (1, 16, 2, 32, mx.float16, False),
        (1, 16, 2, 32, mx.float16, True),
        (1, 32, 4, 64, mx.float16, False),
        (1, 32, 4, 64, mx.float16, True),
        (1, 16, 2, 128, mx.float16, False),
        (1, 16, 2, 64, mx.bfloat16, False),
    ],
)
def test_threadgroup_fast_attention_matches_reference(B, S, H, D, dtype, causal):
    mx.random.seed(208)
    Q = mx.random.normal((B, S, H, D)).astype(dtype)
    K = mx.random.normal((B, S, H, D)).astype(dtype)
    V = mx.random.normal((B, S, H, D)).astype(dtype)
    got = fast_attention(Q, K, V, causal=causal, backend="threadgroup")
    ref = reference_attention(Q, K, V, causal=causal)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
