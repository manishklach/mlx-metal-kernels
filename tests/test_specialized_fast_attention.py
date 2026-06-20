import mlx.core as mx
import pytest

from ops.attention_ops import fast_attention, reference_attention


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(
    ("B", "S", "H", "D", "dtype", "causal", "backend"),
    [
        (1, 16, 2, 64, mx.float16, False, "baseline_d64"),
        (1, 16, 2, 64, mx.float16, True, "baseline_d64"),
        (2, 32, 4, 64, mx.float16, False, "baseline_d64"),
        (1, 16, 2, 128, mx.float16, False, "baseline_d128"),
        (1, 16, 2, 64, mx.bfloat16, False, "baseline_d64"),
    ],
)
def test_specialized_fast_attention_matches_reference(B, S, H, D, dtype, causal, backend):
    mx.random.seed(103)
    Q = mx.random.normal((B, S, H, D)).astype(dtype)
    K = mx.random.normal((B, S, H, D)).astype(dtype)
    V = mx.random.normal((B, S, H, D)).astype(dtype)
    got = fast_attention(Q, K, V, causal=causal, backend=backend)
    ref = reference_attention(Q, K, V, causal=causal)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_specialized_fast_attention_rejects_wrong_d():
    Q64 = mx.random.normal((1, 8, 2, 64)).astype(mx.float16)
    K64 = mx.random.normal((1, 8, 2, 64)).astype(mx.float16)
    V64 = mx.random.normal((1, 8, 2, 64)).astype(mx.float16)
    Q128 = mx.random.normal((1, 8, 2, 128)).astype(mx.float16)
    K128 = mx.random.normal((1, 8, 2, 128)).astype(mx.float16)
    V128 = mx.random.normal((1, 8, 2, 128)).astype(mx.float16)
    with pytest.raises(ValueError, match="requires D == 64"):
        fast_attention(Q128, K128, V128, backend="baseline_d64")
    with pytest.raises(ValueError, match="requires D == 128"):
        fast_attention(Q64, K64, V64, backend="baseline_d128")
