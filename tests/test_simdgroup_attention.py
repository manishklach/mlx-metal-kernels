import mlx.core as mx
import pytest

from ops.attention_ops import fast_attention, reference_attention


def _tol():
    return 3e-2, 3e-2


def _skip_if_unavailable(exc: Exception) -> None:
    message = str(exc)
    if "simdgroup_d64 backend unavailable" in message:
        pytest.skip(message)
    raise exc


@pytest.mark.parametrize(
    ("B", "S", "H", "D", "causal"),
    [
        (1, 16, 2, 64, False),
        (1, 32, 2, 64, False),
        (1, 16, 2, 64, True),
    ],
)
def test_simdgroup_attention_matches_reference(B, S, H, D, causal):
    mx.random.seed(216)
    Q = mx.random.normal((B, S, H, D)).astype(mx.float16)
    K = mx.random.normal((B, S, H, D)).astype(mx.float16)
    V = mx.random.normal((B, S, H, D)).astype(mx.float16)
    try:
        got = fast_attention(Q, K, V, causal=causal, backend="simdgroup_d64")
    except Exception as exc:  # noqa: BLE001
        _skip_if_unavailable(exc)
    ref = reference_attention(Q, K, V, causal=causal)
    mx.eval(got, ref)
    atol, rtol = _tol()
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_simdgroup_attention_rejects_wrong_d():
    Q = mx.random.normal((1, 16, 2, 32)).astype(mx.float16)
    K = mx.random.normal((1, 16, 2, 32)).astype(mx.float16)
    V = mx.random.normal((1, 16, 2, 32)).astype(mx.float16)
    with pytest.raises(ValueError, match="requires D == 64"):
        fast_attention(Q, K, V, backend="simdgroup_d64")


def test_simdgroup_attention_rejects_bfloat16():
    Q = mx.random.normal((1, 16, 2, 64)).astype(mx.bfloat16)
    K = mx.random.normal((1, 16, 2, 64)).astype(mx.bfloat16)
    V = mx.random.normal((1, 16, 2, 64)).astype(mx.bfloat16)
    with pytest.raises(ValueError, match="currently supports only mx.float16"):
        fast_attention(Q, K, V, backend="simdgroup_d64")
