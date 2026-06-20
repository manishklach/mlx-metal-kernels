import pytest

mx = pytest.importorskip("mlx.core")

from ops.gqa_ops import gqa_attention, reference_gqa_attention


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 5e-2, 5e-2
    return 3e-2, 3e-2


def _run_backend(Q, K, V, *, backend, causal):
    try:
        return gqa_attention(Q, K, V, backend=backend, causal=causal)
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if any(token in message for token in ("metal", "compile", "unavailable", "not supported")):
            pytest.skip(f"{backend} unavailable in this environment: {exc}")
        raise


@pytest.mark.parametrize(
    ("backend", "B", "S", "Hq", "Hkv", "D", "causal", "dtype"),
    [
        ("metal_gqa", 1, 8, 4, 2, 16, False, mx.float16),
        ("metal_gqa", 1, 8, 4, 2, 16, True, mx.float16),
        ("metal_gqa_threadgroup", 1, 8, 4, 2, 16, False, mx.float16),
        ("metal_gqa_threadgroup", 1, 8, 4, 2, 16, True, mx.float16),
        ("metal_gqa", 1, 8, 4, 1, 16, False, mx.float16),
        ("metal_gqa_threadgroup", 1, 8, 4, 4, 16, False, mx.float16),
        ("metal_gqa_threadgroup", 1, 16, 8, 2, 64, False, mx.float16),
        ("metal_gqa_threadgroup", 1, 16, 8, 2, 128, False, mx.float16),
        pytest.param("metal_gqa_threadgroup", 1, 8, 4, 2, 16, False, mx.bfloat16, marks=pytest.mark.skipif(not hasattr(mx, "bfloat16"), reason="bf16 unavailable")),
    ],
)
def test_gqa_attention_metal_matches_reference(backend, B, S, Hq, Hkv, D, causal, dtype):
    mx.random.seed(802)
    Q = mx.random.normal((B, S, Hq, D)).astype(dtype)
    K = mx.random.normal((B, S, Hkv, D)).astype(dtype)
    V = mx.random.normal((B, S, Hkv, D)).astype(dtype)
    got = _run_backend(Q, K, V, backend=backend, causal=causal)
    ref = reference_gqa_attention(Q, K, V, causal=causal)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert got.shape == ref.shape
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
