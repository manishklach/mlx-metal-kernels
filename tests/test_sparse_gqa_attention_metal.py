import pytest

mx = pytest.importorskip("mlx.core")

from ops.sparse_attention_ops import (
    SparseAttentionPattern,
    reference_sparse_gqa_attention,
    sparse_gqa_attention,
)


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 6e-2, 6e-2
    return 4e-2, 4e-2


@pytest.mark.parametrize(
    ("Hq", "Hkv", "D", "sink_tokens", "dtype"),
    [
        (4, 2, 16, 0, mx.float16),
        (4, 2, 16, 2, mx.float16),
        (4, 1, 16, 2, mx.float16),
        (4, 4, 16, 2, mx.float16),
        (4, 2, 64, 2, mx.float16),
        pytest.param(4, 2, 64, 2, mx.bfloat16, marks=pytest.mark.skipif(not hasattr(mx, "bfloat16"), reason="bf16 unavailable")),
    ],
)
def test_sparse_gqa_attention_metal_matches_reference(Hq, Hkv, D, sink_tokens, dtype):
    mx.random.seed(902)
    B, S = 1, 16
    Q = mx.random.normal((B, S, Hq, D)).astype(dtype)
    K = mx.random.normal((B, S, Hkv, D)).astype(dtype)
    V = mx.random.normal((B, S, Hkv, D)).astype(dtype)
    pattern_name = "sliding_window_sink" if sink_tokens else "sliding_window"
    pattern = SparseAttentionPattern(pattern=pattern_name, window_size=4, sink_tokens=sink_tokens)
    backend = "metal_sliding_window_sink" if sink_tokens else "metal_sliding_window"
    got = sparse_gqa_attention(Q, K, V, pattern, backend=backend)
    ref = reference_sparse_gqa_attention(Q, K, V, pattern)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
