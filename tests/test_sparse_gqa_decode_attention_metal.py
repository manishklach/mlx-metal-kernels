import pytest

mx = pytest.importorskip("mlx.core")

from ops.sparse_attention_ops import (
    SparseAttentionPattern,
    reference_sparse_gqa_decode_attention,
    sparse_gqa_decode_attention,
)


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 6e-2, 6e-2
    return 4e-2, 4e-2


@pytest.mark.parametrize(
    ("B", "Hq", "Hkv", "D", "sink_tokens", "lengths"),
    [
        (1, 4, 2, 16, 0, 16),
        (1, 4, 2, 16, 2, 16),
        (2, 4, 2, 16, 2, [16, 11]),
        (1, 4, 1, 16, 2, 16),
        (1, 4, 4, 16, 2, 16),
        (1, 4, 2, 64, 2, 16),
    ],
)
def test_sparse_gqa_decode_attention_metal_matches_reference(B, Hq, Hkv, D, sink_tokens, lengths):
    mx.random.seed(903)
    MAX_S = 32
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    pattern_name = "sliding_window_sink" if sink_tokens else "sliding_window"
    pattern = SparseAttentionPattern(pattern=pattern_name, window_size=4, sink_tokens=sink_tokens)
    backend = "metal_sliding_window_sink" if sink_tokens else "metal_sliding_window"
    got = sparse_gqa_decode_attention(q, K_cache, V_cache, lengths, pattern, backend=backend)
    ref = reference_sparse_gqa_decode_attention(q, K_cache, V_cache, lengths, pattern)
    mx.eval(got, ref)
    atol, rtol = _tol(mx.float16)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
