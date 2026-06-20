import mlx.core as mx
import pytest

from ops.decode_ops import reference_decode_attention
from ops.gqa_ops import expand_kv_heads_reference, reference_gqa_decode_attention


@pytest.mark.parametrize(
    ("B", "MAX_S", "Hq", "Hkv", "D", "lengths"),
    [
        (1, 8, 4, 2, 16, 8),
        (2, 16, 8, 2, 16, [8, 12]),
        (1, 8, 4, 1, 16, 8),
    ],
)
def test_reference_gqa_decode_attention_matches_expansion(B, MAX_S, Hq, Hkv, D, lengths):
    mx.random.seed(241)
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    got = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
    ref = reference_decode_attention(
        q,
        expand_kv_heads_reference(K_cache, Hq),
        expand_kv_heads_reference(V_cache, Hq),
        lengths=lengths,
    )
    mx.eval(got, ref)
    assert got.shape == (B, 1, Hq, D)
    assert mx.allclose(got, ref, atol=6e-2, rtol=6e-2).item()
