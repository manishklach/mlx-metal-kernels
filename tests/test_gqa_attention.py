import pytest

mx = pytest.importorskip("mlx.core")

from ops.gqa_ops import reference_gqa_attention, reference_gqa_attention_via_expansion


@pytest.mark.parametrize(
    ("B", "Sq", "Sk", "Hq", "Hkv", "D", "causal"),
    [
        (1, 8, 8, 4, 2, 16, False),
        (1, 8, 8, 4, 2, 16, True),
        (1, 8, 8, 4, 1, 16, False),
        (1, 8, 8, 4, 4, 16, False),
        (2, 8, 8, 8, 2, 16, False),
    ],
)
def test_reference_gqa_attention_matches_expansion(B, Sq, Sk, Hq, Hkv, D, causal):
    mx.random.seed(801)
    Q = mx.random.normal((B, Sq, Hq, D)).astype(mx.float16)
    K = mx.random.normal((B, Sk, Hkv, D)).astype(mx.float16)
    V = mx.random.normal((B, Sk, Hkv, D)).astype(mx.float16)
    got = reference_gqa_attention(Q, K, V, causal=causal)
    ref = reference_gqa_attention_via_expansion(Q, K, V, causal=causal)
    mx.eval(got, ref)
    assert got.shape == (B, Sq, Hq, D)
    assert mx.allclose(got, ref, atol=6e-2, rtol=6e-2).item()


def test_reference_gqa_attention_invalid_head_ratio_raises():
    Q = mx.zeros((1, 8, 3, 16), dtype=mx.float16)
    K = mx.zeros((1, 8, 2, 16), dtype=mx.float16)
    V = mx.zeros((1, 8, 2, 16), dtype=mx.float16)
    with pytest.raises(ValueError, match="divisible"):
        reference_gqa_attention(Q, K, V)


def test_reference_gqa_attention_causal_mismatched_lengths_raise():
    Q = mx.zeros((1, 4, 4, 16), dtype=mx.float16)
    K = mx.zeros((1, 8, 2, 16), dtype=mx.float16)
    V = mx.zeros((1, 8, 2, 16), dtype=mx.float16)
    with pytest.raises(ValueError, match="Sq == Sk"):
        reference_gqa_attention(Q, K, V, causal=True)
