import math

import pytest

mx = pytest.importorskip("mlx.core")

from ops.gqa_ops import expand_kv_heads_reference
from ops.sparse_attention_ops import (
    SparseAttentionPattern,
    build_sparse_attention_mask,
    reference_sparse_gqa_attention,
)


def test_sliding_window_mask_causal():
    pattern = SparseAttentionPattern(pattern="sliding_window", window_size=3)
    mask = build_sparse_attention_mask(4, 6, pattern, start_position=2)
    rows = mask.tolist()
    assert rows[0] == [True, True, True, False, False, False]
    assert rows[1] == [False, True, True, True, False, False]
    assert rows[3] == [False, False, False, True, True, True]


def test_sliding_window_sink_mask_includes_sink_tokens():
    pattern = SparseAttentionPattern(pattern="sliding_window_sink", window_size=3, sink_tokens=2)
    mask = build_sparse_attention_mask(1, 6, pattern, start_position=4)
    assert mask.tolist()[0] == [True, True, True, True, True, False]


def test_sink_local_overlap_does_not_duplicate_reference():
    pattern = SparseAttentionPattern(pattern="sliding_window_sink", window_size=4, sink_tokens=2)
    mask = build_sparse_attention_mask(1, 5, pattern, start_position=1)
    assert mask.tolist()[0] == [True, True, False, False, False]


def test_block_sparse_mask_shape():
    block_mask = [[True, False], [True, True]]
    pattern = SparseAttentionPattern(pattern="block_sparse", block_size=2, block_mask=block_mask)
    mask = build_sparse_attention_mask(4, 4, pattern)
    assert mask.shape == (4, 4)


@pytest.mark.parametrize(("Hq", "Hkv"), [(4, 2), (4, 1), (4, 4)])
def test_reference_sparse_gqa_attention_matches_dense_masked_reference(Hq, Hkv):
    mx.random.seed(901)
    B, Sq, Sk, D = 1, 6, 6, 8
    Q = mx.random.normal((B, Sq, Hq, D)).astype(mx.float16)
    K = mx.random.normal((B, Sk, Hkv, D)).astype(mx.float16)
    V = mx.random.normal((B, Sk, Hkv, D)).astype(mx.float16)
    pattern = SparseAttentionPattern(pattern="sliding_window_sink", window_size=3, sink_tokens=1)
    got = reference_sparse_gqa_attention(Q, K, V, pattern)

    K_exp = expand_kv_heads_reference(K, Hq)
    V_exp = expand_kv_heads_reference(V, Hq)
    mask = build_sparse_attention_mask(Sq, Sk, pattern)
    scores = mx.matmul(
        Q.astype(mx.float32).transpose(0, 2, 1, 3),
        K_exp.astype(mx.float32).transpose(0, 2, 3, 1),
    ) * float(1.0 / math.sqrt(D))
    neg_inf = mx.array(-1.0e9, dtype=scores.dtype)
    mask4 = mask.reshape(1, 1, Sq, Sk)
    scores = mx.where(mask4, scores, neg_inf)
    probs = mx.softmax(scores, axis=-1)
    ref = mx.matmul(probs, V_exp.astype(mx.float32).transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).astype(mx.float16)

    mx.eval(got, ref)
    assert mx.allclose(got, ref, atol=6e-2, rtol=6e-2).item()


def test_invalid_window_size_raises():
    with pytest.raises(ValueError, match="window_size > 0"):
        SparseAttentionPattern(pattern="sliding_window", window_size=0).validate()


def test_invalid_hq_hkv_raises():
    Q = mx.zeros((1, 4, 3, 8), dtype=mx.float16)
    K = mx.zeros((1, 4, 2, 8), dtype=mx.float16)
    V = mx.zeros((1, 4, 2, 8), dtype=mx.float16)
    pattern = SparseAttentionPattern(pattern="sliding_window", window_size=2)
    with pytest.raises(ValueError, match="divisible"):
        reference_sparse_gqa_attention(Q, K, V, pattern)
