import pytest

mx = pytest.importorskip("mlx.core")

from ops.sparse_attention_ops import (
    SparseAttentionPattern,
    build_sparse_attention_mask,
    reference_sparse_gqa_attention,
)


def test_window_size_one_means_current_token_only():
    pattern = SparseAttentionPattern(pattern="sliding_window", window_size=1)
    mask = build_sparse_attention_mask(4, 4, pattern)
    assert mask.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]


def test_window_ge_sequence_matches_causal_dense():
    pattern = SparseAttentionPattern(pattern="sliding_window", window_size=8)
    mask = build_sparse_attention_mask(4, 4, pattern)
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_sink_tokens_sequence_matches_causal_dense():
    pattern = SparseAttentionPattern(pattern="sliding_window_sink", window_size=2, sink_tokens=4)
    mask = build_sparse_attention_mask(4, 4, pattern)
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_sliding_window_sink_expected_positions():
    pattern = SparseAttentionPattern(pattern="sliding_window_sink", window_size=4, sink_tokens=2)
    mask = build_sparse_attention_mask(1, 8, pattern, start_position=5)
    assert mask.tolist()[0] == [True, True, True, True, True, True, False, False]


def test_sparse_dense_agree_when_window_covers_full_sequence():
    mx.random.seed(904)
    Q = mx.random.normal((1, 4, 4, 8)).astype(mx.float16)
    K = mx.random.normal((1, 4, 4, 8)).astype(mx.float16)
    V = mx.random.normal((1, 4, 4, 8)).astype(mx.float16)
    dense = reference_sparse_gqa_attention(Q, K, V, SparseAttentionPattern(pattern="dense"))
    wide = reference_sparse_gqa_attention(Q, K, V, SparseAttentionPattern(pattern="sliding_window", window_size=8))
    mx.eval(dense, wide)
    assert mx.allclose(dense, wide, atol=6e-2, rtol=6e-2).item()
