import mlx.core as mx
import pytest

from ops.gqa_ops import (
    expand_kv_heads_reference,
    q_head_to_kv_head,
    reference_gqa_qkv_split,
    reference_gqa_qkv_split_rope,
    validate_gqa_heads,
)


def test_validate_gqa_heads_valid_and_invalid():
    validate_gqa_heads(4, 2)
    validate_gqa_heads(8, 1)
    with pytest.raises(ValueError, match=">="):
        validate_gqa_heads(2, 4)
    with pytest.raises(ValueError, match="divisible"):
        validate_gqa_heads(6, 4)


def test_q_head_to_kv_head_mapping():
    assert q_head_to_kv_head(0, 4, 2) == 0
    assert q_head_to_kv_head(1, 4, 2) == 0
    assert q_head_to_kv_head(2, 4, 2) == 1
    assert q_head_to_kv_head(3, 4, 2) == 1
    for hq in range(8):
        assert q_head_to_kv_head(hq, 8, 1) == 0


def test_expand_kv_heads_reference_repeats_values():
    kv = mx.arange(1 * 2 * 2 * 3).reshape(1, 2, 2, 3)
    expanded = expand_kv_heads_reference(kv, 4)
    assert expanded.shape == (1, 2, 4, 3)
    assert mx.all(expanded[:, :, 0, :] == expanded[:, :, 1, :]).item()
    assert mx.all(expanded[:, :, 2, :] == expanded[:, :, 3, :]).item()


def test_reference_gqa_qkv_split_shapes():
    B, S, Hq, Hkv, D = 1, 2, 4, 2, 16
    qkv = mx.random.normal((B, S, Hq * D + 2 * Hkv * D)).astype(mx.float16)
    q, k, v = reference_gqa_qkv_split(qkv, Hq, Hkv, D)
    assert q.shape == (B, S, Hq, D)
    assert k.shape == (B, S, Hkv, D)
    assert v.shape == (B, S, Hkv, D)


def test_reference_gqa_qkv_split_rope_shapes():
    B, S, Hq, Hkv, D = 1, 2, 4, 2, 16
    qkv = mx.random.normal((B, S, Hq * D + 2 * Hkv * D)).astype(mx.float16)
    cos = mx.random.normal((S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((S + 4, D // 2)).astype(mx.float32)
    q, k, v = reference_gqa_qkv_split_rope(qkv, cos, sin, Hq, Hkv, D)
    assert q.shape == (B, S, Hq, D)
    assert k.shape == (B, S, Hkv, D)
    assert v.shape == (B, S, Hkv, D)
