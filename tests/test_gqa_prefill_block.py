import pytest

mx = pytest.importorskip("mlx.core")

from ops.gqa_ops import gqa_attention, reference_gqa_attention, reference_gqa_qkv_split
from ops.rope_ops import apply_rope, reference_apply_rope


def test_gqa_prefill_split_and_attention_composition():
    mx.random.seed(803)
    B, S, Hq, Hkv, D = 1, 8, 4, 2, 16
    qkv = mx.random.normal((B, S, Hq * D + 2 * Hkv * D)).astype(mx.float16)
    cos = mx.random.normal((S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((S + 4, D // 2)).astype(mx.float32)
    q, k, v = reference_gqa_qkv_split(qkv, Hq, Hkv, D)
    q_rope = reference_apply_rope(q, cos, sin)
    k_rope = reference_apply_rope(k, cos, sin)
    got = gqa_attention(q_rope, k_rope, v, backend="metal_gqa_threadgroup", causal=True)
    ref = reference_gqa_attention(q_rope, k_rope, v, causal=True)
    mx.eval(got, ref)
    assert mx.allclose(got, ref, atol=6e-2, rtol=6e-2).item()
