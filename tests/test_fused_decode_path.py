import mlx.core as mx
import pytest

from ops.fused_ops import fused_decode_step_from_qkv
from ops.fused_ops import qkv_rope_cache_update, reference_qkv_rope_cache_update
from ops.decode_ops import decode_attention, reference_decode_attention


def _reference_step(qkv, K_cache, V_cache, cos, sin, position, H, D):
    q_rope, K_cache, V_cache = reference_qkv_rope_cache_update(qkv, K_cache, V_cache, cos, sin, position, H=H, D=D)
    out = reference_decode_attention(q_rope, K_cache, V_cache, lengths=position + 1)
    return out, K_cache, V_cache


@pytest.mark.parametrize(("B", "H", "D", "MAX_S", "T"), [(1, 2, 16, 8, 4), (2, 4, 32, 8, 8)])
def test_fused_decode_path(B, H, D, MAX_S, T):
    mx.random.seed(43)
    dtype = mx.float16
    K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    ref_K = K_cache
    ref_V = V_cache
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    for pos in range(T):
        qkv = mx.random.normal((B, 1, 3 * H * D)).astype(dtype)
        got_out, K_cache, V_cache = fused_decode_step_from_qkv(
            qkv, K_cache, V_cache, cos, sin, pos, H=H, D=D, backend="metal"
        )
        ref_out, ref_K, ref_V = _reference_step(qkv, ref_K, ref_V, cos, sin, pos, H, D)
        mx.eval(got_out, K_cache, V_cache, ref_out, ref_K, ref_V)
        assert mx.allclose(got_out, ref_out, atol=2e-2, rtol=2e-2).item()
        assert mx.allclose(K_cache, ref_K, atol=2e-2, rtol=2e-2).item()
        assert mx.allclose(V_cache, ref_V, atol=2e-2, rtol=2e-2).item()
