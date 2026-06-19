import mlx.core as mx
import pytest

from ops.decode_ops import decode_step, reference_decode_step


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(("B", "H", "D", "MAX_S"), [(1, 2, 32, 16), (2, 4, 64, 16)])
def test_decode_step_matches_reference(B, H, D, MAX_S):
    mx.random.seed(13)
    dtype = mx.float16
    T = 8
    K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    ref_K = K_cache
    ref_V = V_cache

    for pos in range(T):
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        k_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        v_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        got_out, K_cache, V_cache = decode_step(q, k_new, v_new, K_cache, V_cache, pos, backend="metal")
        ref_out, ref_K, ref_V = reference_decode_step(q, k_new, v_new, ref_K, ref_V, pos)
        mx.eval(got_out, K_cache, V_cache, ref_out, ref_K, ref_V)
        atol, rtol = _tolerances(dtype)
        assert mx.allclose(got_out, ref_out, atol=atol, rtol=rtol).item()
        assert mx.allclose(K_cache, ref_K, atol=atol, rtol=rtol).item()
        assert mx.allclose(V_cache, ref_V, atol=atol, rtol=rtol).item()
