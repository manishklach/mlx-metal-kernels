import mlx.core as mx
import pytest

from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_step, reference_paged_decode_step


@pytest.mark.parametrize(("B", "H", "D", "MAX_S", "PAGE_SIZE"), [(1, 2, 32, 16, 4), (2, 4, 64, 16, 4)])
def test_paged_decode_step(B, H, D, MAX_S, PAGE_SIZE):
    mx.random.seed(82)
    dtype = mx.float16
    T = 8
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    ref_K, ref_V = K_pages, V_pages
    for pos in range(T):
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        k_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        v_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        got_out, K_pages, V_pages = paged_decode_step(q, k_new, v_new, K_pages, V_pages, block_table, pos, backend="metal")
        ref_out, ref_K, ref_V = reference_paged_decode_step(q, k_new, v_new, ref_K, ref_V, block_table, pos)
        mx.eval(got_out, K_pages, V_pages, ref_out, ref_K, ref_V)
        assert mx.allclose(got_out, ref_out, atol=2e-2, rtol=2e-2).item()
        assert mx.allclose(K_pages, ref_K, atol=2e-2, rtol=2e-2).item()
        assert mx.allclose(V_pages, ref_V, atol=2e-2, rtol=2e-2).item()
