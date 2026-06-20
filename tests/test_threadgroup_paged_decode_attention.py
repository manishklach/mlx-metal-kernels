import mlx.core as mx
import pytest

from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention, reference_paged_decode_attention


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(
    ("B", "MAX_S", "PAGE_SIZE", "H", "D", "lengths", "dtype"),
    [
        (1, 16, 4, 2, 32, 8, mx.float16),
        (1, 32, 8, 4, 64, 32, mx.float16),
        (2, 32, 8, 4, 64, [12, 20], mx.float16),
        (1, 16, 4, 2, 128, 16, mx.float16),
        (1, 16, 4, 2, 64, 10, mx.bfloat16),
    ],
)
def test_threadgroup_paged_decode_attention_matches_reference(B, MAX_S, PAGE_SIZE, H, D, lengths, dtype):
    mx.random.seed(207)
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    K_pages = mx.random.normal(K_pages.shape).astype(dtype)
    V_pages = mx.random.normal(V_pages.shape).astype(dtype)
    q = mx.random.normal((B, 1, H, D)).astype(dtype)
    got = paged_decode_attention(q, K_pages, V_pages, block_table, lengths, backend="metal_threadgroup")
    ref = reference_paged_decode_attention(q, K_pages, V_pages, block_table, lengths)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
