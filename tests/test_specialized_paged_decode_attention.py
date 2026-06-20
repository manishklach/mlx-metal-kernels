import mlx.core as mx
import pytest

from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention, reference_paged_decode_attention


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(
    ("B", "MAX_S", "PAGE_SIZE", "H", "D", "lengths", "dtype", "backend"),
    [
        (1, 16, 4, 2, 64, 8, mx.float16, "metal_d64"),
        (2, 32, 8, 4, 64, [12, 20], mx.float16, "metal_d64"),
        (1, 16, 4, 2, 128, 8, mx.float16, "metal_d128"),
        (1, 16, 4, 2, 64, 10, mx.bfloat16, "metal_d64"),
    ],
)
def test_specialized_paged_decode_attention_matches_reference(B, MAX_S, PAGE_SIZE, H, D, lengths, dtype, backend):
    mx.random.seed(102)
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    K_pages = mx.random.normal(K_pages.shape).astype(dtype)
    V_pages = mx.random.normal(V_pages.shape).astype(dtype)
    q = mx.random.normal((B, 1, H, D)).astype(dtype)
    got = paged_decode_attention(q, K_pages, V_pages, block_table, lengths, backend=backend)
    ref = reference_paged_decode_attention(q, K_pages, V_pages, block_table, lengths)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_specialized_paged_decode_attention_rejects_wrong_d():
    K64, V64, block_table64 = allocate_paged_kv_cache(1, 8, 2, 64, 4, mx.float16)
    K128, V128, block_table128 = allocate_paged_kv_cache(1, 8, 2, 128, 4, mx.float16)
    q64 = mx.random.normal((1, 1, 2, 64)).astype(mx.float16)
    q128 = mx.random.normal((1, 1, 2, 128)).astype(mx.float16)
    with pytest.raises(ValueError, match="requires D == 64"):
        paged_decode_attention(q128, K128, V128, block_table128, 4, backend="metal_d64")
    with pytest.raises(ValueError, match="requires D == 128"):
        paged_decode_attention(q64, K64, V64, block_table64, 4, backend="metal_d128")
