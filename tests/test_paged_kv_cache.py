import mlx.core as mx
import pytest

from ops.paged_kv_ops import (
    allocate_paged_kv_cache,
    block_table_lookup,
    paged_kv_cache_update,
    reference_block_table_lookup,
    reference_paged_kv_cache_update,
)


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(("B", "MAX_S", "PAGE_SIZE", "positions"), [(1, 16, 4, 3), (2, 32, 8, [2, 17])])
@pytest.mark.parametrize("backend", ["reference", "metal"])
def test_block_table_lookup(B, MAX_S, PAGE_SIZE, positions, backend):
    _, _, block_table = allocate_paged_kv_cache(B, MAX_S, 2, 16, PAGE_SIZE, mx.float16)
    got_page, got_offset = block_table_lookup(block_table, positions, PAGE_SIZE, backend=backend)
    ref_page, ref_offset = reference_block_table_lookup(block_table, positions, PAGE_SIZE)
    mx.eval(got_page, got_offset, ref_page, ref_offset)
    assert mx.all(got_page == ref_page).item()
    assert mx.all(got_offset == ref_offset).item()


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "MAX_S", "PAGE_SIZE", "H", "D", "dtype", "positions"),
    [
        (1, 16, 4, 2, 16, mx.float16, 3),
        (2, 32, 8, 4, 32, mx.float16, [2, 17]),
        (2, 32, 4, 4, 64, mx.bfloat16, [0, 31]),
    ],
)
def test_paged_kv_cache_update(B, MAX_S, PAGE_SIZE, H, D, dtype, positions, backend):
    mx.random.seed(80)
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    K_pages = mx.random.normal(K_pages.shape).astype(dtype)
    V_pages = mx.random.normal(V_pages.shape).astype(dtype)
    k_new = mx.random.normal((B, 1, H, D)).astype(dtype)
    v_new = mx.random.normal((B, 1, H, D)).astype(dtype)
    got_K, got_V = paged_kv_cache_update(K_pages, V_pages, k_new, v_new, block_table, positions, backend=backend)
    ref_K, ref_V = reference_paged_kv_cache_update(K_pages, V_pages, k_new, v_new, block_table, positions)
    mx.eval(got_K, got_V, ref_K, ref_V)
    atol, rtol = _tol(dtype)
    assert got_K.shape == K_pages.shape and got_V.shape == V_pages.shape
    assert mx.allclose(got_K, ref_K, atol=atol, rtol=rtol).item()
    assert mx.allclose(got_V, ref_V, atol=atol, rtol=rtol).item()
