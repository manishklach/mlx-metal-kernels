import mlx.core as mx
import pytest

from ops.kv_cache_ops import kv_cache_update, reference_kv_cache_update


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize("backend", ["reference", "metal"])
@pytest.mark.parametrize(
    ("B", "MAX_S", "H", "D", "dtype", "positions"),
    [
        (1, 8, 2, 16, mx.float16, 3),
        (2, 16, 4, 32, mx.float16, [2, 7]),
        (2, 16, 4, 64, mx.bfloat16, [0, 15]),
    ],
)
def test_kv_cache_update_matches_reference(B, MAX_S, H, D, dtype, positions, backend):
    mx.random.seed(10)
    K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    k_new = mx.random.normal((B, 1, H, D)).astype(dtype)
    v_new = mx.random.normal((B, 1, H, D)).astype(dtype)

    got_K, got_V = kv_cache_update(K_cache, V_cache, k_new, v_new, positions, backend=backend)
    ref_K, ref_V = reference_kv_cache_update(K_cache, V_cache, k_new, v_new, positions)
    mx.eval(got_K, got_V, ref_K, ref_V)

    atol, rtol = _tolerances(dtype)
    assert got_K.shape == K_cache.shape
    assert got_V.shape == V_cache.shape
    assert got_K.dtype == dtype
    assert got_V.dtype == dtype
    assert mx.allclose(got_K, ref_K, atol=atol, rtol=rtol).item()
    assert mx.allclose(got_V, ref_V, atol=atol, rtol=rtol).item()


def test_kv_cache_update_only_changes_selected_positions():
    mx.random.seed(11)
    B, MAX_S, H, D = 2, 16, 4, 32
    dtype = mx.float16
    positions = mx.array([2, 7], dtype=mx.int32)
    K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    k_new = mx.random.normal((B, H, D)).astype(dtype)
    v_new = mx.random.normal((B, H, D)).astype(dtype)

    got_K, got_V = kv_cache_update(K_cache, V_cache, k_new, v_new, positions, backend="reference")
    mx.eval(got_K, got_V)

    untouched = mx.arange(MAX_S).reshape(1, MAX_S, 1, 1) != positions.reshape(B, 1, 1, 1)
    assert mx.all(mx.where(untouched, got_K == K_cache, mx.ones_like(got_K, dtype=mx.bool_))).item()
    assert mx.all(mx.where(untouched, got_V == V_cache, mx.ones_like(got_V, dtype=mx.bool_))).item()
