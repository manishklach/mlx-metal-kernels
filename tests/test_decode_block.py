import mlx.core as mx
import pytest

from ops.decode_block_ops import decode_block_from_qkv, reference_decode_block_from_qkv


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


def _make_qkv(B, H, D, dtype, layout):
    if layout == "packed":
        return mx.random.normal((B, 1, 3 * H * D)).astype(dtype)
    return mx.random.normal((B, 1, 3, H, D)).astype(dtype)


@pytest.mark.parametrize(("B", "MAX_S", "H", "D", "T", "dtype"), [(1, 8, 2, 16, 4, mx.float16), (2, 16, 4, 32, 8, mx.float16), (1, 8, 2, 16, 4, mx.bfloat16)])
@pytest.mark.parametrize("layout", ["packed", "explicit"])
def test_decode_block_from_qkv(B, MAX_S, H, D, T, dtype, layout):
    mx.random.seed(91)
    K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    ref_K = K_cache
    ref_V = V_cache
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    atol, rtol = _tol(dtype)

    for pos in range(T):
        qkv = _make_qkv(B, H, D, dtype, layout)
        got_out, K_cache, V_cache = decode_block_from_qkv(
            qkv, K_cache, V_cache, cos, sin, pos, H=H, D=D, backend="metal"
        )
        ref_out, ref_K, ref_V = reference_decode_block_from_qkv(
            qkv, ref_K, ref_V, cos, sin, pos, H=H, D=D
        )
        mx.eval(got_out, K_cache, V_cache, ref_out, ref_K, ref_V)
        assert mx.allclose(got_out, ref_out, atol=atol, rtol=rtol).item()
        assert mx.allclose(K_cache, ref_K, atol=atol, rtol=rtol).item()
        assert mx.allclose(V_cache, ref_V, atol=atol, rtol=rtol).item()
