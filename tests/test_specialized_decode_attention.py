import mlx.core as mx
import pytest

from ops.decode_ops import decode_attention, reference_decode_attention


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


@pytest.mark.parametrize(
    ("B", "MAX_S", "H", "D", "lengths", "dtype", "backend"),
    [
        (1, 16, 2, 64, 8, mx.float16, "metal_d64"),
        (2, 32, 4, 64, [12, 20], mx.float16, "metal_d64"),
        (1, 16, 2, 128, 8, mx.float16, "metal_d128"),
        (1, 16, 2, 64, 10, mx.bfloat16, "metal_d64"),
    ],
)
def test_specialized_decode_attention_matches_reference(B, MAX_S, H, D, lengths, dtype, backend):
    mx.random.seed(101)
    q = mx.random.normal((B, 1, H, D)).astype(dtype)
    K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
    got = decode_attention(q, K_cache, V_cache, lengths=lengths, backend=backend)
    ref = reference_decode_attention(q, K_cache, V_cache, lengths=lengths)
    mx.eval(got, ref)
    atol, rtol = _tol(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_specialized_decode_attention_rejects_wrong_d():
    q64 = mx.random.normal((1, 1, 2, 64)).astype(mx.float16)
    q128 = mx.random.normal((1, 1, 2, 128)).astype(mx.float16)
    K64 = mx.random.normal((1, 8, 2, 64)).astype(mx.float16)
    V64 = mx.random.normal((1, 8, 2, 64)).astype(mx.float16)
    K128 = mx.random.normal((1, 8, 2, 128)).astype(mx.float16)
    V128 = mx.random.normal((1, 8, 2, 128)).astype(mx.float16)
    with pytest.raises(ValueError, match="requires D == 64"):
        decode_attention(q128, K128, V128, lengths=4, backend="metal_d64")
    with pytest.raises(ValueError, match="requires D == 128"):
        decode_attention(q64, K64, V64, lengths=4, backend="metal_d128")
