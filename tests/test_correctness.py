import math

import mlx.core as mx
import pytest

from ops.attention_ops import (
    decode_attention,
    fast_attention,
    fast_attention_with_split,
    reference_attention,
)


def _tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


def _check(B, S, H, D, dtype, causal, backend):
    mx.random.seed(0)
    Q = mx.random.normal((B, S, H, D)).astype(dtype)
    K = mx.random.normal((B, S, H, D)).astype(dtype)
    V = mx.random.normal((B, S, H, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)

    got = fast_attention(Q, K, V, scale=scale, causal=causal, backend=backend)
    ref = reference_attention(Q, K, V, scale=scale, causal=causal)
    mx.eval(got, ref)

    # The baseline kernel accumulates fp32 and writes fp16/bf16, so tolerances
    # need to account for low precision output.
    atol, rtol = _tolerances(dtype)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


@pytest.mark.parametrize("backend", ["baseline", "row_parallel", "tiled_kv", "reference"])
@pytest.mark.parametrize(
    ("B", "S", "H", "D", "dtype", "causal"),
    [
        (1, 8, 2, 32, mx.float16, False),
        (1, 8, 2, 32, mx.float16, True),
        (1, 16, 2, 64, mx.float16, False),
        (1, 16, 2, 64, mx.bfloat16, True),
        (1, 32, 4, 64, mx.float16, False),
        (2, 16, 4, 128, mx.bfloat16, False),
    ],
)
def test_attention_matches_reference(B, S, H, D, dtype, causal, backend):
    _check(B=B, S=S, H=H, D=D, dtype=dtype, causal=causal, backend=backend)


def test_decode_attention_matches_reference():
    mx.random.seed(7)
    B, Sq, Sk, H, D = 2, 1, 16, 4, 64
    q = mx.random.normal((B, Sq, H, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, Sk, H, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, Sk, H, D)).astype(mx.float16)
    got = decode_attention(q, K_cache, V_cache, backend="auto")
    ref = reference_attention(q, K_cache, V_cache)
    mx.eval(got, ref)
    atol, rtol = _tolerances(mx.float16)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()


def test_decode_attention_rejects_non_singleton_query():
    mx.random.seed(8)
    q = mx.random.normal((1, 2, 2, 32)).astype(mx.float16)
    K_cache = mx.random.normal((1, 8, 2, 32)).astype(mx.float16)
    V_cache = mx.random.normal((1, 8, 2, 32)).astype(mx.float16)
    with pytest.raises(ValueError, match="q.shape\\[1\\] == 1"):
        decode_attention(q, K_cache, V_cache)


def test_split_reference_matches_reference():
    mx.random.seed(1)
    B, S, H, D = 1, 32, 2, 32
    Q = mx.random.normal((B, S, H, D)).astype(mx.float16)
    K = mx.random.normal((B, S, H, D)).astype(mx.float16)
    V = mx.random.normal((B, S, H, D)).astype(mx.float16)
    got = fast_attention_with_split(Q, K, V, num_splits=4, causal=True)
    ref = reference_attention(Q, K, V, causal=True)
    mx.eval(got, ref)
    atol, rtol = _tolerances(mx.float16)
    assert mx.allclose(got, ref, atol=atol, rtol=rtol).item()
