import math

import mlx.core as mx

from ops.attention_ops import fast_attention, reference_attention


def main():
    B, S, H, D = 1, 32, 4, 64
    dtype = mx.float16
    mx.random.seed(42)
    Q = mx.random.normal((B, S, H, D)).astype(dtype)
    K = mx.random.normal((B, S, H, D)).astype(dtype)
    V = mx.random.normal((B, S, H, D)).astype(dtype)

    y = fast_attention(Q, K, V, scale=1.0 / math.sqrt(D), causal=True)
    ref = reference_attention(Q, K, V, causal=True)
    mx.eval(y, ref)
    print("output shape:", y.shape)
    print("max abs error:", mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item())


if __name__ == "__main__":
    main()
