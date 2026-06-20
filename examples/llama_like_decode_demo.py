from __future__ import annotations

import mlx.core as mx

from models import KernelBackendConfig, LlamaLikeKernelAdapter, tiny_debug_config


def main():
    mx.random.seed(301)
    config = tiny_debug_config()
    adapter = LlamaLikeKernelAdapter(
        config,
        KernelBackendConfig(matvec_backend="metal_parallel", attention_backend="metal", use_autotune=False),
        cache_layout="contiguous",
    )
    states = adapter.init_cache(1, dtype=mx.float16)
    weights = adapter.make_demo_quantized_weights(bits=4, group_size=32)
    cos, sin = adapter.build_rope_tables(dtype=mx.float32)

    x = mx.random.normal((1, 1, config.hidden_size)).astype(mx.float16)
    state = states[0]
    for pos in range(3):
        y, state = adapter.decode_layer_quantized_from_fused_qkv(
            x,
            state,
            cos,
            sin,
            pos,
            attn_norm_weight=weights["attn_norm_weight"].astype(mx.float16),
            ffn_norm_weight=weights["ffn_norm_weight"].astype(mx.float16),
            qkv_w=weights["qkv_w"],
            qkv_scales=weights["qkv_scales"],
            out_w=weights["out_w"],
            out_scales=weights["out_scales"],
            gate_w=weights["gate_w"],
            gate_scales=weights["gate_scales"],
            up_w=weights["up_w"],
            up_scales=weights["up_scales"],
            down_w=weights["down_w"],
            down_scales=weights["down_scales"],
            bits=4,
            group_size=32,
        )
        mx.eval(y)
        print(f"step={pos} output_shape={y.shape} matvec_backend={adapter.choose_backend('q4_matvec_decode', {'B': 1, 'K': config.hidden_size, 'N': config.hidden_size, 'group_size': 32}, 'float16')}")
        x = y

    print("This demo uses random toy weights and is not a real checkpoint loader.")


if __name__ == "__main__":
    main()
