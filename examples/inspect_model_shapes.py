from __future__ import annotations

import mlx.core as mx

from models import LlamaLikeKernelAdapter, fused_qkv_spec, llama_8b_like, llama_layer_weight_specs, tiny_debug_config


def _print_config(name, config):
    adapter = LlamaLikeKernelAdapter(config)
    state = adapter.init_cache(1, dtype=mx.float16)[0]
    fused = fused_qkv_spec(config)
    first = llama_layer_weight_specs(config)[0]
    print(f"== {name} ==")
    print(f"hidden_size={config.hidden_size}")
    print(f"heads={config.num_attention_heads}")
    print(f"head_dim={config.head_dim}")
    print(f"q_output_dim={config.q_output_dim()}")
    print(f"kv_output_dim={config.kv_output_dim()}")
    print(f"qkv_output_dim={config.qkv_output_dim()}")
    print(f"mlp_intermediate={config.intermediate_size}")
    print(f"fused_qkv_shape={fused.expected_shape()}")
    print(f"q_proj_shape={first.q_proj.expected_shape()}")
    print(f"o_proj_shape={first.o_proj.expected_shape()}")
    print(f"gate_proj_shape={first.gate_proj.expected_shape()}")
    print(f"cache_shape={state.K_cache.shape if state.K_cache is not None else state.K_pages.shape}")
    print()


def main():
    _print_config("tiny_debug", tiny_debug_config())
    _print_config("llama_8b_like", llama_8b_like())


if __name__ == "__main__":
    main()
