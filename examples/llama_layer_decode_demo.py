from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from models.llama_config import tiny_gqa_debug_config
from ops.llama_layer_ops import create_random_quantized_llama_layer_weights, init_llama_layer_cache, llama_layer_decode_loop


def main():
    config = tiny_gqa_debug_config()
    weights = create_random_quantized_llama_layer_weights(config, bits=4, dtype=mx.float16, seed=0)
    cache = init_llama_layer_cache(config, 1, config.max_position_embeddings, dtype=mx.float16)
    cos = mx.random.normal((config.max_position_embeddings + 4, config.head_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((config.max_position_embeddings + 4, config.head_dim // 2)).astype(mx.float32)
    inputs = mx.random.normal((1, 4, config.hidden_size)).astype(mx.float16)
    outputs, final_cache = llama_layer_decode_loop(inputs, weights, cache, cos, sin, config, backend_preset="fused_experimental")
    mx.eval(outputs, *final_cache)
    print(f"config={config.to_dict()}")
    print(f"qkv_shape={weights.shapes()['qkv_w']}")
    print(f"cache_shape={final_cache[0].shape}")
    print("backend_preset=fused_experimental")
    print(f"output_shape={outputs.shape}")
    print(f"final_cache_shape={final_cache[0].shape}")
    print("This is a synthetic single-layer decode demo, not a full checkpoint runtime.")


if __name__ == "__main__":
    main()
