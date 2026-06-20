from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for this demo") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import LlamaLikeConfig


def _load_stack_ops():
    path = ROOT / "ops" / "llama_stack_ops.py"
    spec = importlib.util.spec_from_file_location("llama_stack_ops_demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    stack_ops = _load_stack_ops()
    config = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=8,
        vocab_size=64,
        model_type="llama_stack_demo",
    ).validate()
    weights = stack_ops.create_random_quantized_llama_stack_weights(config, vocab_size=64, bits=4, seed=7)
    cache = stack_ops.init_llama_stack_cache(config, 1, config.max_position_embeddings)
    cos, sin = stack_ops._build_rope_tables_numpy(config, config.max_position_embeddings + 1)
    inputs = np.random.default_rng(8).normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    outputs, final_cache = stack_ops.llama_stack_decode_loop(inputs, weights, cache, cos, sin, config, backend_preset="fused_experimental", return_logits=True)
    print("config:", config.to_dict())
    print("num_layers:", weights.num_layers())
    print("cache_shapes:", final_cache.shapes())
    print("output_shape:", tuple(int(dim) for dim in outputs.shape))
    print("backend_preset:", "fused_experimental")


if __name__ == "__main__":
    main()
