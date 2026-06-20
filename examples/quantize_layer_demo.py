from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for this demo") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CheckpointAdapter, CheckpointQuantizer, InMemoryTensorStore, QuantizationConfig, tiny_gqa_debug_config


def _synthetic_layer_tensors(config, layer_idx: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    stem = f"model.layers.{layer_idx}"
    return {
        f"{stem}.self_attn.q_proj.weight": rng.normal(size=(config.q_output_dim(), config.hidden_size)).astype(np.float16),
        f"{stem}.self_attn.k_proj.weight": rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16),
        f"{stem}.self_attn.v_proj.weight": rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16),
        f"{stem}.self_attn.o_proj.weight": rng.normal(size=(config.hidden_size, config.q_output_dim())).astype(np.float16),
        f"{stem}.mlp.gate_proj.weight": rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16),
        f"{stem}.mlp.up_proj.weight": rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16),
        f"{stem}.mlp.down_proj.weight": rng.normal(size=(config.hidden_size, config.intermediate_size)).astype(np.float16),
        f"{stem}.input_layernorm.weight": rng.normal(size=(config.hidden_size,)).astype(np.float16),
        f"{stem}.post_attention_layernorm.weight": rng.normal(size=(config.hidden_size,)).astype(np.float16),
    }


def main():
    config = tiny_gqa_debug_config()
    tensors = _synthetic_layer_tensors(config, seed=42)
    adapter = CheckpointAdapter(config, InMemoryTensorStore(tensors))
    quantizer = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32))
    package = quantizer.quantize_layer(0)
    report = quantizer.report()

    print("Synthetic quantized layer package")
    print("This is a correctness-first packaging demo, not a production checkpoint quantizer.")
    print("qkv packed shape:", package.qkv.shapes()["weight"])
    print("o_proj packed shape:", package.o_proj.shapes()["weight"])
    print("gate_proj packed shape:", package.gate_proj.shapes()["weight"])
    print("up_proj packed shape:", package.up_proj.shapes()["weight"])
    print("down_proj packed shape:", package.down_proj.shapes()["weight"])
    print("qkv scales shape:", package.qkv.shapes()["scales"])
    print("metrics keys:", sorted(report.metrics))
    print("qkv relative_rmse:", round(report.metrics["model.layers.0.self_attn.qkv_proj.fused_weight"]["relative_rmse"], 6))
    kernel_weights = package.to_kernel_weights(config)
    print("kernel qkv_w shape:", tuple(int(dim) for dim in kernel_weights.qkv_w.shape))


if __name__ == "__main__":
    main()
