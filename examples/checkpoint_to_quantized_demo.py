from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for this demo") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CheckpointAdapter, CheckpointQuantizer, InMemoryTensorStore, QuantizationConfig, tiny_debug_config


def _synthetic_checkpoint(config, seed: int = 7):
    rng = np.random.default_rng(seed)
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        stem = f"model.layers.{layer_idx}"
        tensors.update(
            {
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
        )
    return tensors


def main():
    config = tiny_debug_config()
    adapter = CheckpointAdapter(config, InMemoryTensorStore(_synthetic_checkpoint(config)))
    validation = adapter.validate()
    print("validation_ok:", validation.ok)
    quantizer = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32))
    packages = quantizer.quantize_layers([0])
    report = quantizer.report()
    print("quantized_layers:", len(packages))
    print("quantized_tensors:", len(report.quantized_tensors))
    print("skipped_tensors:", len(report.skipped_tensors))
    print("errors:", report.errors)
    print("This demo uses synthetic local tensors only and is not a production checkpoint converter.")


if __name__ == "__main__":
    main()
