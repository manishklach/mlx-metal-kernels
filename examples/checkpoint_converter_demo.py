#!/usr/bin/env python3
"""
Demo of the checkpoint converter scaffold.

Creates a tiny GQA config and synthetic in-memory tensors, converts 2 layers
to q4, writes package JSON, loads it back, and prints a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    import numpy as np

    from models.checkpoint_adapter import CheckpointAdapter, CheckpointAdapterConfig, adapter_from_in_memory_tensors
    from models.checkpoint_converter import CheckpointConverter, CheckpointConverterConfig
    from models.llama_config import tiny_gqa_debug_config
    from models.quantized_package_io import QuantizedCheckpointPackage

    config = tiny_gqa_debug_config()
    rng = np.random.default_rng(42)
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        stem = f"model.layers.{layer_idx}"
        tensors[f"{stem}.self_attn.q_proj.weight"] = rng.normal(size=(config.q_output_dim(), config.hidden_size)).astype(np.float16)
        tensors[f"{stem}.self_attn.k_proj.weight"] = rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16)
        tensors[f"{stem}.self_attn.v_proj.weight"] = rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16)
        tensors[f"{stem}.self_attn.o_proj.weight"] = rng.normal(size=(config.hidden_size, config.q_output_dim())).astype(np.float16)
        tensors[f"{stem}.mlp.gate_proj.weight"] = rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16)
        tensors[f"{stem}.mlp.up_proj.weight"] = rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16)
        tensors[f"{stem}.mlp.down_proj.weight"] = rng.normal(size=(config.hidden_size, config.intermediate_size)).astype(np.float16)
        tensors[f"{stem}.input_layernorm.weight"] = np.ones((config.hidden_size,), dtype=np.float16)
        tensors[f"{stem}.post_attention_layernorm.weight"] = np.ones((config.hidden_size,), dtype=np.float16)

    adapter = adapter_from_in_memory_tensors(
        config,
        tensors,
        adapter_config=CheckpointAdapterConfig(fuse_qkv=True),
    )
    cc_config = CheckpointConverterConfig(bits=4, group_size=32, layers=[0, 1])
    converter = CheckpointConverter(adapter, cc_config)
    package, report = converter.convert()
    if not report.ok:
        print(f"Conversion failed: {report.errors}", file=sys.stderr)
        return 1

    out_dir = Path(__file__).resolve().parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quantized_package.json"
    converter.save_package(package, out_path)
    print(f"Package saved to: {out_path}")

    loaded = QuantizedCheckpointPackage.load_json(str(out_path))
    loaded.validate()
    print(f"Loaded package: {loaded.model_type}, {loaded.num_layers()} layers, {loaded.tensor_count()} tensors")
    print(f"Summary: {loaded.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
