#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    import numpy as np

    from models.checkpoint_adapter import CheckpointAdapterConfig, adapter_from_in_memory_tensors
    from models.checkpoint_converter import CheckpointConverter, CheckpointConverterConfig
    from models.llama_config import tiny_gqa_debug_config
    from models.quantized_package_io import QuantizedCheckpointPackage
    from models.quantized_package_writer import PackageWriterConfig, QuantizedPackageWriter

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
    metadata_path = out_dir / "quantized_package.json"
    converter.save_package(package, metadata_path)

    writer = QuantizedPackageWriter(PackageWriterConfig(tensor_subdir="tensors"))
    quantized_packages = converter.convert_layers([0, 1])
    write_report = writer.write_tensors(package, quantized_packages, out_dir)

    if write_report.errors:
        print(f"Writer errors: {write_report.errors}", file=sys.stderr)
        return 1

    print(f"Wrote {write_report.tensor_count} tensor files ({write_report.total_bytes} bytes)")
    print(f"Package JSON: {write_report.package_path}")

    loaded = QuantizedCheckpointPackage.load_json(str(metadata_path))
    loaded.validate(allow_partial=True)
    print(f"Loaded package: {loaded.model_type}, {loaded.num_layers()} layers, {loaded.tensor_count()} tensors")
    print(f"Has tensor data: {loaded.has_tensor_data()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
