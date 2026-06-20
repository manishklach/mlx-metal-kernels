#!/usr/bin/env python3
"""
CLI for converting a local checkpoint layout into a repo-native quantized package.

Two modes:
  1. Synthetic demo -- creates tiny random weights, converts to q4/q8, writes package JSON.
  2. Manifest-based -- loads a JSON manifest, validates shapes, and either dry-runs
     or errors if tensor data is needed but not available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a checkpoint layout to a quantized package.",
    )
    parser.add_argument("--manifest", type=str, default=None, help="Path to a JSON checkpoint manifest.")
    parser.add_argument("--output", type=str, default=None, help="Output path for the package JSON.")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits.")
    parser.add_argument("--group-size", type=int, default=32, help="Quantization group size.")
    parser.add_argument("--layers", type=str, default="all", help='Layer indices, e.g. "0,1,2" or "all".')
    parser.add_argument("--model-family", type=str, default="llama", help='Model family, e.g. "llama".')
    parser.add_argument("--config-json", type=str, default=None, help="Optional LlamaLikeConfig JSON.")
    parser.add_argument("--synthetic-demo", action="store_true", help="Create tiny synthetic config/tensors and convert them.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial layer conversion.")
    parser.add_argument("--no-fuse-qkv", action="store_true", dest="no_fuse_qkv", help="Skip QKV fusion.")
    parser.add_argument("--save-tensor-data", action="store_true", help="Not yet implemented.")
    parser.add_argument("--with-zeros", action="store_true", help="Materialize zero-point metadata.")
    parser.add_argument("--symmetric", action="store_true", default=True, help="Use symmetric quantization.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and plan without converting.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed conversion info.")
    return parser


def _parse_layer_indices(layers_str: str, max_layers: int) -> list[int]:
    if layers_str == "all":
        return list(range(max_layers))
    indices: list[int] = []
    for part in layers_str.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= max_layers:
            raise ValueError(f"layer index {idx} out of range [0, {max_layers})")
        indices.append(idx)
    return sorted(set(indices))


def _load_config_from_json(path: str):
    from models.llama_config import LlamaLikeConfig

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LlamaLikeConfig.from_dict(data)


def _build_synthetic_demo_config():
    from models.llama_config import tiny_gqa_debug_config

    return tiny_gqa_debug_config()


def _build_synthetic_tensors(config):
    import numpy as np

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
    return tensors


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.synthetic_demo and not args.manifest:
        parser.print_help()
        print("\nError: either --synthetic-demo or --manifest is required.", file=sys.stderr)
        return 1

    if args.save_tensor_data:
        print("Error: save-tensor-data is not yet implemented.", file=sys.stderr)
        return 1

    from models.checkpoint_adapter import CheckpointAdapter, CheckpointAdapterConfig, adapter_from_in_memory_tensors
    from models.checkpoint_converter import CheckpointConverter, CheckpointConverterConfig
    from models.checkpoint_manifest import CheckpointManifest
    from models.tensor_store import ManifestTensorStore

    if args.synthetic_demo:
        config = _build_synthetic_demo_config()
        if args.config_json:
            print("Warning: --config-json ignored in synthetic-demo mode.", file=sys.stderr)
        tensors = _build_synthetic_tensors(config)
        adapter = adapter_from_in_memory_tensors(
            config,
            tensors,
            adapter_config=CheckpointAdapterConfig(fuse_qkv=not args.no_fuse_qkv),
        )
        output_path = args.output or "generated_quantized_package.json"
        model_type = "llama_like_gqa_demo"
    elif args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: manifest file not found: {manifest_path}", file=sys.stderr)
            return 1
        manifest = CheckpointManifest.load_json(manifest_path)
        if args.config_json:
            config = _load_config_from_json(args.config_json)
        else:
            from models.llama_config import LlamaLikeConfig
            config = LlamaLikeConfig(
                hidden_size=4096,
                intermediate_size=11008,
                num_attention_heads=32,
                num_key_value_heads=32,
                head_dim=128,
                num_hidden_layers=32,
                max_position_embeddings=4096,
            ).validate()
        tensor_store = ManifestTensorStore(manifest)
        adapter = CheckpointAdapter(
            config,
            tensor_store,
            adapter_config=CheckpointAdapterConfig(fuse_qkv=not args.no_fuse_qkv),
        )
        output_path = args.output or "conversion_plan.json"
        model_type = manifest.model_type

    max_layers = adapter.config.num_hidden_layers
    try:
        layer_indices = _parse_layer_indices(args.layers, max_layers)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cc_config = CheckpointConverterConfig(
        bits=args.bits,
        group_size=args.group_size,
        layers=layer_indices,
        fuse_qkv=not args.no_fuse_qkv,
        save_tensor_data=args.save_tensor_data,
        allow_partial=args.allow_partial,
        symmetric=args.symmetric,
        with_zeros=args.with_zeros,
    )
    converter = CheckpointConverter(adapter, cc_config)

    print(f"Model type:    {model_type}")
    print(f"Layers:        {len(layer_indices)} selected ({layer_indices[0]}-{layer_indices[-1]})")
    print(f"Bits:          {args.bits}")
    print(f"Group size:    {args.group_size}")
    print(f"Fuse QKV:      {not args.no_fuse_qkv}")
    print(f"Dry run:       {args.dry_run}")
    print(f"Output:        {output_path}")

    if args.dry_run or args.manifest:
        if args.synthetic_demo:
            print("Synthetic demo mode does not support dry-run.")
        else:
            print(f"\nConversion plan (manifest-only, dry-run):")
            desc = adapter.describe()
            print(f"  tensors:        {desc.get('tensor_count', '?')}")
            print(f"  hidden_size:    {desc.get('hidden_size', '?')}")
            print(f"  num_heads:      {desc.get('heads', '?')}")
            print(f"  num_kv_heads:   {desc.get('kv_heads', '?')}")
            print(f"  layers:         {desc.get('layers', '?')}")
            print(f"\nDry-run plan written to: {output_path}")
            converter.convert(output_path=output_path)
            return 0

    try:
        package, report = converter.convert(output_path=output_path)
    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    if report.ok:
        print(f"Conversion OK.")
        print(f"  layers converted: {len(report.layers_converted)}")
        print(f"  tensor count:     {report.tensor_count}")
        print(f"  output:           {report.output_path}")
        if report.warnings:
            for w in report.warnings:
                print(f"  warning: {w}")
        if args.verbose and package is not None:
            print(f"\nPackage summary:")
            for k, v in package.summary().items():
                print(f"  {k}: {v}")
    else:
        print(f"Conversion FAILED.", file=sys.stderr)
        for err in report.errors:
            print(f"  error: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
