#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a quantized checkpoint package JSON.")
    parser.add_argument("package_json", type=str, help="Path to the quantized package JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print full tensor metadata.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pkg_path = Path(args.package_json)
    if not pkg_path.exists():
        print(f"Error: package file not found: {pkg_path}", file=sys.stderr)
        return 1

    from models.quantized_package_io import QuantizedCheckpointPackage

    try:
        package = QuantizedCheckpointPackage.load_json(str(pkg_path))
    except Exception as exc:
        print(f"Error: failed to load package: {exc}", file=sys.stderr)
        return 1

    try:
        package.validate(allow_partial=True)
    except Exception as exc:
        print(f"Warning: package validation issue: {exc}", file=sys.stderr)

    summary = package.summary()
    print(f"Format version: {summary['format_version']}")
    print(f"Model type:     {summary['model_type']}")
    print(f"Num layers:     {summary['num_layers']}")
    print(f"Tensor count:   {summary['tensor_count']}")
    print(f"Bits:           {summary.get('bits', '?')}")
    print(f"Group size:     {summary.get('group_size', '?')}")
    print(f"Config keys:    {', '.join(summary.get('config_keys', []))}")
    print(f"Global tensors: {', '.join(summary.get('global_tensors', [])) or '(none)'}")

    if args.verbose:
        for layer_key, tensors in summary.get("per_layer", {}).items():
            print(f"\n  Layer {layer_key}:")
            for name, shape in tensors.items():
                print(f"    {name}: {shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
