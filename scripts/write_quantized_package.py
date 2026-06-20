#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write quantized package tensor data to disk."
    )
    parser.add_argument(
        "package_json",
        type=str,
        help="Path to the quantized package JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to parent of package_json.",
    )
    parser.add_argument(
        "--tensor-subdir",
        type=str,
        default="tensors",
        help="Subdirectory for tensor files within the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count tensors without writing files.",
    )
    parser.add_argument(
        "--checksum-algorithm",
        type=str,
        default="sha256",
        choices=("sha256", "sha1", "md5"),
        help="Hash algorithm for tensor file checksums.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic tensor data matching the package metadata. "
        "No checkpoint adapter needed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic tensor generation.",
    )
    return parser


def _build_synthetic_layer_packages(package, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    from models.quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage

    layer_packages: list[QuantizedLlamaLayerPackage] = []
    for layer_meta in package.layers:
        linears = {}
        norm_weights = {}
        for key, tensor_meta in layer_meta.tensors.items():
            if tensor_meta.role in ("norm", "final_norm"):
                norm_weights[key] = rng.normal(
                    size=tensor_meta.original_shape
                ).astype(np.float16)
            else:
                linears[key] = _build_synthetic_linear(rng, tensor_meta)

        layer_packages.append(
            QuantizedLlamaLayerPackage(
                layer_idx=layer_meta.layer_idx,
                input_layernorm_weight=norm_weights.get(
                    "input_layernorm",
                    rng.normal(
                        size=package.config.get("hidden_size", 64)
                    ).astype(np.float16),
                ),
                post_attention_layernorm_weight=norm_weights.get(
                    "post_attention_layernorm",
                    rng.normal(
                        size=package.config.get("hidden_size", 64)
                    ).astype(np.float16),
                ),
                qkv=linears.get(
                    "qkv",
                    QuantizedLinearPackage(
                        name="qkv",
                        weight=rng.normal(size=(128, 32)).astype(np.float16),
                        scales=rng.normal(size=(128, 2)).astype(np.float16),
                        zeros=None,
                        bits=4,
                        group_size=32,
                        original_shape=(128, 64),
                    ),
                ),
                o_proj=linears.get(
                    "o_proj",
                    QuantizedLinearPackage(
                        name="o_proj",
                        weight=rng.normal(size=(64, 64)).astype(np.float16),
                        scales=rng.normal(size=(64, 4)).astype(np.float16),
                        zeros=None,
                        bits=4,
                        group_size=32,
                        original_shape=(64, 128),
                    ),
                ),
                gate_proj=linears.get(
                    "gate_proj",
                    QuantizedLinearPackage(
                        name="gate_proj",
                        weight=rng.normal(size=(128, 32)).astype(np.float16),
                        scales=rng.normal(size=(128, 2)).astype(np.float16),
                        zeros=None,
                        bits=4,
                        group_size=32,
                        original_shape=(128, 64),
                    ),
                ),
                up_proj=linears.get(
                    "up_proj",
                    QuantizedLinearPackage(
                        name="up_proj",
                        weight=rng.normal(size=(128, 32)).astype(np.float16),
                        scales=rng.normal(size=(128, 2)).astype(np.float16),
                        zeros=None,
                        bits=4,
                        group_size=32,
                        original_shape=(128, 64),
                    ),
                ),
                down_proj=linears.get(
                    "down_proj",
                    QuantizedLinearPackage(
                        name="down_proj",
                        weight=rng.normal(size=(64, 64)).astype(np.float16),
                        scales=rng.normal(size=(64, 4)).astype(np.float16),
                        zeros=None,
                        bits=4,
                        group_size=32,
                        original_shape=(64, 128),
                    ),
                ),
            )
        )
    return layer_packages


def _build_synthetic_linear(rng, tensor_meta):
    import numpy as np
    from models.quantized_layer_package import QuantizedLinearPackage

    return QuantizedLinearPackage(
        name=tensor_meta.name,
        weight=rng.normal(size=tensor_meta.packed_shape).astype(np.float16),
        scales=rng.normal(size=tensor_meta.scales_shape).astype(np.float16),
        zeros=rng.normal(size=tensor_meta.zeros_shape).astype(np.float16)
        if tensor_meta.zeros_shape
        else None,
        bits=tensor_meta.bits,
        group_size=tensor_meta.group_size,
        original_shape=tensor_meta.original_shape,
    )


def _build_synthetic_global_tensors(package, rng):
    import numpy as np
    global_tensors = {}
    for name, tensor_meta in package.global_tensors.items():
        global_tensors[name] = rng.normal(
            size=tensor_meta.original_shape
        ).astype(np.float16)
    return global_tensors


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    pkg_path = Path(args.package_json)
    if not pkg_path.exists():
        print(f"Error: package file not found: {pkg_path}", file=sys.stderr)
        return 1

    from models.quantized_package_io import QuantizedCheckpointPackage
    from models.quantized_package_writer import (
        PackageWriterConfig,
        QuantizedPackageWriter,
    )

    try:
        package = QuantizedCheckpointPackage.load_json(str(pkg_path))
        package.validate(allow_partial=True)
    except Exception as exc:
        print(f"Error: failed to load package: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else pkg_path.parent
    writer = QuantizedPackageWriter(
        PackageWriterConfig(
            tensor_subdir=args.tensor_subdir,
            checksum_algorithm=args.checksum_algorithm,
        )
    )

    summary = package.summary()
    print(f"Package: {pkg_path}")
    print(f"Layers:  {summary['num_layers']}")
    print(f"Tensors: {summary['tensor_count']}")

    if args.dry_run:
        print("\nDry run: no files written.")
        return 0

    if args.synthetic:
        print(f"Generating synthetic tensors (seed={args.seed})...")
        layer_packages = _build_synthetic_layer_packages(package, args.seed)
        import numpy as np
        global_tensors = _build_synthetic_global_tensors(
            package, np.random.default_rng(args.seed + 1)
        )
    else:
        print(
            "Error: --synthetic is required (no checkpoint adapter integration in this CLI yet).",
            file=sys.stderr,
        )
        return 1

    report = writer.write_tensors(
        package, layer_packages, output_dir, global_tensors=global_tensors
    )

    if report.errors:
        for err in report.errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"\nWrote {report.tensor_count} tensor files ({report.total_bytes} bytes)")
    print(f"Package JSON updated: {report.package_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
