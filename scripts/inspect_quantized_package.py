#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a quantized checkpoint package JSON.")
    parser.add_argument("package_json", type=str, help="Path to the quantized package JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print full tensor metadata.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional local tokenizer path.")
    parser.add_argument(
        "--tokenizer-kind",
        type=str,
        default="auto",
        choices=("auto", "hf-tokenizers", "sentencepiece", "char", "whitespace"),
        help="Tokenizer loader to use when --validate-alignment is requested.",
    )
    parser.add_argument("--validate-alignment", action="store_true", help="Run structured alignment validation.")
    parser.add_argument("--bits", type=int, default=None, help="Optional expected bits override for alignment validation.")
    parser.add_argument("--group-size", type=int, default=None, help="Optional expected group_size override for alignment validation.")
    parser.add_argument("--check-tensor-files", action="store_true", help="Check tensor file existence and integrity.")
    parser.add_argument("--check-checksums", action="store_true", help="Also validate checksums of tensor files.")
    parser.add_argument("--package-root", type=str, default=None, help="Root directory for resolving relative tensor paths.")
    return parser


def _load_tokenizer(args):
    from models.tokenizer_adapters import OptionalDependencyError, TokenizerAdapterFactory

    if args.tokenizer is None and args.tokenizer_kind in ("char", "whitespace"):
        return TokenizerAdapterFactory.from_file(None, kind=args.tokenizer_kind)
    if args.tokenizer is None:
        return None
    kind = None if args.tokenizer_kind == "auto" else args.tokenizer_kind
    try:
        return TokenizerAdapterFactory.from_file(args.tokenizer, kind=kind)
    except OptionalDependencyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed to load tokenizer: {exc}") from exc


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

    if args.check_tensor_files:
        from models.smoke_test import inspect_package_executability

        pkg_root = Path(args.package_root) if args.package_root else pkg_path.parent
        print(f"\nTensor file check (root: {pkg_root}):")
        issues = package.validate_tensor_files(
            pkg_root, check_checksums=args.check_checksums
        )
        if issues:
            for iss in issues:
                print(f"  Issue: {iss}")
        else:
            print("  All tensor files present and valid.")
        executability = inspect_package_executability(
            package, pkg_path, check_checksums=args.check_checksums
        )
        print(f"  Executable: {executability['executable']}")
        if executability["missing_tensor_data"]:
            print(f"  Missing: {len(executability['missing_tensor_data'])} entries")
        if executability["checksum_mismatches"]:
            print(
                f"  Checksum mismatches: {len(executability['checksum_mismatches'])}"
            )
        for cm in executability["checksum_mismatches"]:
            print(f"    {cm}")

    if args.validate_alignment:
        from models.alignment import (
            validate_quantization_alignment,
            validate_tokenizer_against_package,
        )
        from models.tokenizer_adapters import OptionalDependencyError

        tokenizer = None
        if args.tokenizer is not None or args.tokenizer_kind in ("char", "whitespace"):
            try:
                tokenizer = _load_tokenizer(args)
            except OptionalDependencyError as exc:
                print(f"Error: optional tokenizer dependency is missing: {exc}", file=sys.stderr)
                return 1
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}", file=sys.stderr)
                return 1

        reports = [validate_quantization_alignment(package, bits=args.bits, group_size=args.group_size)]
        if tokenizer is not None:
            reports.append(validate_tokenizer_against_package(tokenizer, package))

        issues = []
        for report in reports:
            issues.extend(report.issues)
        ok = not any(issue.severity == "error" for issue in issues)
        merged_summary = {
            "package": str(pkg_path),
            "tokenizer": type(tokenizer).__name__ if tokenizer is not None else None,
            "bits": summary.get("bits"),
            "group_size": summary.get("group_size"),
        }
        from models.alignment import AlignmentReport

        report = AlignmentReport(ok=ok, issues=issues, summary=merged_summary)
        print("\nAlignment report:")
        print(report.pretty_print())
        if not report.ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
