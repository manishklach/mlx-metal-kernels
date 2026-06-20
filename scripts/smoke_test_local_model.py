#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import SmokeTestConfig, run_local_smoke_test


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a conservative local smoke test for package/tokenizer/runtime alignment.")
    parser.add_argument("--package", dest="package_path", type=str, default=None, help="Path to local quantized package JSON.")
    parser.add_argument("--tokenizer", dest="tokenizer_path", type=str, default=None, help="Path to local tokenizer file.")
    parser.add_argument(
        "--tokenizer-kind",
        type=str,
        default="auto",
        choices=("auto", "hf-tokenizers", "sentencepiece", "char", "whitespace"),
        help="Tokenizer kind for local loading.",
    )
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text for synthetic smoke mode.")
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Number of tokens to generate in synthetic smoke mode.")
    parser.add_argument(
        "--backend-preset",
        type=str,
        default="fused_experimental",
        choices=("reference", "metal", "tiled", "fused_experimental"),
        help="Tiny pipeline backend preset to use for synthetic fallback.",
    )
    parser.add_argument("--bits", type=int, default=None, help="Optional expected quantization bits.")
    parser.add_argument("--group-size", type=int, default=None, help="Optional expected quantization group size.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Run validation only without generation.")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Allow synthetic fallback generation or future tensor-data execution.")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--synthetic-fallback", action="store_true", help="Explicitly allow synthetic tiny-generation fallback.")
    parser.add_argument("--require-tensor-data", action="store_true", help="Fail if executable tensor data is missing.")
    parser.add_argument("--no-alignment", dest="validate_alignment", action="store_false", help="Skip alignment validation.")
    parser.set_defaults(validate_alignment=True)
    parser.add_argument("--use-prefill", dest="use_prefill", action="store_true", help="Use prefill in synthetic tiny-generation fallback.")
    parser.add_argument("--no-prefill", dest="use_prefill", action="store_false", help="Disable prefill in synthetic tiny-generation fallback.")
    parser.set_defaults(use_prefill=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic fallback generation.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")
    parser.add_argument("--verbose", action="store_true", help="Reserved for future expanded output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dry_run and not args.synthetic_fallback and not args.require_tensor_data:
        print(
            "Error: --no-dry-run requires either --synthetic-fallback or --require-tensor-data. "
            "Metadata-only packages cannot execute directly in this repo yet.",
            file=sys.stderr,
        )
        return 1

    config = SmokeTestConfig(
        package_path=args.package_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_kind=args.tokenizer_kind,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        backend_preset=args.backend_preset,
        bits=args.bits,
        group_size=args.group_size,
        dry_run=args.dry_run,
        synthetic_fallback=args.synthetic_fallback,
        require_tensor_data=args.require_tensor_data,
        validate_alignment=args.validate_alignment,
        use_prefill=args.use_prefill,
        seed=args.seed,
    )

    try:
        report = run_local_smoke_test(config)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.pretty_print())

    if not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
