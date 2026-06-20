#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo tokenizer adapters.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer file.")
    parser.add_argument("--kind", type=str, default=None, help="Tokenizer kind: auto|hf-tokenizers|sentencepiece|char|whitespace")
    parser.add_argument("--text", type=str, default="Hello world, this is a tokenizer demo!", help="Text to encode/decode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    from models.tokenizer_adapters import TokenizerAdapterFactory, describe_tokenizer

    try:
        tokenizer = TokenizerAdapterFactory.from_file(args.tokenizer, kind=args.kind)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Optional dependency missing: {exc}", file=sys.stderr)
        return 1

    info = describe_tokenizer(tokenizer, source=args.tokenizer)
    print(f"Tokenizer:      {info.kind}")
    print(f"Source:         {info.source or '(built-in)'}")
    print(f"Vocab size:     {info.vocab_size}")
    print(f"BOS token id:   {info.bos_token_id}")
    print(f"EOS token id:   {info.eos_token_id}")
    print(f"PAD token id:   {info.pad_token_id}")
    print(f"UNK token id:   {info.unk_token_id}")

    text = args.text
    print(f"\nInput text: {text!r}")
    encoded = tokenizer.encode(text)
    print(f"Encoded ids:   {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text:  {decoded!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
