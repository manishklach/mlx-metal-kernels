from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import TinyGenerationPipeline, TinyGenerationPipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--backend-preset", choices=["reference", "metal", "tiled", "fused_experimental"], default="fused_experimental")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TinyGenerationPipelineConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_hidden_layers=args.num_layers,
        bits=args.bits,
        backend_preset=args.backend_preset,
    ).validate()
    pipeline = TinyGenerationPipeline(config=config)
    result = pipeline.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    print("This demo uses synthetic random weights. Output text is not meaningful language generation.")
    print("prompt:", result.prompt)
    print("prompt ids:", result.prompt_ids)
    print("generated ids:", result.generated_ids)
    print("all ids:", result.all_ids)
    print("decoded text:", result.text)
    print("metadata:", json.dumps(result.metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
