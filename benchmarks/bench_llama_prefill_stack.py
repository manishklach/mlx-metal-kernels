from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional path
    mx = None

import numpy as np

from models.generation import _optional_llama_prefill_ops, _optional_llama_stack_ops
from models.llama_config import LlamaLikeConfig


def _dtype_from_string(name: str):
    if mx is None:
        return np.float32
    return mx.float16 if name == "float16" else mx.bfloat16


def _make_inputs(B: int, S: int, hidden_size: int, dtype, seed: int):
    if mx is None:
        return np.random.default_rng(seed).normal(size=(B, S, hidden_size)).astype(np.float32)
    mx.random.seed(seed)
    return mx.random.normal((B, S, hidden_size)).astype(dtype)


def _make_rope(prefill_module, config: LlamaLikeConfig, dtype):
    if mx is None:
        return prefill_module._build_rope_tables_numpy(config, config.max_position_embeddings + 1)
    from models.llama_config import build_rope_tables

    return build_rope_tables(config, seq_len=config.max_position_embeddings + 1, dtype=mx.float32)


def _mean_ms(samples: list[float]) -> float:
    return statistics.fmean(samples)


def _time_fn(fn, iters: int) -> dict[str, float]:
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        if mx is not None:
            try:
                if isinstance(out, tuple):
                    mx.eval(*[item for item in out if hasattr(item, "shape")])
                elif hasattr(out, "shape"):
                    mx.eval(out)
            except Exception:  # noqa: BLE001
                pass
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "mean_ms": _mean_ms(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iters": len(samples),
    }


def _config(args):
    return LlamaLikeConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.MAX_S,
        vocab_size=args.vocab_size if args.with_lm_head else None,
    ).validate()


def _prefill_backend(prefill_module, name: str):
    mapping = {
        "reference": prefill_module.reference_prefill_backend_config,
        "metal": prefill_module.metal_prefill_backend_config,
        "tiled": prefill_module.tiled_prefill_backend_config,
        "fused_experimental": prefill_module.fused_experimental_prefill_backend_config,
    }
    return mapping[name]()


def _compare_decode_ingest(stack_ops, weights, inputs, config, args):
    cache = stack_ops["init_llama_stack_cache"](config, args.B, args.MAX_S, cache_layout=args.cache, dtype=_dtype_from_string(args.dtype) if mx is not None else None)
    cos, sin = (_optional_llama_prefill_ops()["module"]._build_rope_tables_numpy(config, args.MAX_S + 1) if mx is None else None, None)
    if mx is None:
        cos, sin = _optional_llama_prefill_ops()["module"]._build_rope_tables_numpy(config, args.MAX_S + 1)
    else:
        from models.llama_config import build_rope_tables

        cos, sin = build_rope_tables(config, seq_len=args.MAX_S + 1, dtype=mx.float32)

    def fn():
        return stack_ops["module"].llama_stack_decode_loop(
            inputs,
            weights,
            cache,
            cos,
            sin,
            config,
            backend_preset=args.backend_preset if args.backend_preset != "all" else "reference",
            cache_layout=args.cache,
            return_logits=args.with_lm_head,
        )

    return _time_fn(fn, args.iters)["mean_ms"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend-preset", choices=["reference", "metal", "tiled", "fused_experimental", "all"], default="all")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--with-lm-head", action="store_true")
    parser.add_argument("--compare-decode-ingest", action="store_true")
    args = parser.parse_args()

    if args.cache == "paged":
        raise NotImplementedError("Paged prefill is not implemented yet.")

    stack_ops = _optional_llama_stack_ops()
    prefill_ops = _optional_llama_prefill_ops()
    if stack_ops is None or prefill_ops is None:
        raise RuntimeError("The prefill benchmark requires the stack and prefill scaffolds.")
    prefill_module = prefill_ops["module"]
    config = _config(args)
    dtype = _dtype_from_string(args.dtype)
    weights = stack_ops["create_random_quantized_llama_stack_weights"](
        config,
        vocab_size=args.vocab_size,
        bits=args.bits,
        group_size=32,
        dtype=dtype if mx is not None else None,
        seed=args.seed,
        include_embedding=True,
        include_lm_head=args.with_lm_head,
    )
    inputs = _make_inputs(args.B, args.S, args.hidden_size, dtype, args.seed + 1)
    cos, sin = _make_rope(prefill_module, config, dtype)
    backends = ["reference", "metal", "tiled", "fused_experimental"] if args.backend_preset == "all" else [args.backend_preset]
    reference_ms = None
    decode_ingest_ms = _compare_decode_ingest(stack_ops, weights, inputs, config, args) if args.compare_decode_ingest else None

    for backend in backends:
        backend_config = _prefill_backend(prefill_module, backend)

        def fn():
            cache = stack_ops["init_llama_stack_cache"](config, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype if mx is not None else None)
            return prefill_module.llama_stack_prefill(
                inputs,
                weights,
                cache,
                cos,
                sin,
                config,
                backend_config=backend_config,
                return_logits=args.with_lm_head,
            )

        timing = _time_fn(fn, args.iters)
        if backend == "reference":
            reference_ms = timing["mean_ms"]
        if args.validate and backend != "reference":
            ref_cache = stack_ops["init_llama_stack_cache"](config, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype if mx is not None else None)
            opt_cache = stack_ops["init_llama_stack_cache"](config, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype if mx is not None else None)
            ref = prefill_module.reference_llama_stack_prefill(inputs, weights, ref_cache, cos, sin, config)
            got = prefill_module.llama_stack_prefill(inputs, weights, opt_cache, cos, sin, config, backend_config=backend_config, return_logits=args.with_lm_head)
            ref_out = ref[0]
            got_out = got[0]
            ref_np = np.asarray(ref_out)
            got_np = np.asarray(got_out)
            atol = 2.0e-1 if args.dtype == "bfloat16" else 1.5e-1
            rtol = atol
            if not np.allclose(ref_np, got_np, atol=atol, rtol=rtol):
                raise AssertionError(f"Validation failed for backend {backend}")
        print(
            {
                "bits": args.bits,
                "cache": args.cache,
                "B": args.B,
                "S": args.S,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "vocab_size": args.vocab_size,
                "dtype": args.dtype,
                "backend_preset": backend,
                "total_ms": timing["mean_ms"],
                "ms_per_prompt_token": timing["mean_ms"] / max(args.S, 1),
                "tokens_per_second": (args.B * args.S * 1000.0) / timing["mean_ms"],
                "speedup_vs_reference": (reference_ms / timing["mean_ms"]) if reference_ms is not None else None,
                "speedup_vs_decode_ingest": (decode_ingest_ms / timing["mean_ms"]) if decode_ingest_ms is not None else None,
            }
        )


if __name__ == "__main__":
    main()
