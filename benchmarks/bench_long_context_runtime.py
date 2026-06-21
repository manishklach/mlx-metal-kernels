from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

from models.llama_config import LlamaLikeConfig, tiny_gqa_debug_config
from models.long_context_runtime import (
    LongContextRuntime,
    LongContextRuntimeConfig,
    create_long_context_runtime_state,
)


@dataclass
class LongContextRuntimeBenchResult:
    prompt_len: int = 0
    shared_prefix_len: int = 0
    matched_prefix_len: int = 0
    window_size: int = 0
    sink_tokens: int = 0
    offload_ratio: float = 0.0
    quantized_kv: bool = False
    blocks_offloaded: int = 0
    blocks_prefetched: int = 0
    full_baseline_ms: float = 0.0
    integrated_runtime_ms: float = 0.0
    speedup_vs_baseline: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_len": self.prompt_len,
            "shared_prefix_len": self.shared_prefix_len,
            "matched_prefix_len": self.matched_prefix_len,
            "window_size": self.window_size,
            "sink_tokens": self.sink_tokens,
            "offload_ratio": self.offload_ratio,
            "quantized_kv": self.quantized_kv,
            "blocks_offloaded": self.blocks_offloaded,
            "blocks_prefetched": self.blocks_prefetched,
            "full_baseline_ms": self.full_baseline_ms,
            "integrated_runtime_ms": self.integrated_runtime_ms,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "metadata": dict(self.metadata),
        }


def _build_config(args):
    return LlamaLikeConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.prompt_len + args.max_new_tokens + 64,
        vocab_size=args.vocab_size,
        model_type="bench_long_context",
    ).validate()


def _build_runtime_config(args):
    from ops.sparse_attention_ops import SparseAttentionPattern

    sparse_pattern = None
    if args.sparse:
        sparse_pattern = SparseAttentionPattern(
            pattern="sliding_window_sink" if args.sink_tokens > 0 else "sliding_window",
            window_size=args.window_size,
            sink_tokens=args.sink_tokens,
            causal=True,
        )

    offload_policy = None
    if args.offload:
        offload_ratio = args.offload_ratio
        from models.kv_offload_policy import KVOffloadPolicyConfig

        block_size = 128
        total_blocks = (args.prompt_len + args.max_new_tokens + block_size - 1) // block_size
        keep_recent = max(1, int(total_blocks * (1.0 - offload_ratio)))
        keep_sink = 1
        offload_policy = KVOffloadPolicyConfig(
            block_size=block_size,
            keep_recent_blocks=keep_recent,
            keep_sink_blocks=keep_sink,
        ).validate()

    quantized_kv_config = None
    if args.quantized_kv:
        from ops.quantized_kv_cache_ops import QuantizedKVCacheConfig
        quantized_kv_config = QuantizedKVCacheConfig(bits=args.kv_bits, group_size=32).validate()

    return LongContextRuntimeConfig(
        use_prefix_cache=args.prefix_reuse,
        use_sparse_attention=args.sparse,
        use_kv_offload=args.offload,
        use_quantized_kv=args.quantized_kv,
        sparse_pattern=sparse_pattern,
        offload_policy=offload_policy,
        quantized_kv_config=quantized_kv_config,
        backend_preset=args.backend_preset,
        seed=args.seed,
    ).validate()


def _time_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000


def run_benchmark(args):
    import mlx.core as mx

    model_config = _build_config(args)
    runtime_config = _build_runtime_config(args)

    from ops.llama_stack_ops import create_random_quantized_llama_stack_weights
    weights = create_random_quantized_llama_stack_weights(
        model_config,
        vocab_size=args.vocab_size,
        bits=args.bits,
        group_size=32,
        dtype=mx.float16,
        seed=args.seed,
        include_embedding=True,
        include_lm_head=True,
    )

    prompt_a = list(range(args.prompt_len))
    prompt_b = list(range(args.shared_prefix_len)) + list(range(args.shared_prefix_len, args.prompt_len))

    def create_and_prefill(prompt):
        runtime = LongContextRuntime(
            model_config=model_config,
            stack_weights=weights,
            embedding=weights.embedding,
            lm_head=weights.lm_head,
            runtime_config=runtime_config,
        )
        state = runtime.init_state(max_seq_len=args.prompt_len + args.max_new_tokens + 64)
        state, report = runtime.prefill(prompt, state=state)
        return state, report, runtime

    baseline_config = LongContextRuntimeConfig(
        use_prefix_cache=False,
        use_sparse_attention=False,
        use_kv_offload=False,
        use_quantized_kv=False,
        backend_preset=args.backend_preset,
    ).validate()

    def baseline_prefill(prompt):
        baseline_runtime = LongContextRuntime(
            model_config=model_config,
            stack_weights=weights,
            embedding=weights.embedding,
            lm_head=weights.lm_head,
            runtime_config=baseline_config,
        )
        state = baseline_runtime.init_state(max_seq_len=args.prompt_len + args.max_new_tokens + 64)
        state, _ = baseline_runtime.prefill(prompt, state=state)
        return state

    full_baseline_ms = _time_fn(lambda: baseline_prefill(prompt_a), warmup=1, iters=args.iters)
    state_a, report_a, runtime = create_and_prefill(prompt_a)
    state_b, report_b, _ = create_and_prefill(prompt_b)

    def create_and_decode(state):
        for t in range(args.max_new_tokens):
            state, report = runtime.decode_one(0, state)
        return state

    integrated_ms = _time_fn(lambda: create_and_decode(state_b), warmup=1, iters=args.iters)

    result = LongContextRuntimeBenchResult(
        prompt_len=args.prompt_len,
        shared_prefix_len=args.shared_prefix_len,
        matched_prefix_len=report_b.matched_prefix_length if args.prefix_reuse else 0,
        window_size=args.window_size if args.sparse else 0,
        sink_tokens=args.sink_tokens if args.sparse else 0,
        offload_ratio=args.offload_ratio if args.offload else 0.0,
        quantized_kv=args.quantized_kv,
        blocks_offloaded=report_b.blocks_offloaded,
        blocks_prefetched=report_b.blocks_prefetched,
        full_baseline_ms=full_baseline_ms,
        integrated_runtime_ms=integrated_ms,
        speedup_vs_baseline=full_baseline_ms / max(integrated_ms, 1e-9),
        metadata={
            "prefix_cache_hit_a": report_a.prefix_cache_hit,
            "prefix_cache_hit_b": report_b.prefix_cache_hit,
            "sparse_positions_a": report_a.sparse_positions_count,
            "sparse_positions_b": report_b.sparse_positions_count,
            "backend_preset": args.backend_preset,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "vocab_size": args.vocab_size,
            "bits": args.bits,
        },
    )

    print(f"prompt_len={result.prompt_len}")
    print(f"shared_prefix_len={result.shared_prefix_len}")
    print(f"matched_prefix_len={result.matched_prefix_len}")
    print(f"window_size={result.window_size}")
    print(f"sink_tokens={result.sink_tokens}")
    print(f"offload_ratio={result.offload_ratio}")
    print(f"quantized_kv={result.quantized_kv}")
    print(f"blocks_offloaded={result.blocks_offloaded}")
    print(f"blocks_prefetched={result.blocks_prefetched}")
    print(f"full_baseline_ms={result.full_baseline_ms:.3f}")
    print(f"integrated_runtime_ms={result.integrated_runtime_ms:.3f}")
    print(f"speedup_vs_baseline={result.speedup_vs_baseline:.3f}x")

    return result


def main():
    parser = argparse.ArgumentParser(description="Long-context runtime benchmark")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prefix-reuse", action="store_true")
    parser.add_argument("--shared-prefix-len", type=int, default=384)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload-ratio", type=float, default=0.5)
    parser.add_argument("--quantized-kv", action="store_true")
    parser.add_argument("--kv-bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--backend-preset", default="fused_experimental")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_benchmark(args)
    print("\nJSON:")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
