import argparse

import mlx.core as mx

from benchmark_utils import dtype_from_string, time_fn
from models.llama_config import LlamaLikeConfig
from ops.llama_layer_ops import create_random_quantized_llama_layer_weights, init_llama_layer_cache, llama_layer_decode_loop


def _presets(selection):
    if selection == "all":
        return ["reference", "metal", "tiled", "fused_experimental"]
    return [selection]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", choices=[4, 8], type=int, default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend-preset", choices=["reference", "metal", "tiled", "fused_experimental", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    dtype = dtype_from_string(args.dtype)
    mx.random.seed(args.seed)
    cfg = LlamaLikeConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_hidden_layers=1,
        max_position_embeddings=args.MAX_S,
    ).validate()
    weights = create_random_quantized_llama_layer_weights(cfg, bits=args.bits, group_size=32, dtype=dtype, seed=args.seed)
    inputs = mx.random.normal((args.B, args.T, args.hidden_size)).astype(dtype)
    cos = mx.random.normal((args.MAX_S + 4, args.head_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((args.MAX_S + 4, args.head_dim // 2)).astype(mx.float32)

    reference_out = None
    if args.validate:
        cache = init_llama_layer_cache(cfg, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype, page_size=args.PAGE_SIZE)
        reference_out, _ = llama_layer_decode_loop(inputs, weights, cache, cos, sin, cfg, backend_preset="reference", cache_layout=args.cache)
        mx.eval(reference_out)

    results = {}
    for preset in _presets(args.backend_preset):
        if args.validate and preset != "reference":
            cache = init_llama_layer_cache(cfg, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype, page_size=args.PAGE_SIZE)
            got, _ = llama_layer_decode_loop(inputs, weights, cache, cos, sin, cfg, backend_preset=preset, cache_layout=args.cache)
            mx.eval(got)
            if not mx.allclose(got, reference_out, atol=1.2e-1, rtol=1.2e-1).item():
                raise AssertionError(f"{preset} failed validation against reference")
        results[preset] = time_fn(
            lambda p=preset: llama_layer_decode_loop(
                inputs,
                weights,
                init_llama_layer_cache(cfg, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype, page_size=args.PAGE_SIZE),
                cos,
                sin,
                cfg,
                backend_preset=p,
                cache_layout=args.cache,
            ),
            warmup=3,
            iters=args.iters,
        )

    reference_ms = results.get("reference", {}).get("mean_ms")
    tiled_ms = results.get("tiled", {}).get("mean_ms")
    for preset, timing in results.items():
        total_ms = timing["mean_ms"]
        ms_per_token = total_ms / args.T
        tps = (args.B * args.T) / (total_ms / 1e3)
        line = (
            f"bits={args.bits} cache={args.cache} B={args.B} T={args.T} hidden_size={args.hidden_size} "
            f"intermediate_size={args.intermediate_size} num_heads={args.num_heads} num_kv_heads={args.num_kv_heads} "
            f"head_dim={args.head_dim} gqa_group_size={args.num_heads // args.num_kv_heads} max_seq_len={args.MAX_S} "
            f"dtype={args.dtype} backend_preset={preset} total_ms={total_ms:.3f} ms_per_token={ms_per_token:.3f} tokens_per_second={tps:.3f}"
        )
        if reference_ms is not None and preset != "reference":
            line += f" speedup_vs_reference={reference_ms / total_ms:.3f}"
        if tiled_ms is not None and preset in ("reference", "fused_experimental", "metal"):
            line += f" speedup_vs_tiled={tiled_ms / total_ms:.3f}"
        print(line)


if __name__ == "__main__":
    main()
