# MLX Metal Kernels — Performance Report

> Fill every bracketed field with a real measurement before publishing. If a number was not
> measured on this exact machine, leave the row out rather than estimate it. This report is
> the evidence behind every speed claim in the README — it should be regenerable by anyone
> with `git clone` + an Apple Silicon Mac.

## 1. Machine & environment

| Field | Value |
|---|---|
| Chip | [e.g. Apple M3 Max, 16-core GPU] |
| Unified memory | [e.g. 64 GB] |
| macOS version | [e.g. 15.x] |
| Python version | [e.g. 3.11.x] |
| MLX version | [`python -c "import mlx; print(mlx.__version__)"`] |
| Repo version / commit | [`git rev-parse --short HEAD`, e.g. v0.19.0 @ abc1234] |
| Power state | [Plugged in / On battery — note this, it affects clocking] |
| Date measured | [YYYY-MM-DD] |
| Other apps running | [Idle machine / specify if not] |

## 2. Methodology

- Warm-up iterations: [N] (discarded)
- Measured iterations: [N]
- Statistic reported: [median / p50, with min-max or stdev noted]
- Correctness gate: every optimized backend is compared against its pure-MLX reference path
  within dtype-appropriate tolerance (`atol`/`rtol` — state values) **before** it is benchmarked.
  Backends that fail correctness are excluded from these results, not just flagged.
- Benchmark generation command:
  ```
  python benchmarks/run_all_benchmarks.py --full \
    --output benchmarks/results/local_results.json \
    --csv benchmarks/results/local_results.csv
  python scripts/save_benchmark_report.py benchmarks/results/local_results.json \
    --output docs/performance_report_local.md
  ```

## 3. Headline summary

This is the table that belongs in the README. Keep it to 5-8 rows — the single best verified
result per kernel family, not every shape you tested.

| Kernel family | Best backend | Shape | Metric | Value | vs. reference | vs. MLX-native |
|---|---|---|---|---|---|---|
| Attention (prefill) | [backend] | [B,S,H,D,dtype] | latency (ms) | [x.xx] | [x.xx×] | [x.xx×] |
| Decode attention | [backend] | [B,MAX_S,H,D] | latency (ms) | [x.xx] | [x.xx×] | [x.xx×] |
| Paged decode attention | [backend] | [B,PAGE_SIZE,H,D] | latency (ms) | [x.xx] | [x.xx×] | [x.xx×] |
| q4 matvec decode | [backend] | [K,N,group_size] | latency (ms) | [x.xx] | [x.xx×] | [x.xx×] |
| Quantized decode block | [backend] | [H,D,MAX_S,T] | latency (ms) | [x.xx] | [x.xx×] | [x.xx×] |
| Quantized MLP block | [backend] | [hidden,intermediate] | latency (ms) | [x.xx] | [x.xx×] | [x.xx×] |
| End-to-end decode (if available) | [backend] | [model, bits] | tok/s | [x.x] | — | [x.xx× vs. mlx-lm or reference] |

`vs. reference` = speedup over the pure-MLX reference implementation in this repo.
`vs. MLX-native` = speedup over `mlx.nn` / built-in MLX ops doing the equivalent work, where
a fair comparison exists. Leave the cell as `n/a` rather than guess.

## 4. Detailed results

### 4.1 Attention

Command:
```
python benchmarks/bench_attention.py --backend all --S 128 --H 8 --D 64 --dtype float16
```

| Backend | S | H | D | dtype | Latency (ms) | Speedup vs. reference |
|---|---|---|---|---|---|---|
| reference | | | | | | 1.00× |
| metal | | | | | | |
| row_parallel | | | | | | |
| tiled_kv | | | | | | |
| threadgroup_v2 | | | | | | |
| simdgroup_d64 | | | | | | |

### 4.2 Decode & KV-cache

Commands:
```
python benchmarks/bench_decode_attention.py --B 2 --MAX_S 32 --H 8 --D 64 --length 32 --dtype float16 --backend all
python benchmarks/bench_paged_decode_attention.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --H 8 --D 64 --length 128 --dtype float16 --backend all
python benchmarks/bench_gqa_decode_attention.py --B 1 --MAX_S 128 --Hq 32 --Hkv 8 --D 128 --dtype float16 --cache contiguous --backend all
```

| Benchmark | Backend | Shape | Latency (ms) | Speedup vs. reference |
|---|---|---|---|---|
| decode_attention | | | | |
| paged_decode_attention | | | | |
| gqa_decode_attention | | | | |

### 4.3 Quantization (q4/q8 matvec)

Commands:
```
python benchmarks/bench_dequant.py --bits 4 --M 4096 --K 4096 --dtype float16 --backend all
python benchmarks/bench_quant_matvec_parallel.py --bits 4 --B 1 --K 4096 --N 4096 --group-size 32 --dtype float16 --backend all
python benchmarks/bench_quant_matvec_tiled.py --bits 4 --B 1 --K 4096 --N 4096 --dtype float16 --backend all
```

| Bits | Backend | K | N | group_size | Latency (ms) | GB/s effective | Speedup vs. reference |
|---|---|---|---|---|---|---|---|
| 4 | | | | | | | |
| 8 | | | | | | | |

### 4.4 Transformer blocks

Commands:
```
python benchmarks/bench_quantized_decode_block.py --bits 4 --cache contiguous --B 1 --K 4096 --H 32 --D 128 --MAX_S 128 --T 16 --dtype float16 --backend-preset parallel
python benchmarks/bench_quantized_mlp_block.py --bits 4 --B 1 --S 1 --hidden-size 4096 --intermediate-size 11008 --dtype float16 --backend-preset all
python benchmarks/bench_fused_mlp_kernels.py --bits 4 --B 1 --S 1 --hidden-size 4096 --intermediate-size 11008 --dtype float16 --backend-preset all --validate
python benchmarks/bench_llama_layer_decode.py --bits 4 --B 1 --T 16 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --MAX_S 128 --dtype float16 --backend-preset all --validate
```

| Block | Backend preset | Shape | Latency (ms) | Speedup vs. reference |
|---|---|---|---|---|
| quantized_decode_block | | | | |
| quantized_mlp_block | | | | |
| fused_mlp (experimental) | | | | |
| llama_layer_decode (synthetic) | | | | |

### 4.5 End-to-end generation (if real checkpoint loaded)

Only fill this in if tokens were generated from real model weights, not the synthetic/random
weight scaffolding. State the model name, parameter count, and quantization bits explicitly.

| Model | Params | Bits | Backend preset | Prefill tok/s | Decode tok/s | Memory (GB) |
|---|---|---|---|---|---|---|
| | | | | | | |

If this section is empty, say so explicitly in the README rather than omitting it silently —
"end-to-end generation on real weights has not yet been benchmarked" is an honest, useful
status, and matches this repo's own correctness-first philosophy.

## 5. Known limitations of this report

- Single machine, single thermal state — not representative of all Apple Silicon chips.
- [Add any other caveats: battery vs. plugged in, background load, MLX version sensitivity, etc.]
- Autotune cache (if used) is specific to this machine and should not be copied to other hardware.

## 6. Raw data

- JSON: `benchmarks/results/local_results.json`
- CSV: `benchmarks/results/local_results.csv`
- Regenerate with the commands in Section 2.
