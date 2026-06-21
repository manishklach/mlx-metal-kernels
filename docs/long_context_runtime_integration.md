# Long-Context Runtime Integration

## 1. Purpose

This PR integrates the long-context runtime pieces into one explicit experimental path:

- prefix KV-cache reuse
- sparse attention needed-position planning
- KV offload residency checks and prefetch
- optional quantized KV metadata

The result is a single, explicit `LongContextRuntime` class that orchestrates these concerns without assuming production serving or async scheduling.

## 2. What this integrates

| Concern | Module | Status |
|---|---|---|
| Prefix cache | `models/prefix_cache.py` | Existing |
| Sparse attention | `ops/sparse_attention_ops.py` | Existing |
| KV offload metadata | `models/kv_offload.py` | Existing |
| KV offload store | `models/kv_offload_store.py` | Existing |
| KV offload policy | `models/kv_offload_policy.py` | Existing |
| KV offload ops | `ops/kv_offload_ops.py` | Existing |
| Quantized KV | `ops/quantized_kv_cache_ops.py` | Existing |
| **Integration runtime** | `models/long_context_runtime.py` | **New** |
| **Integration ops** | `ops/long_context_ops.py` | **New** |

## 3. Prefix reuse

- `LongContextRuntimeConfig.use_prefix_cache=True` enables prefix-cache via `InMemoryPrefixCache`.
- Each prefill calls `prefill_with_prefix_reuse` with a fingerprint computed from model config.
- The report shows `prefix_cache_hit` and `matched_prefix_length`.

## 4. Sparse attention planning

- `LongContextRuntimeConfig.use_sparse_attention=True` enables sparse needed-position planning.
- `needed_positions_for_sparse_decode` computes which K/V positions are visible under the configured `SparseAttentionPattern`.
- When offload is also enabled, the runtime ensures resident blocks before allowing sparse attention to proceed.

## 5. KV offload prefetch

- `LongContextRuntimeConfig.use_kv_offload=True` enables offload/policy planning.
- After prefill and each decode step, `plan_offload_blocks` is called to offload cold blocks.
- Before decode step, `ensure_blocks_ready_for_attention` prefetches needed offloaded blocks.
- The runtime blocks attention if required blocks are missing (no silent fallback to dense).

## 6. Quantized KV integration status

- `LongContextRuntimeConfig.use_quantized_kv=True` sets the quantized KV flag.
- The runtime records `quantized_kv_enabled` in the report and stores the config.
- Full quantized KV-cache decode routing through the stack is scaffolded and documented as future work.
- Tests verify that config combinations are accepted or rejected with clear errors.

## 7. Runtime state and reports

`LongContextRuntimeState` carries:

- `stack_cache` — the active MLX cache
- `prefix_cache` — optional `InMemoryPrefixCache`
- `residency_map` — optional `KVResidencyMap`
- `offload_store` — optional `InMemoryKVOffloadStore`
- `quantized_kv_cache` — optional metadata dict

`LongContextRuntimeReport` provides:

- `ok`, `events`, `prefix_cache_hit`, `matched_prefix_length`, `suffix_length`
- `sparse_positions_count`, `blocks_needed`, `blocks_prefetched`, `blocks_offloaded`
- `quantized_kv_enabled`, `metadata`
- `.errors()`, `.warnings()`, `.summary()`, `.to_dict()`, `.pretty_print()`

## 8. Example flow

```python
runtime = LongContextRuntime(
    model_config=config,
    stack_weights=weights,
    embedding=weights.embedding,
    runtime_config=runtime_config,
)
state = runtime.init_state(max_seq_len=4096)
state, report = runtime.prefill(prompt_A_tokens, state=state)
state, report = runtime.prefill(prompt_B_tokens, state=state)
result = runtime.generate(prompt_B_tokens, max_new_tokens=8)
```

## 9. Benchmark commands

Quick:
```
python benchmarks/bench_long_context_runtime.py --prompt-len 32 --shared-prefix-len 24 --max-new-tokens 2 --prefix-reuse --sparse --window-size 8 --sink-tokens 2 --offload --offload-ratio 0.5 --num-layers 1 --hidden-size 64 --intermediate-size 128 --num-heads 4 --num-kv-heads 2 --head-dim 16 --bits 4 --backend-preset fused_experimental --iters 3
```

Full:
```
python benchmarks/bench_long_context_runtime.py --prompt-len 512 --shared-prefix-len 384 --max-new-tokens 8 --prefix-reuse --sparse --window-size 128 --sink-tokens 4 --offload --offload-ratio 0.5 --num-layers 2 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --bits 4 --backend-preset fused_experimental
```

## 10. Current limitations

- experimental explicit mode only
- synthetic weights/examples; no real model quality
- B=1 only (multi-batch is future work)
- contiguous cache only (paged is future work)
- no real async IO or DMA offload
- no production flash scheduling
- sparse stack decode may be scaffolded if not fully routed through existing stack decode
- quantized KV full-stack routing is scaffolded as metadata only (future work)

## 11. Future work

- full sparse decode routing through stack layers
- quantized KV stack decode routing
- async prefetch scheduler
- paged/offloaded KV-cache
- prefix-cache + offload persistence
- real package tensor-data loader
