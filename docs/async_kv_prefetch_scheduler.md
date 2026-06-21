# Async KV Prefetch Scheduler

## 1. Purpose

The KV prefetch scheduler is a deterministic step-based prefetch layer for long-context offload experiments. It plans which KV blocks to prefetch before attention needs them, submits prefetch requests, tracks in-flight requests, simulates latency, and reports scheduler activity.

## 2. Relationship to KV offload tier

PR #39 introduced KVBlockId, KVResidencyMap, and InMemoryKVOffloadStore for offload metadata. PR #41 integrated sparse attention planning with offload/prefetch. This PR (#43) adds a scheduler layer that can plan and issue KV block prefetches before attention needs them, rather than fetching on demand.

## 3. Why prefetch scheduling matters

On-demand prefetch (fetching a block right when attention needs it) introduces latency proportional to the store access time. A scheduler can issue prefetches proactively (lookahead), overlap prefetch with ongoing decode work, and provide visibility into prefetch activity via reports.

## 4. Step-based scheduler design

The scheduler uses a deterministic step model:

- `submit()` queues a prefetch request
- `advance_step()` moves queued requests to in-flight (up to max_in_flight), then completes requests whose simulated latency has elapsed
- `poll_ready()` returns completed results
- `wait_for()` advances until specified blocks are ready or a timeout is reached

No background threads are required.

## 5. Requests, queue, in-flight, completed

- `KVPrefetchRequest`: a prefetch request with request_id, block_id, priority, issue_step, ready_step, status
- Statuses: `queued`, `in_flight`, `complete`, `failed`, `cancelled`
- `KVPrefetchResult`: returned by `poll_ready()` and `wait_for()`, with ok/status/message
- `KVPrefetchSchedulerStats`: tracks queued, in_flight, complete, failed, cancelled counts

## 6. Sparse attention lookahead

`prefetch_blocks_for_sparse_decode()` plans prefetch for upcoming decode positions:

- Uses `needed_positions_for_sparse_decode` for current + lookahead positions
- Maps positions to block IDs
- Submits only non-resident, offloaded blocks
- Returns list of submitted requests

## 7. Speculative draft lookahead

`prefetch_blocks_for_speculative_draft()` plans prefetch for upcoming speculative draft verification:

- For each position from current_length+1 to current_length+draft_length
- Uses sparse pattern to determine needed KV positions
- Submits blocks that are offloaded

## 8. Long-context runtime integration

`LongContextRuntimeConfig.use_prefetch_scheduler=True` enables the scheduler. During `long_context_decode_step()`, the prefetch scheduler:

1. Plans prefetch blocks using sparse pattern + lookahead
2. Submits requests
3. Advances the scheduler step
4. Polls for completed requests
5. Ensures blocks are resident before attention (via the existing `ensure_blocks_ready_for_attention`)

Events are emitted for `prefetch_submitted`, `prefetch_complete`, `prefetch_failed`, `scheduler_step`.

## 9. Simulated latency

`simulated_latency_steps` controls how many `advance_step()` calls must pass before an in-flight request is considered complete. `simulated_latency_ms` is a metadata field. Both are clearly labeled as simulated.

## 10. Benchmark commands

```bash
# Quick run
python benchmarks/bench_kv_prefetch_scheduler.py --seq-len 256 --block-size 64 --window-size 128 --lookahead-tokens 2 --max-in-flight 2 --simulated-latency-steps 1

# Full run
python benchmarks/bench_kv_prefetch_scheduler.py --seq-len 4096 --block-size 128 --num-layers 2 --num-kv-heads 8 --head-dim 128 --window-size 512 --sink-tokens 4 --lookahead-tokens 8 --max-in-flight 4 --simulated-latency-steps 2 --offload-ratio 0.75
```

## 11. Current limitations

- deterministic step scheduler (not real async)
- simulated latency (no real IO timing)
- no production threads required
- no real DMA
- no kernel-level IO overlap
- contiguous cache first
- B=1 first
- file-backed store is local `.npy`
- speculative draft lookahead is scaffolded (no real speculative decode integration)

## 12. Future work

- real background worker
- mmap-backed KV blocks
- async file IO
- Metal compute/IO overlap
- latency-aware sparse scheduling
- speculative draft-aware prefetch
- prefetch cancellation based on rejected speculative tokens
