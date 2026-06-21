# Parallel speculative verification

## Purpose

This document describes the parallel/staged speculative verification path added in PR #42. It extends the v0.38 speculative decoding scaffold with a batched verification API that verifies multiple proposed draft tokens against a target pipeline in a single staged pass.

## Relationship to PR38 speculative scaffold

PR38 introduced the core speculative decoding infrastructure:

- `SpeculativeConfig`, `DraftProposal`, `VerificationResult`, `SpeculativeStepResult`, `SpeculativeGenerationResult`
- Three draft proposers: `FixedDraftProposer`, `RandomDraftProposer`, `GreedySelfDraftProposer`
- `PipelineTargetVerifier` — sequential target verifier that generates one token at a time via `generate_ids`
- `SpeculativeGenerator` — accept/reject loop driving generation
- `ops/speculative_cache_ops.py` — `commit_accepted_cache`, `discard_suffix`

PR #42 adds:

- `ParallelVerificationConfig` — configuration for staged/parallel verification
- `ParallelVerificationPassResult` — structured result with `accept_mask`, `staged_cache`, `to_verification_result()`
- `parallel_verify_tokens` — core verification function that clones the cache and runs a staged decode loop
- `ParallelTargetVerifier` — `TargetVerifier` implementation wrapping `parallel_verify_tokens`
- `commit_parallel_verification_cache` — staged cache commit helper
- `TinyGenerationPipeline.generate_speculative(verifier_mode="parallel")` — pipeline integration
- Benchmark, test, example, and documentation

## Sequential vs parallel/staged verification

### Sequential (PipelineTargetVerifier)

```
for each draft token:
    generate target token via pipeline.generate_ids
    append to all_ids
    update state
compare proposed vs target → accept_mask
```

The sequential verifier generates each target token one at a time through the full pipeline, committing each token's KV-cache incrementally.

### Parallel/staged (ParallelTargetVerifier)

```
clone committed cache → staged cache
for each draft token:
    run decode_step on staged cache
    collect logits
compare proposed vs target → accept_mask
commit accepted positions from staged cache to committed cache
```

The parallel/staged verifier:

1. Clones the committed KV-cache into a staging area.
2. Runs decode steps for all proposed tokens against the staged cache.
3. Computes accept/reject mask from target logits.
4. Returns the staged cache without modifying the committed cache.
5. The caller decides which positions to commit.

Because full continuation prefill (`start_position > 0`) is not yet implemented, the staged verification currently uses a decode loop rather than a true batched prefill. This is documented as `verification_path="decode_loop_staged"`.

## Draft token verification semantics

For verifying draft tokens `[d0, d1, ..., dK-1]` after context C:

- The target token for `d0` is sampled from logits produced after consuming context C.
- The target token for `d1` is sampled after feeding `d0` to the model in the staged cache.
- This matches exact speculative decoding semantics: the target model verifies each draft token by computing the likelihood of the next token given all previously accepted tokens.

The current staged decode loop feeds each draft token sequentially into the staged cache and collects per-position logits.

## Accept/reject mask

The accept/reject mask uses `compute_accept_mask` from the existing speculative decoding module:

```python
accept_mask = compute_accept_mask(proposed_tokens, target_tokens, require_exact_match=True)
```

By default, `require_exact_match=True` enforces prefix-contiguous acceptance: `[True, True, True, False, ...]`. No gaps are allowed in the accepted prefix.

## Staged KV-cache

The staged KV-cache is a deep clone of the committed cache before verification. After verification:

- The staged cache contains KV values for all proposed token positions.
- The committed cache is untouched.
- The caller inspects `accepted_count` and calls `commit_parallel_verification_cache` to copy accepted positions.

This design ensures the committed cache is never mutated until acceptance is determined.

## Cache commit

`commit_parallel_verification_cache` delegates to `commit_accepted_cache` from `speculative_cache_ops.py`:

- If `accepted_count > 0`: copies positions `[0, accepted_count)` from staged to committed.
- `include_replacement=False` (default): only accepted prefix is copied.
- Contiguous cache only. Paged cache raises `NotImplementedError`.

## Integration with TinyGenerationPipeline

`TinyGenerationPipeline.generate_speculative(verifier_mode="parallel")`:

- Creates a `ParallelTargetVerifier` with a `ParallelVerificationConfig`.
- The `SpeculativeGenerator` uses this verifier in the accept/reject loop.
- Result metadata includes `verifier_mode="parallel"`, `average_accepted`, and `verification_path`.

## Benchmarks

Run the CLI benchmark:

```bash
python benchmarks/bench_parallel_speculative_verify.py \
    --prompt-len 8 --max-new-tokens 16 --draft-length 4 \
    --draft-mode fixed --verifier both \
    --num-layers 1 --hidden-size 64 --intermediate-size 128 \
    --num-heads 4 --num-kv-heads 2 --head-dim 16 \
    --bits 4 --backend-preset fused_experimental
```

The benchmark reports mean ms, acceptance rate, and tokens per step for both sequential and parallel verifiers.

## Current limitations

- Synthetic/random weights only.
- B=1 first. B>1 is future work.
- Exact speculative semantics are scaffolded via staged decode loop.
- True batched prefill verification (`verification_path="prefill_batch"`) depends on continuation prefill support (`start_position > 0`), which is not yet implemented.
- No production draft model. Only synthetic proposers.
- Contiguous cache first. Paged speculative cache is not implemented.
- No tree speculation (single draft sequence only).
- No Metal accept-mask or cache-commit kernels.

## Future work

- True batched target verification over K tokens (once continuation prefill is available).
- Metal accept-mask kernel.
- Speculative cache commit kernel.
- Tree speculative decoding.
- MTP-trained heads.
- Acceptance-rate adaptive draft length.
- Long-context speculative integration.
