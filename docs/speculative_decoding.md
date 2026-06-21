# Speculative Decoding / MTP Scaffold

This module provides a **correctness-first scaffold** for speculative decoding with a target model and a draft model.

## Scope

- **B=1, contiguous cache first.** Paged speculative cache raises `NotImplementedError`.
- **Synthetic draft proposers** only (fixed, random, greedy self). No real draft model checkpoints.
- **Synthetic MTP head** (random weights, deterministic seed). No trained MTP weights.
- **Sequential target verification.** Optimized batched/parallel verification is future work.
- **No speedup claims.** Benchmarks measure plumbing overhead only.

## Quickstart

```python
from models import FixedDraftProposer, SpeculativeConfig, SpeculativeGenerator
from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig

pipeline = TinyGenerationPipeline(config=TinyGenerationPipelineConfig(...))
proposer = FixedDraftProposer([10, 20, 30, 40])
cfg = SpeculativeConfig(draft_length=4, max_new_tokens=16).validate()

gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
result = gen.generate_text("Hello")
print(result.text)
print(f"Acceptance rate: {result.acceptance_rate():.2f}")
```

Or via the pipeline shortcut:

```python
result = pipeline.generate_speculative("Hello", max_new_tokens=16, draft_length=4, draft_mode="fixed")
```

## Core Types

### `SpeculativeConfig`
| Field | Default | Description |
|-------|---------|-------------|
| `draft_length` | 4 | Number of draft tokens per step |
| `max_new_tokens` | 16 | Total tokens to generate |
| `temperature` | 1.0 | Sampling temperature for non-greedy verification |
| `top_k` | None | Top-k filtering |
| `top_p` | None | Top-p (nucleus) filtering |
| `greedy_verify` | True | Use greedy target verification |
| `seed` | 0 | Random seed |
| `require_exact_match` | True | Reject all after first mismatch |
| `cache_layout` | "contiguous" | Cache layout ("contiguous" or "paged") |

### `DraftProposal`
- `token_ids: list[int]` — proposed next tokens
- `logits: Any | None` — optional draft logits
- `metadata: dict[str, Any]`

### `VerificationResult`
- `accept_mask: list[bool]` — which draft tokens matched the target
- `accepted_count / rejected_count: int`
- `replacement_token_id: int | None` — first mismatched target token
- `accepted_tokens()`, `rejected_tokens()`, `all_committed_tokens()`

### `SpeculativeStepResult`
- `proposal: DraftProposal`
- `verification: VerificationResult`
- `committed_token_ids: list[int]` — tokens actually added this step
- `accepted_count: int`
- `cache_committed: bool`

### `SpeculativeGenerationResult`
- `acceptance_rate() -> float` — total_accepted / total_proposed
- `tokens_per_step() -> float` — average committed tokens per step
- `to_dict()` — serializable dict

## Accept/Reject Logic

```
proposed: [A, B, C, D]
target:   [A, B, X, Y]
accept:   [✓, ✓, ✗, ✗]
committed: [A, B, X]  (accepted prefix + replacement)
```

- `require_exact_match=True` (default): first mismatch rejects all subsequent tokens.
- `verify_draft_tokens(proposed, target)` returns a `VerificationResult`.

## Draft Proposers

| Proposer | Description |
|----------|-------------|
| `FixedDraftProposer(fixed_ids)` | Always proposes the same token IDs |
| `RandomDraftProposer(vocab_size, seed)` | Proposes random uniform tokens |
| `GreedySelfDraftProposer(pipeline)` | Uses the target model itself as draft (greedy) |

## Target Verifier

`PipelineTargetVerifier(pipeline)` runs the target model greedily for verification. This is a **sequential** verifier — optimized parallel verification is future work.

## Cache Ops (`ops/speculative_cache_ops.py`)

| Function | Description |
|----------|-------------|
| `commit_accepted_cache(draft, committed, accepted_count)` | Copy `accepted_count` positions from draft cache into committed cache |
| `discard_suffix(cache, suffix_start)` | Slice cache to `suffix_start` positions (raises `NotImplementedError` for paged) |

## MTP Scaffold (`models/mtp.py`)

| Type | Description |
|------|-------------|
| `MTPConfig` | num_draft_tokens, hidden_size, num_layers, seed, max_seq_len |
| `SyntheticMTPHead` | Random-weight MTP head producing logits from hidden states |
| `mtp_propose_tokens(head, hidden, num_tokens)` | Sample token IDs from MTP head logits |

## Performance Notes

- Speculative decoding adds overhead vs. baseline greedy generation for small models.
- The `bench_speculative_decoding.py` benchmark measures this overhead explicitly.
- Acceptance rates with random/fixed draft proposers are expected to be near zero for random and high for fixed (when tokens happen to match).

## Design Decisions

- **Separate from `generate()`** — speculative mode is opt-in via `generate_speculative()`.
- **Lazy imports** — `mlx.core` imports are guarded with `try/except ImportError`.
- **Numpy fallbacks** — all helpers work with numpy arrays; mlx arrays handled where available.
- **Deterministic seeds** — controlled via `SpeculativeConfig.seed` and proposer seed.
- **No mutation of committed cache** — commit operation returns a new cache.
