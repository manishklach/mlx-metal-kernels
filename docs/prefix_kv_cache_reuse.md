# Prefix KV-Cache Reuse

## Overview

Prefix KV-cache reuse enables caching of previously computed key-value (KV) cache entries so that when a new prompt shares a prefix with a previously seen prompt, the shared portion's KV cache can be reused without recomputation.

This is a **correctness-first scaffold** — it works on the existing contiguous cache layout with synthetic tiny models. Paged cache reuse and production-style serving cache are reserved for future work.

## Architecture

### Components

1. **`ops/kv_cache_reuse_ops.py`** — Low-level K/V cache manipulation
   - `clone_layer_cache` / `clone_stack_cache`: Deep-copy cache entries
   - `slice_layer_cache`: Extract first N positions from a layer cache
   - `copy_prefix_cache_into`: Copy prefix from one cache into another
   - `cache_prefix_equal`: Compare two caches (optionally up to length N)

2. **`models/prefix_cache.py`** — Cache data structures and reuse logic
   - `PrefixFingerprint`: Hash of model config + tokenizer identity prevents cross-model reuse
   - `PrefixCacheEntry`: Stores fingerprint, token_ids, and cloned stack cache
   - `PrefixCacheMatch`: Result with matched_length, suffix tokens, and entry reference
   - `InMemoryPrefixCache`: LRU-evicting dict of entries
   - `prefill_with_prefix_reuse`: High-level function that checks cache and either reuses or does full prefill

3. **`models/tiny_generation_pipeline.py`** — Pipeline integration
   - `TinyGenerationPipelineConfig.use_prefix_cache`: Opt-in flag (default: `False`)
   - `TinyGenerationPipeline.prefix_cache`: `InMemoryPrefixCache` instance when enabled
   - `_prefill_with_cache`: Internal method handling cache lookup, clone, suffix decode, and store

### Design Decisions

- **Token-based matching**: Prefix matching compares raw token IDs, not text. Two prompts that are semantically identical but tokenized differently will not share cache.
- **Opt-in only**: `use_prefix_cache=False` by default; existing behavior is unchanged.
- **Suffix fallback**: When a partial prefix match is found, the suffix is processed via token-by-token `decode_step` (since continuation prefill with `start_position > 0` is not yet implemented in the underlying ops).
- **Exact match fast path**: When the entire prompt matches, the cached cache is cloned and logits are obtained via a single decode step for the last token.
- **Cache fingerprinting**: The fingerprint includes the model config hash and tokenizer class name. Two different models or tokenizers never share cache entries.
- **SHA256 prefix**: Only the first 16 hex characters of the SHA256 hash are used as the fingerprint key (sufficient for uniqueness in this context).

### Limitations

- Contiguous cache layout only (paged raises `NotImplementedError`)
- Suffix after partial match uses decode_step (slow path per-suffix-token)
- No production serving-cache support
- Single-batch only (B=1)

## Usage

### With the generation pipeline

```python
from models import TinyGenerationPipeline, TinyGenerationPipelineConfig

config = TinyGenerationPipelineConfig(
    use_prefix_cache=True,
    # ... other config
)
pipeline = TinyGenerationPipeline(config=config)

# First call: full prefill, stored in cache
result1 = pipeline.generate("Hello world", max_new_tokens=8, greedy=True)

# Second call with same prompt: exact match, reused
result2 = pipeline.generate("Hello world", max_new_tokens=8, greedy=True)

# Extended prefix: "Hello world" reused, " again" decoded
result3 = pipeline.generate("Hello world again", max_new_tokens=8, greedy=True)
```

### With the low-level API

```python
from models import create_synthetic_stack_generation_model
from models.generation import GenerationConfig
from models.prefix_cache import InMemoryPrefixCache, prefill_with_prefix_reuse

model = create_synthetic_stack_generation_model(seed=42)
cache = InMemoryPrefixCache(max_size=64)

# Full prefill (stores in cache)
logits, state, meta = prefill_with_prefix_reuse(
    [10, 20, 30, 40, 50], model, prefix_cache=cache,
    generation_config=GenerationConfig(max_new_tokens=4),
)

# Exact match (reuses cached cache)
logits, state, meta = prefill_with_prefix_reuse(
    [10, 20, 30, 40, 50], model, prefix_cache=cache,
    generation_config=GenerationConfig(max_new_tokens=4),
)  # meta["prefix_cache_hit"] == True

# Partial match (reuses first 3, decodes last 2)
logits, state, meta = prefill_with_prefix_reuse(
    [10, 20, 30, 99, 100], model, prefix_cache=cache,
    generation_config=GenerationConfig(max_new_tokens=4),
)  # meta["matched_length"] == 3
```

### Manually managing the cache

```python
from models import compute_fingerprint
from ops.kv_cache_reuse_ops import cache_prefix_equal, clone_stack_cache

cache = InMemoryPrefixCache(max_size=32)
cache.clear()
cache.stats()
```

## Testing

```bash
pytest tests/test_kv_cache_reuse_ops.py -v
pytest tests/test_prefix_cache.py -v
pytest tests/test_prefill_with_prefix_reuse.py -v
```

## Benchmarking

```bash
python benchmarks/bench_prefix_cache_reuse.py --validate --prompt-tokens 8 --reused-tokens 6 --iters 10
```

## Files Changed

| File | Change |
|------|--------|
| `ops/kv_cache_reuse_ops.py` | New — clone/slice/copy/cache_prefix_equal ops |
| `models/prefix_cache.py` | New — fingerprint, cache entry/match, InMemoryPrefixCache, prefill_with_prefix_reuse |
| `models/tiny_generation_pipeline.py` | Modified — use_prefix_cache flag, _prefill_with_cache |
| `models/__init__.py` | Added exports |
| `ops/__init__.py` | Added exports |
| `tests/test_kv_cache_reuse_ops.py` | New |
| `tests/test_prefix_cache.py` | New |
| `tests/test_prefill_with_prefix_reuse.py` | New |
| `benchmarks/bench_prefix_cache_reuse.py` | New |
| `benchmarks/backend_registry.py` | Added prefix_cache_reuse |
| `benchmarks/run_all_benchmarks.py` | Added prefix cache reuse suite |
| `examples/prefix_cache_reuse_demo.py` | New |
| `docs/prefix_kv_cache_reuse.md` | New |
| `README.md` | Updated roadmap |
| `docs/roadmap.md` | Marked v0.37 |
