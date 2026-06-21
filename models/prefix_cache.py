from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for prefix_cache") from exc

def _generation_classes():
    from models.generation import GenerationConfig, ToyGenerationState

    return GenerationConfig, ToyGenerationState


def _config_fingerprint(config) -> str:
    d = config.to_dict()
    d.pop("vocab_size", None)
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _update_hasher_with_array(hasher, value: Any) -> None:
    array = np.asarray(value)
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    flat = array.reshape(-1) if array.ndim > 0 else array.reshape(1)
    sample_count = min(256, int(flat.shape[0]))
    if sample_count == 0:
        return
    sample = flat[:sample_count]
    if flat.shape[0] > sample_count:
        tail_count = min(64, int(flat.shape[0] - sample_count))
        if tail_count > 0:
            sample = np.concatenate([sample, flat[-tail_count:]])
    hasher.update(sample.tobytes())


def _update_hasher(hasher, value: Any) -> None:
    if value is None:
        hasher.update(b"none")
        return
    if isinstance(value, (str, bytes, int, float, bool)):
        hasher.update(repr(value).encode("utf-8"))
        return
    if isinstance(value, dict):
        for key in sorted(value):
            hasher.update(str(key).encode("utf-8"))
            _update_hasher(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(str(type(value)).encode("utf-8"))
        for item in value:
            _update_hasher(hasher, item)
        return
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        _update_hasher_with_array(hasher, value)
        return
    if hasattr(value, "__dict__"):
        hasher.update(type(value).__name__.encode("utf-8"))
        for key in sorted(vars(value)):
            hasher.update(str(key).encode("utf-8"))
            _update_hasher(hasher, getattr(value, key))
        return
    hasher.update(repr(value).encode("utf-8"))


def _model_fingerprint(model) -> str:
    hasher = hashlib.sha256()
    hasher.update(type(model).__name__.encode("utf-8"))
    weights = getattr(model, "stack_weights", None)
    if weights is None:
        weights = getattr(model, "layer_weights", None)
    if weights is not None:
        _update_hasher(hasher, weights)
    else:
        hasher.update(f"object-id:{id(model)}".encode("utf-8"))
    return hasher.hexdigest()[:16]


def compute_fingerprint(config, tokenizer=None, *, model=None) -> str:
    parts = [_config_fingerprint(config)]
    if tokenizer is not None:
        tname = type(tokenizer).__name__
        parts.append(tname)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is not None:
            parts.append(f"tokenizer_vocab:{int(vocab_size)}")
    if model is not None:
        parts.append(_model_fingerprint(model))
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class PrefixCacheEntry:
    fingerprint: str
    token_ids: list[int]
    stack_cache: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefixCacheMatch:
    matched: bool
    matched_length: int
    suffix_token_ids: list[int]
    entry: PrefixCacheEntry | None


class InMemoryPrefixCache:
    def __init__(self, max_size: int = 64):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._entries: list[PrefixCacheEntry] = []
        self._access_counter = 0
        self._entry_access: list[int] = []

    def lookup(self, token_ids: list[int], fingerprint: str) -> PrefixCacheMatch:
        if not token_ids:
            return PrefixCacheMatch(matched=False, matched_length=0, suffix_token_ids=[], entry=None)
        best_match = PrefixCacheMatch(matched=False, matched_length=0, suffix_token_ids=list(token_ids), entry=None)
        best_idx = -1
        for idx, entry in enumerate(self._entries):
            if entry.fingerprint != fingerprint:
                continue
            common = self._common_prefix_length(entry.token_ids, token_ids)
            if common > 0 and common > best_match.matched_length:
                best_match = PrefixCacheMatch(
                    matched=True,
                    matched_length=common,
                    suffix_token_ids=token_ids[common:],
                    entry=entry,
                )
                best_idx = idx
        if best_idx >= 0:
            self._entry_access[best_idx] = self._access_counter
            self._access_counter += 1
        return best_match

    def store(self, entry: PrefixCacheEntry) -> None:
        self._entries.append(entry)
        self._entry_access.append(self._access_counter)
        self._access_counter += 1
        if len(self._entries) > self.max_size:
            self._evict_one()

    def _evict_one(self) -> None:
        if not self._entries:
            return
        oldest_idx = min(range(len(self._entries)), key=lambda i: self._entry_access[i])
        self._entries.pop(oldest_idx)
        self._entry_access.pop(oldest_idx)

    def clear(self) -> None:
        self._entries.clear()
        self._entry_access.clear()
        self._access_counter = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        if not self._entries:
            return {"size": 0, "max_size": self.max_size, "fingerprints": []}
        fp_counts: dict[str, int] = {}
        total_tokens = 0
        for e in self._entries:
            fp_counts[e.fingerprint] = fp_counts.get(e.fingerprint, 0) + 1
            total_tokens += len(e.token_ids)
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "fingerprints": list(fp_counts.keys()),
            "fingerprint_counts": fp_counts,
            "avg_token_ids": total_tokens / len(self._entries) if self._entries else 0,
        }

    @staticmethod
    def _common_prefix_length(a: list[int], b: list[int]) -> int:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i


def _clone_stack_cache_safe(stack_cache):
    try:
        from ops.kv_cache_reuse_ops import clone_stack_cache as _csc
    except ImportError:
        raise RuntimeError(
            "Prefix KV-cache reuse requires mlx. "
            "Install mlx and ensure the ops package can be loaded."
        ) from None
    return _csc(stack_cache)


def _copy_prefix_cache_safe(src_cache, dst_cache, length: int):
    try:
        from ops.kv_cache_reuse_ops import copy_prefix_cache_into as _copy
    except ImportError:
        raise RuntimeError(
            "Prefix KV-cache reuse requires mlx. "
            "Install mlx and ensure the ops package can be loaded."
        ) from None
    return _copy(src_cache, dst_cache, length)


def prefill_with_prefix_reuse(
    token_ids: list[int],
    model,
    generation_config=None,
    prefix_cache: InMemoryPrefixCache | None = None,
    *,
    fingerprint: str | None = None,
    state=None,
    max_seq_len=None,
):
    GenerationConfig, ToyGenerationState = _generation_classes()

    def _run_full_prefill(run_state):
        logits_local, updated_state_local = model.prefill_token_ids(
            token_ids,
            run_state,
            generation_config=generation_config,
        )
        if prefix_cache is not None:
            cached_cache_local = _clone_stack_cache_safe(updated_state_local.cache)
            prefix_cache.store(
                PrefixCacheEntry(
                    fingerprint=fp,
                    token_ids=list(token_ids),
                    stack_cache=cached_cache_local,
                    metadata={"num_prompt_tokens": len(token_ids)},
                )
            )
        return logits_local, updated_state_local

    if not token_ids:
        raise ValueError("token_ids must contain at least one token")
    if state is None:
        if max_seq_len is None:
            max_seq_len = getattr(model.config, "max_position_embeddings", max(token_ids) + 64)
        state = model.init_state(B=1, max_seq_len=max_seq_len)
    fp = fingerprint or compute_fingerprint(model.config, getattr(model, "tokenizer", None), model=model)
    if prefix_cache is None or prefix_cache.size == 0:
        if fingerprint is not None:
            _ = fingerprint
        logits, updated_state = _run_full_prefill(state)
        metadata = {
            "prefix_cache_hit": False,
            "cache_available": prefix_cache is not None,
            "matched_length": 0,
            "suffix_length": len(token_ids),
            "suffix_mode": "full_prefill",
        }
        return logits, updated_state, metadata
    match = prefix_cache.lookup(token_ids, fp)
    if match.matched:
        suffix = match.suffix_token_ids
        if suffix:
            cache_clone = _clone_stack_cache_safe(match.entry.stack_cache)
            state = ToyGenerationState(
                cache=cache_clone,
                position=match.matched_length,
                generated_ids=list(token_ids[:match.matched_length]),
            )
            logits = None
            for token_id in suffix:
                logits, state = model.decode_step(token_id, state, generation_config=generation_config)
            suffix_mode = "decode_suffix"
        else:
            fresh_state = model.init_state(B=1, max_seq_len=match.entry.stack_cache.max_seq_len)
            logits, state = model.prefill_token_ids(
                token_ids,
                fresh_state,
                generation_config=generation_config,
            )
            suffix_mode = "exact_match_prefill"
        metadata = {
            "prefix_cache_hit": True,
            "cache_available": True,
            "matched_length": match.matched_length,
            "suffix_length": len(suffix),
            "suffix_mode": suffix_mode,
        }
        return logits, state, metadata
    logits, updated_state = _run_full_prefill(state)
    metadata = {
        "prefix_cache_hit": False,
        "cache_available": True,
        "matched_length": 0,
        "suffix_length": len(token_ids),
        "suffix_mode": "full_prefill",
    }
    return logits, updated_state, metadata
