from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for long_context_runtime") from exc

from .kv_offload import (
    KVBlockId,
    KVResidencyMap,
    partition_sequence_into_blocks,
    token_positions_to_block_ids,
)
from .kv_offload_policy import (
    KVOffloadPlan,
    KVOffloadPolicyConfig,
    plan_offload_blocks,
    plan_prefetch_for_sparse_attention,
)
from .kv_offload_store import InMemoryKVOffloadStore
from .prefix_cache import (
    InMemoryPrefixCache,
    PrefixCacheEntry,
    PrefixCacheMatch,
    compute_fingerprint,
    prefill_with_prefix_reuse,
)


def _safe_sparse_pattern(pattern: Any | None = None):
    if pattern is not None:
        return pattern
    return {
        "pattern": "sliding_window",
        "window_size": 512,
        "sink_tokens": 4,
        "causal": True,
    }


def _safe_offload_policy(policy: Any | None = None):
    if policy is not None:
        return policy
    return KVOffloadPolicyConfig(
        block_size=128,
        keep_recent_blocks=4,
        keep_sink_blocks=1,
    ).validate()


def _safe_quantized_kv_config(cfg: Any | None = None):
    if cfg is not None:
        return cfg
    return {"bits": 8, "group_size": 32}


@dataclass
class LongContextRuntimeConfig:
    use_prefix_cache: bool = True
    use_sparse_attention: bool = True
    use_kv_offload: bool = True
    use_quantized_kv: bool = False

    sparse_pattern: Any | None = None
    offload_policy: Any | None = None
    quantized_kv_config: Any | None = None

    backend_preset: str = "fused_experimental"
    attention_backend: str = "metal_sliding_window"
    decode_attention_backend: str = "metal_sliding_window"
    cache_layout: str = "contiguous"

    model_id: str | None = None
    tokenizer_id: str | None = None
    seed: int = 0

    def validate(self) -> LongContextRuntimeConfig:
        if self.cache_layout not in ("contiguous",):
            raise NotImplementedError(
                f"cache_layout={self.cache_layout!r} is not implemented; "
                "only 'contiguous' is supported."
            )
        if self.use_sparse_attention and self.sparse_pattern is None:
            self.sparse_pattern = _safe_sparse_pattern()
        if self.use_kv_offload and self.offload_policy is None:
            self.offload_policy = _safe_offload_policy()
        if self.use_quantized_kv and self.quantized_kv_config is None:
            self.quantized_kv_config = _safe_quantized_kv_config()
        if self.use_quantized_kv and self.use_kv_offload:
            pass
        if self.cache_layout == "paged" and self.use_kv_offload:
            raise NotImplementedError("paged cache + offload is not implemented")
        if self.cache_layout == "paged" and self.use_quantized_kv:
            raise NotImplementedError("paged cache + quantized kv is not implemented")
        if self.use_quantized_kv and self.use_sparse_attention:
            pass
        return self

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "use_prefix_cache": self.use_prefix_cache,
            "use_sparse_attention": self.use_sparse_attention,
            "use_kv_offload": self.use_kv_offload,
            "use_quantized_kv": self.use_quantized_kv,
            "backend_preset": self.backend_preset,
            "attention_backend": self.attention_backend,
            "decode_attention_backend": self.decode_attention_backend,
            "cache_layout": self.cache_layout,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "seed": self.seed,
        }
        if self.sparse_pattern is not None:
            try:
                d["sparse_pattern"] = self.sparse_pattern.to_dict()
            except AttributeError:
                d["sparse_pattern"] = str(self.sparse_pattern)
        if self.offload_policy is not None:
            d["offload_policy"] = self.offload_policy.to_dict()
        return d


@dataclass
class LongContextEvent:
    kind: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


_EVENT_KINDS = frozenset({
    "prefix_cache_hit",
    "prefix_cache_miss",
    "sparse_positions",
    "offload_plan",
    "prefetch",
    "missing_resident_blocks",
    "quantized_kv_enabled",
    "fallback",
    "warning",
    "error",
})


@dataclass
class LongContextRuntimeReport:
    ok: bool
    events: list[LongContextEvent]
    prefix_cache_hit: bool = False
    matched_prefix_length: int = 0
    suffix_length: int = 0
    sparse_positions_count: int = 0
    blocks_needed: int = 0
    blocks_prefetched: int = 0
    blocks_offloaded: int = 0
    quantized_kv_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def errors(self) -> list[LongContextEvent]:
        return [e for e in self.events if e.kind == "error"]

    def warnings(self) -> list[LongContextEvent]:
        return [e for e in self.events if e.kind == "warning"]

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "prefix_cache_hit": self.prefix_cache_hit,
            "matched_prefix_length": self.matched_prefix_length,
            "suffix_length": self.suffix_length,
            "sparse_positions_count": self.sparse_positions_count,
            "blocks_needed": self.blocks_needed,
            "blocks_prefetched": self.blocks_prefetched,
            "blocks_offloaded": self.blocks_offloaded,
            "quantized_kv_enabled": self.quantized_kv_enabled,
            "events": [{"kind": e.kind, "message": e.message, "metadata": dict(e.metadata)} for e in self.events],
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()

    def pretty_print(self) -> str:
        lines = [f"LongContextRuntimeReport ok={self.ok}"]
        lines.append(f"  prefix_cache_hit={self.prefix_cache_hit} matched={self.matched_prefix_length} suffix={self.suffix_length}")
        lines.append(f"  sparse_positions={self.sparse_positions_count}")
        lines.append(f"  blocks_needed={self.blocks_needed} prefetched={self.blocks_prefetched} offloaded={self.blocks_offloaded}")
        lines.append(f"  quantized_kv={self.quantized_kv_enabled}")
        lines.append(f"  events ({len(self.events)}):")
        for e in self.events:
            lines.append(f"    [{e.kind}] {e.message}")
        for k, v in self.metadata.items():
            lines.append(f"  {k}={v}")
        return "\n".join(lines)


@dataclass
class LongContextRuntimeState:
    stack_cache: Any
    prefix_cache: Any | None = None
    residency_map: Any | None = None
    offload_store: Any | None = None
    quantized_kv_cache: Any | None = None
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        desc: dict[str, Any] = {
            "position": self.position,
            "has_prefix_cache": self.prefix_cache is not None,
            "has_residency_map": self.residency_map is not None,
            "has_offload_store": self.offload_store is not None,
            "has_quantized_kv_cache": self.quantized_kv_cache is not None,
        }
        if self.prefix_cache is not None:
            desc["prefix_cache_size"] = self.prefix_cache.size
        if self.residency_map is not None:
            desc["residency"] = self.residency_map.summary()
        if self.offload_store is not None:
            desc["offload_store_stats"] = self.offload_store.stats()
        desc["metadata"] = dict(self.metadata)
        return desc


def _optional_stack_ops():
    try:
        from ops.llama_stack_ops import (
            LlamaStackCache,
            init_llama_stack_cache,
        )
        return LlamaStackCache, init_llama_stack_cache
    except ImportError:
        return None, None


def create_long_context_runtime_state(
    *,
    config: Any,
    stack_weights: Any,
    runtime_config: LongContextRuntimeConfig,
    B: int = 1,
    max_seq_len: int = 4096,
    dtype=None,
) -> LongContextRuntimeState:
    runtime_config = runtime_config.validate()
    LlamaStackCache, init_llama_stack_cache_fn = _optional_stack_ops()
    if init_llama_stack_cache_fn is None:
        raise RuntimeError("MLX stack ops are required to create runtime state.")

    stack_cache = init_llama_stack_cache_fn(config, B, max_seq_len, cache_layout=runtime_config.cache_layout, dtype=dtype)

    prefix_cache: InMemoryPrefixCache | None = None
    if runtime_config.use_prefix_cache:
        prefix_cache = InMemoryPrefixCache(max_size=64)

    residency_map: KVResidencyMap | None = None
    offload_store: InMemoryKVOffloadStore | None = None
    if runtime_config.use_kv_offload:
        policy = runtime_config.offload_policy
        if policy is None:
            policy = KVOffloadPolicyConfig().validate()
        residency_map = KVResidencyMap()
        offload_store = InMemoryKVOffloadStore()
        num_layers = getattr(config, "num_hidden_layers", 1)
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads // 2)
        head_dim = getattr(config, "head_dim", 64)
        for layer_idx in range(num_layers):
            blocks = partition_sequence_into_blocks(
                layer_idx=layer_idx,
                batch_idx=0,
                seq_len=max_seq_len,
                block_size=policy.block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=str(dtype or "float16"),
            )
            for meta in blocks:
                residency_map.add_block(meta)

    quantized_kv_cache: Any = None
    if runtime_config.use_quantized_kv:
        quantized_kv_cache = {"enabled": True, "config": runtime_config.quantized_kv_config, "data": None}

    return LongContextRuntimeState(
        stack_cache=stack_cache,
        prefix_cache=prefix_cache,
        residency_map=residency_map,
        offload_store=offload_store,
        quantized_kv_cache=quantized_kv_cache,
        position=0,
    )


def long_context_prefill(
    token_ids,
    *,
    embedding,
    stack_weights,
    state: LongContextRuntimeState,
    model_config,
    runtime_config: LongContextRuntimeConfig,
    cos,
    sin,
):
    from models.generation import GenerationConfig, ToyGenerationState

    runtime_config = runtime_config.validate()
    events: list[LongContextEvent] = []
    report = LongContextRuntimeReport(ok=True, events=events)

    gen_config = GenerationConfig(backend_preset=runtime_config.backend_preset, max_new_tokens=0)

    B = 1

    if runtime_config.use_prefix_cache and state.prefix_cache is not None:
        fp = compute_fingerprint(model_config, tokenizer=getattr(embedding, "tokenizer", None), model=None)
        logits, result_state, meta = prefill_with_prefix_reuse(
            list(token_ids),
            None,
            generation_config=gen_config,
            prefix_cache=state.prefix_cache,
            fingerprint=fp,
            state=ToyGenerationState(cache=state.stack_cache, position=state.position, generated_ids=list(token_ids)),
            max_seq_len=getattr(model_config, "max_position_embeddings", 4096),
        )
        report.prefix_cache_hit = meta.get("prefix_cache_hit", False)
        report.matched_prefix_length = meta.get("matched_length", 0)
        report.suffix_length = meta.get("suffix_length", len(token_ids))
        if report.prefix_cache_hit:
            events.append(LongContextEvent(
                kind="prefix_cache_hit",
                message=f"Prefix cache hit: matched {report.matched_prefix_length} tokens, suffix {report.suffix_length} tokens",
                metadata={"matched_length": report.matched_prefix_length, "suffix_length": report.suffix_length},
            ))
        else:
            events.append(LongContextEvent(
                kind="prefix_cache_miss",
                message=f"Prefix cache miss. Prefilled {len(token_ids)} tokens.",
                metadata={"num_tokens": len(token_ids)},
            ))
        state.stack_cache = result_state.cache
        state.position = result_state.position
    else:
        if runtime_config.use_prefix_cache:
            events.append(LongContextEvent(
                kind="fallback",
                message="Prefix cache enabled but cache is None; running full prefill.",
                metadata={},
            ))
        state.position = len(token_ids)

    if runtime_config.use_kv_offload and state.residency_map is not None and state.offload_store is not None:
        policy = runtime_config.offload_policy or KVOffloadPolicyConfig().validate()
        num_layers = getattr(model_config, "num_hidden_layers", 1)
        for layer_idx in range(num_layers):
            plan = plan_offload_blocks(
                state.residency_map,
                current_position=state.position,
                policy_config=policy,
                layer_idx=layer_idx,
                batch_idx=0,
            )
            if plan.offload:
                from ops.kv_offload_ops import extract_kv_block
                for bid in plan.offload:
                    meta = state.residency_map.get(bid)
                    if meta is None or not meta.resident:
                        continue
                    layer_cache = state.stack_cache[bid.layer_idx]
                    K_block, V_block = extract_kv_block(layer_cache, meta.start_token, meta.end_token)
                    state.offload_store.put_block(bid, K_block, V_block)
                    meta.resident = False
                    meta.offloaded = True
                    report.blocks_offloaded += 1
                    events.append(LongContextEvent(
                        kind="offload_plan",
                        message=f"Offloaded block {bid.to_string()}",
                        metadata={"block_id": bid.to_string()},
                    ))
        if report.blocks_offloaded > 0:
            events.append(LongContextEvent(
                kind="offload_plan",
                message=f"Offloaded {report.blocks_offloaded} blocks after prefill",
                metadata={"count": report.blocks_offloaded},
            ))

    if runtime_config.use_quantized_kv:
        report.quantized_kv_enabled = True
        qcfg = runtime_config.quantized_kv_config or {}
        bits = qcfg.get("bits", 8) if isinstance(qcfg, dict) else getattr(qcfg, "bits", 8)
        report.metadata["quantized_kv_bits"] = bits
        events.append(LongContextEvent(
            kind="quantized_kv_enabled",
            message=f"Quantized KV enabled (bits={bits}). Full quantized routing is scaffold.",
            metadata={"bits": bits},
        ))
        if runtime_config.use_sparse_attention:
            events.append(LongContextEvent(
                kind="warning",
                message="Quantized KV + sparse attention: combined routing is scaffolded.",
                metadata={},
            ))

    return None, state, report


def long_context_decode_step(
    token_id,
    *,
    embedding,
    stack_weights,
    state: LongContextRuntimeState,
    model_config,
    runtime_config: LongContextRuntimeConfig,
    cos,
    sin,
):
    from models.generation import GenerationConfig, ToyGenerationState

    runtime_config = runtime_config.validate()
    events: list[LongContextEvent] = []
    report = LongContextRuntimeReport(ok=True, events=events)

    new_position = state.position + 1
    needed_positions: list[int] = []

    if runtime_config.use_sparse_attention:
        from ops.long_context_ops import needed_positions_for_sparse_decode
        needed_positions = needed_positions_for_sparse_decode(
            length=new_position,
            sparse_pattern=runtime_config.sparse_pattern,
        )
        report.sparse_positions_count = len(needed_positions)
        events.append(LongContextEvent(
            kind="sparse_positions",
            message=f"Sparse decode at position {new_position}: {len(needed_positions)} needed positions",
            metadata={"position": new_position, "count": len(needed_positions)},
        ))
        if runtime_config.use_kv_offload and state.residency_map is not None and state.offload_store is not None:
            policy = runtime_config.offload_policy or KVOffloadPolicyConfig().validate()
            num_layers = getattr(model_config, "num_hidden_layers", 1)
            for layer_idx in range(num_layers):
                try:
                    from ops.long_context_ops import ensure_blocks_ready_for_attention
                    if isinstance(state.stack_cache[layer_idx], tuple):
                        layer_cache = state.stack_cache[layer_idx]
                    else:
                        layer_cache = state.stack_cache[layer_idx]
                    if isinstance(layer_cache, tuple):
                        updated_cache = ensure_blocks_ready_for_attention(
                            layer_idx=layer_idx,
                            batch_idx=0,
                            needed_positions=needed_positions,
                            residency_map=state.residency_map,
                            offload_store=state.offload_store,
                            layer_cache=layer_cache,
                            block_size=policy.block_size,
                            report=report,
                        )
                        merged = list(state.stack_cache)
                        merged[layer_idx] = updated_cache
                        state.stack_cache = merged
                except RuntimeError as e:
                    events.append(LongContextEvent(
                        kind="missing_resident_blocks",
                        message=str(e),
                        metadata={},
                    ))
                    report.ok = False
                    state.position = state.position
                    return None, state, report
        else:
            events.append(LongContextEvent(
                kind="warning",
                message="Sparse planning performed; offload disabled. Stack decode uses dense attention (scaffolded).",
                metadata={},
            ))
    else:
        events.append(LongContextEvent(
            kind="fallback",
            message="Sparse attention disabled; using full KV decode.",
            metadata={},
        ))

    if runtime_config.use_kv_offload and state.residency_map is not None:
        policy = runtime_config.offload_policy or KVOffloadPolicyConfig().validate()
        num_layers = getattr(model_config, "num_hidden_layers", 1)
        for layer_idx in range(num_layers):
            plan = plan_offload_blocks(
                state.residency_map,
                current_position=new_position,
                policy_config=policy,
                layer_idx=layer_idx,
                batch_idx=0,
            )
            if plan.offload:
                for bid in plan.offload:
                    meta = state.residency_map.get(bid)
                    if meta is None or not meta.resident:
                        continue
                    from ops.kv_offload_ops import extract_kv_block
                    layer_cache = state.stack_cache[bid.layer_idx]
                    K_block, V_block = extract_kv_block(layer_cache, meta.start_token, meta.end_token)
                    state.offload_store.put_block(bid, K_block, V_block)
                    meta.resident = False
                    meta.offloaded = True
                    report.blocks_offloaded += 1
                    events.append(LongContextEvent(
                        kind="offload_plan",
                        message=f"Offloaded block {bid.to_string()} after decode step",
                        metadata={"block_id": bid.to_string()},
                    ))

    state.position = new_position

    return None, state, report


class LongContextRuntime:
    def __init__(
        self,
        model_config,
        stack_weights,
        embedding,
        lm_head=None,
        tokenizer=None,
        runtime_config=None,
    ):
        self.model_config = model_config
        self.stack_weights = stack_weights
        self.embedding = embedding
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.runtime_config = (runtime_config or LongContextRuntimeConfig()).validate()

    def init_state(self, max_seq_len=4096) -> LongContextRuntimeState:
        return create_long_context_runtime_state(
            config=self.model_config,
            stack_weights=self.stack_weights,
            runtime_config=self.runtime_config,
            B=1,
            max_seq_len=max_seq_len,
            dtype=None,
        )

    def prefill(self, prompt_or_token_ids, state=None):
        if isinstance(prompt_or_token_ids, str):
            if self.tokenizer is None:
                raise ValueError("tokenizer required for text prompts")
            token_ids = self.tokenizer.encode(prompt_or_token_ids)
        else:
            token_ids = list(prompt_or_token_ids)
        if state is None:
            state = self.init_state()
        from .llama_config import build_rope_tables
        cos, sin = build_rope_tables(self.model_config, max_seq_len=getattr(self.model_config, "max_position_embeddings", 4096))
        logits, updated_state, report = long_context_prefill(
            token_ids,
            embedding=self.embedding,
            stack_weights=self.stack_weights,
            state=state,
            model_config=self.model_config,
            runtime_config=self.runtime_config,
            cos=cos,
            sin=sin,
        )
        return updated_state, report

    def decode_one(self, token_id, state):
        from .llama_config import build_rope_tables
        cos, sin = build_rope_tables(self.model_config, max_seq_len=getattr(self.model_config, "max_position_embeddings", 4096))
        logits, updated_state, report = long_context_decode_step(
            token_id,
            embedding=self.embedding,
            stack_weights=self.stack_weights,
            state=state,
            model_config=self.model_config,
            runtime_config=self.runtime_config,
            cos=cos,
            sin=sin,
        )
        return updated_state, report

    def generate(self, prompt, max_new_tokens=8, greedy=True):
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("tokenizer required for text prompts")
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)
        state, prefill_report = self.prefill(prompt, state=None)
        reports = [prefill_report]
        all_ids = list(token_ids)
        for _ in range(max_new_tokens):
            if not state.metadata.get("continue", True):
                break
            next_token_id = 0
            state, decode_report = self.decode_one(next_token_id, state)
            reports.append(decode_report)
            all_ids.append(next_token_id)
        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(all_ids)
        result = {
            "generated_ids": all_ids,
            "text": text,
            "reports": reports,
            "num_reports": len(reports),
            "total_errors": sum(len(r.errors()) for r in reports),
            "total_warnings": sum(len(r.warnings()) for r in reports),
            "prefix_cache_hit": any(r.prefix_cache_hit for r in reports),
            "total_prefetched": sum(r.blocks_prefetched for r in reports),
            "total_offloaded": sum(r.blocks_offloaded for r in reports),
        }
        return result

    def generate_speculative_long_context(self, prompt, max_new_tokens=8, draft_length=4, verifier_mode="sequential"):
        """
        Experimental scaffold: parallel speculative verification for long-context runtime.
        Currently delegates to TinyGenerationPipeline for the speculative path.
        Full long-context + speculative integration is future work.
        """
        from .tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig

        pipe_cfg = TinyGenerationPipelineConfig(
            hidden_size=self.model_config.hidden_size,
            intermediate_size=getattr(self.model_config, "intermediate_size", self.model_config.hidden_size * 4),
            num_attention_heads=self.model_config.num_attention_heads,
            num_key_value_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            num_hidden_layers=self.model_config.num_hidden_layers,
            max_position_embeddings=self.model_config.max_position_embeddings,
            vocab_size=self.embedding.shape[0] if hasattr(self.embedding, "shape") else 128,
            bits=4,
            group_size=32,
            dtype="float16",
            backend_preset=self.runtime_config.backend_preset,
            cache_layout=self.runtime_config.cache_layout,
            use_prefill=True,
            use_prefix_cache=self.runtime_config.use_prefix_cache,
        )
        pipe = TinyGenerationPipeline(config=pipe_cfg, tokenizer=self.tokenizer)
        return pipe.generate_speculative(
            prompt,
            max_new_tokens=max_new_tokens,
            draft_length=draft_length,
            draft_mode="fixed",
            verifier_mode=verifier_mode,
        )
