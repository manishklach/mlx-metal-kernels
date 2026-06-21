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
    raise RuntimeError("numpy is required for speculative_verify_ops") from exc


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


# ---------------------------------------------------------------------------
# Verification config
# ---------------------------------------------------------------------------


@dataclass
class ParallelVerificationConfig:
    draft_length: int = 4
    mode: str = "greedy_exact"
    backend_preset: str = "fused_experimental"
    cache_layout: str = "contiguous"
    use_prefill_for_draft: bool = True
    return_logits: bool = True
    commit_cache: bool = False

    def validate(self) -> ParallelVerificationConfig:
        if self.draft_length <= 0:
            raise ValueError(f"draft_length must be positive, got {self.draft_length}")
        if self.mode not in ("greedy_exact",):
            raise ValueError(f"mode must be 'greedy_exact', got {self.mode!r}")
        if self.cache_layout not in ("contiguous",):
            raise NotImplementedError(
                f"cache_layout={self.cache_layout!r} is not supported; only 'contiguous' is implemented"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_length": self.draft_length,
            "mode": self.mode,
            "backend_preset": self.backend_preset,
            "cache_layout": self.cache_layout,
            "use_prefill_for_draft": self.use_prefill_for_draft,
            "return_logits": self.return_logits,
            "commit_cache": self.commit_cache,
        }


# ---------------------------------------------------------------------------
# Verification pass result
# ---------------------------------------------------------------------------


@dataclass
class ParallelVerificationPassResult:
    proposed_token_ids: list[int]
    target_token_ids: list[int]
    accept_mask: list[bool]
    accepted_count: int
    replacement_token_id: int | None = None
    logits: Any | None = None
    staged_cache: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepted_tokens(self) -> list[int]:
        return [tid for tid, acc in zip(self.proposed_token_ids, self.accept_mask) if acc]

    def committed_tokens(self) -> list[int]:
        committed = self.accepted_tokens()
        if self.replacement_token_id is not None and len(self.accept_mask) > 0 and not all(self.accept_mask):
            committed.append(self.replacement_token_id)
        return committed

    def to_verification_result(self, require_exact_match: bool = True):
        from models.speculative_decoding import verify_draft_tokens

        return verify_draft_tokens(
            self.proposed_token_ids,
            self.target_token_ids,
            replacement_token_id=self.replacement_token_id,
            require_exact_match=require_exact_match,
        )


# ---------------------------------------------------------------------------
# Proposed-token embedding helper
# ---------------------------------------------------------------------------


def embed_proposed_tokens(
    proposed_token_ids: list[int],
    embedding,
    *,
    B: int = 1,
):
    if B != 1:
        raise NotImplementedError("embed_proposed_tokens supports only B=1")
    token_ids = np.atleast_1d(np.asarray(proposed_token_ids, dtype=np.int64))
    K = token_ids.shape[0]
    if K == 0:
        raise ValueError("proposed_token_ids must be non-empty")
    from ops.llama_stack_ops import embed_token_ids as _embed

    embedded = _embed(token_ids, embedding)
    if _is_mlx_array(embedding):
        return embedded
    return embedded.reshape(B, K, -1)


# ---------------------------------------------------------------------------
# Target tokens from verification logits
# ---------------------------------------------------------------------------


def target_tokens_from_verification_logits(
    logits_list: list,
    *,
    proposed_token_ids: list[int],
    mode: str = "greedy_exact",
) -> list[int]:
    if mode != "greedy_exact":
        raise ValueError(f"mode must be 'greedy_exact', got {mode!r}")
    if not logits_list:
        return []
    target_ids: list[int] = []
    if _is_mlx_array(logits_list[0]):
        for logits in logits_list:
            token_id = int(mx.argmax(logits, axis=-1).item())
            target_ids.append(token_id)
    else:
        for logits in logits_list:
            logits_np = np.asarray(logits).ravel()
            token_id = int(np.argmax(logits_np))
            target_ids.append(token_id)
    return target_ids


# ---------------------------------------------------------------------------
# Parallel target verification pass
# ---------------------------------------------------------------------------


def _clone_for_staging(stack_cache: Any) -> Any:
    try:
        from ops.speculative_cache_ops import _clone_stack_cache as _clone
    except ImportError:
        raise RuntimeError("_clone_stack_cache not available")
    return _clone(stack_cache)


def parallel_verify_tokens(
    *,
    context_token_ids: list[int],
    proposed_token_ids: list[int],
    pipeline=None,
    model_config=None,
    stack_weights=None,
    stack_cache=None,
    embedding=None,
    lm_head=None,
    cos=None,
    sin=None,
    position=None,
    verification_config=None,
) -> ParallelVerificationPassResult:
    if pipeline is not None:
        model_config = pipeline.llama_config
        stack_weights = pipeline.stack_weights
        embedding = pipeline.model.embedding
        lm_head = pipeline.model.lm_head
        stack_cache = pipeline.model._state.cache if pipeline.model._state is not None else None

    if verification_config is None:
        verification_config = ParallelVerificationConfig().validate()
    else:
        verification_config = verification_config.validate()

    if not proposed_token_ids:
        return ParallelVerificationPassResult(
            proposed_token_ids=[],
            target_token_ids=[],
            accept_mask=[],
            accepted_count=0,
            metadata={"verification_path": "no_proposed_tokens"},
        )

    K = len(proposed_token_ids)
    if stack_cache is None:
        raise ValueError("stack_cache is required for parallel_verify_tokens")
    if position is None:
        raise ValueError("position is required for parallel_verify_tokens")

    from models.generation import GenerationConfig, ToyGenerationState

    from ops.speculative_cache_ops import _clone_stack_cache as _clone_cache

    staged_cache = _clone_cache(stack_cache)

    if verification_config.use_prefill_for_draft:
        verification_path = "decode_loop_staged"
    else:
        verification_path = "decode_loop_staged"

    from ops.llama_stack_ops import llama_stack_decode_step

    logits_list: list[Any] = []
    staged_state = ToyGenerationState(cache=staged_cache, position=position, generated_ids=list(context_token_ids))

    for i, draft_token in enumerate(proposed_token_ids):
        embedded = embed_proposed_tokens([draft_token], embedding, B=1)
        cos_i = cos
        sin_i = sin
        from ops.llama_stack_ops import LlamaStackBackendConfig

        backend_cfg = LlamaStackBackendConfig(
            layer_backend_preset=verification_config.backend_preset,
            cache_layout=verification_config.cache_layout,
        )
        logits, _, updated_cache = llama_stack_decode_step(
            embedded,
            stack_weights,
            staged_state.cache,
            cos_i,
            sin_i,
            staged_state.position,
            model_config,
            backend_config=backend_cfg,
        )
        staged_state.cache = updated_cache
        staged_state.position += 1
        staged_state.generated_ids.append(int(draft_token))

        logits_np = np.asarray(logits)
        if logits_np.ndim == 3:
            logits_list.append(logits_np[0, 0, :])
        elif logits_np.ndim == 2:
            logits_list.append(logits_np[0, :])
        else:
            logits_list.append(logits_np.ravel())

    target_token_ids = target_tokens_from_verification_logits(
        logits_list,
        proposed_token_ids=proposed_token_ids,
        mode=verification_config.mode,
    )

    from models.speculative_decoding import compute_accept_mask, accepted_prefix_length

    accept_mask = compute_accept_mask(
        proposed_token_ids,
        target_token_ids,
        require_exact_match=True,
    )
    accepted_count = accepted_prefix_length(accept_mask)
    rejected_count = K - accepted_count
    replacement_token_id: int | None = None
    if rejected_count > 0 and accepted_count < len(target_token_ids):
        replacement_token_id = int(target_token_ids[accepted_count])

    result = ParallelVerificationPassResult(
        proposed_token_ids=list(proposed_token_ids),
        target_token_ids=list(target_token_ids),
        accept_mask=list(accept_mask),
        accepted_count=accepted_count,
        replacement_token_id=replacement_token_id,
        logits=logits_list if verification_config.return_logits else None,
        staged_cache=staged_cache,
        metadata={
            "verification_path": verification_path,
            "draft_length": K,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "use_prefill_for_draft": verification_config.use_prefill_for_draft,
            "start_position": position,
        },
    )
    return result


# ---------------------------------------------------------------------------
# Staged cache commit helper
# ---------------------------------------------------------------------------


def commit_parallel_verification_cache(
    committed_cache,
    staged_cache,
    *,
    start_position: int,
    accepted_count: int,
    include_replacement: bool = False,
    replacement_position: int | None = None,
):
    if committed_cache.cache_layout == "paged":
        raise NotImplementedError("paged cache commit is not implemented yet")
    if committed_cache.cache_layout != staged_cache.cache_layout:
        raise ValueError("committed and staged cache layouts must match")

    from ops.speculative_cache_ops import commit_accepted_cache

    copy_count = accepted_count
    if include_replacement and replacement_position is not None:
        copy_count = replacement_position + 1

    return commit_accepted_cache(
        draft_cache=staged_cache,
        committed_cache=committed_cache,
        accepted_count=copy_count,
    )
