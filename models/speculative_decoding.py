from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for speculative_decoding") from exc

from models.sampling import greedy_sample


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    draft_length: int = 4
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    greedy_verify: bool = True
    seed: int | None = 0
    backend_preset: str = "fused_experimental"
    require_exact_match: bool = True
    use_prefill: bool = True
    cache_layout: str = "contiguous"

    def validate(self) -> SpeculativeConfig:
        if self.draft_length <= 0:
            raise ValueError(f"draft_length must be positive, got {self.draft_length}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive when set, got {self.top_k}")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1] when set, got {self.top_p}")
        if self.cache_layout not in ("contiguous", "paged"):
            raise ValueError(f"cache_layout must be 'contiguous' or 'paged', got {self.cache_layout}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_length": self.draft_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "greedy_verify": self.greedy_verify,
            "seed": self.seed,
            "backend_preset": self.backend_preset,
            "require_exact_match": self.require_exact_match,
            "use_prefill": self.use_prefill,
            "cache_layout": self.cache_layout,
        }


# ---------------------------------------------------------------------------
# Draft proposal
# ---------------------------------------------------------------------------

@dataclass
class DraftProposal:
    token_ids: list[int]
    logits: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def length(self) -> int:
        return len(self.token_ids)


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    proposed_token_ids: list[int]
    target_token_ids: list[int]
    accept_mask: list[bool]
    accepted_count: int
    rejected_count: int
    replacement_token_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepted_tokens(self) -> list[int]:
        return [tid for tid, acc in zip(self.proposed_token_ids, self.accept_mask) if acc]

    def rejected_tokens(self) -> list[int]:
        return [tid for tid, acc in zip(self.proposed_token_ids, self.accept_mask) if not acc]

    def all_committed_tokens(self) -> list[int]:
        committed = self.accepted_tokens()
        if self.replacement_token_id is not None and self.rejected_count > 0:
            committed.append(self.replacement_token_id)
        return committed


# ---------------------------------------------------------------------------
# Speculative step result
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeStepResult:
    proposal: DraftProposal
    verification: VerificationResult
    committed_token_ids: list[int]
    accepted_count: int
    cache_committed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Speculative generation result
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeGenerationResult:
    prompt: str | None
    prompt_ids: list[int]
    generated_ids: list[int]
    all_ids: list[int]
    text: str | None
    steps: list[SpeculativeStepResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def acceptance_rate(self) -> float:
        total_proposed = sum(s.proposal.length() for s in self.steps)
        total_accepted = sum(s.accepted_count for s in self.steps)
        if total_proposed == 0:
            return 0.0
        return total_accepted / total_proposed

    def tokens_per_step(self) -> float:
        if not self.steps:
            return 0.0
        total_committed = sum(len(s.committed_token_ids) for s in self.steps)
        return total_committed / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "all_ids": list(self.all_ids),
            "text": self.text,
            "num_steps": len(self.steps),
            "acceptance_rate": self.acceptance_rate(),
            "tokens_per_step": self.tokens_per_step(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Accept / reject logic
# ---------------------------------------------------------------------------

def compute_accept_mask(
    proposed_token_ids: list[int],
    target_token_ids: list[int],
    *,
    require_exact_match: bool = True,
) -> list[bool]:
    if not proposed_token_ids:
        return []
    n = min(len(proposed_token_ids), len(target_token_ids))
    mask: list[bool] = []
    for i in range(n):
        match = proposed_token_ids[i] == target_token_ids[i]
        if require_exact_match:
            if match and (i == 0 or mask[-1]):
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(match)
    while len(mask) < len(proposed_token_ids):
        mask.append(False)
    return mask


def accepted_prefix_length(accept_mask: list[bool]) -> int:
    count = 0
    for val in accept_mask:
        if val:
            count += 1
        else:
            break
    return count


def verify_draft_tokens(
    proposed_token_ids: list[int],
    target_token_ids: list[int],
    *,
    replacement_token_id: int | None = None,
    require_exact_match: bool = True,
) -> VerificationResult:
    if not proposed_token_ids:
        return VerificationResult(
            proposed_token_ids=[],
            target_token_ids=target_token_ids,
            accept_mask=[],
            accepted_count=0,
            rejected_count=0,
        )
    accept_mask = compute_accept_mask(proposed_token_ids, target_token_ids, require_exact_match=require_exact_match)
    accepted = accepted_prefix_length(accept_mask)
    rejected = len(proposed_token_ids) - accepted
    if replacement_token_id is None and rejected > 0 and accepted < len(target_token_ids):
        replacement_token_id = int(target_token_ids[accepted])
    return VerificationResult(
        proposed_token_ids=list(proposed_token_ids),
        target_token_ids=list(target_token_ids),
        accept_mask=accept_mask,
        accepted_count=accepted,
        rejected_count=rejected,
        replacement_token_id=replacement_token_id,
    )


# ---------------------------------------------------------------------------
# Draft proposer interface and implementations
# ---------------------------------------------------------------------------

class DraftProposer:
    def propose(self, context_ids: list[int], max_tokens: int, *, seed=None) -> DraftProposal:
        raise NotImplementedError


class GreedySelfDraftProposer(DraftProposer):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._vocab_size = pipeline.vocab_size

    def propose(self, context_ids: list[int], max_tokens: int, *, seed=None) -> DraftProposal:
        _ = seed
        if max_tokens <= 0:
            return DraftProposal(token_ids=[])
        from models.generation import GenerationConfig
        gen_cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            backend_preset=self.pipeline.config.backend_preset,
        )
        all_ids = self.pipeline.generate_ids(
            list(context_ids),
            generation_config=gen_cfg,
            validate_alignment=False,
        )
        draft_ids = all_ids[len(context_ids):len(context_ids) + max_tokens]
        return DraftProposal(
            token_ids=draft_ids,
            metadata={"draft_mode": "self", "draft_length": len(draft_ids)},
        )


class RandomDraftProposer(DraftProposer):
    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self._seed = seed

    def propose(self, context_ids: list[int], max_tokens: int, *, seed=None) -> DraftProposal:
        _ = context_ids
        if max_tokens <= 0:
            return DraftProposal(token_ids=[])
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        ids = [int(rng.integers(0, self.vocab_size)) for _ in range(max_tokens)]
        return DraftProposal(
            token_ids=ids,
            metadata={"draft_mode": "random", "seed": seed if seed is not None else self._seed},
        )


class FixedDraftProposer(DraftProposer):
    def __init__(self, fixed_ids: list[int]):
        self.fixed_ids = list(fixed_ids)

    def propose(self, context_ids: list[int], max_tokens: int, *, seed=None) -> DraftProposal:
        _ = context_ids, seed
        ids = list(self.fixed_ids[:max_tokens])
        return DraftProposal(
            token_ids=ids,
            metadata={"draft_mode": "fixed", "draft_length": len(ids)},
        )


# ---------------------------------------------------------------------------
# Target verifier interface and implementation
# ---------------------------------------------------------------------------

class TargetVerifier:
    def verify(self, context_ids, proposed_token_ids, *, state=None, config=None) -> VerificationResult:
        raise NotImplementedError


class PipelineTargetVerifier(TargetVerifier):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def verify(self, context_ids, proposed_token_ids, *, state=None, config=None) -> VerificationResult:
        _ = state
        speculative_config = (config or SpeculativeConfig()).validate()
        if not proposed_token_ids:
            return VerificationResult(
                proposed_token_ids=[],
                target_token_ids=[],
                accept_mask=[],
                accepted_count=0,
                rejected_count=0,
            )
        from models.generation import GenerationConfig
        greedy_cfg = GenerationConfig(
            max_new_tokens=len(proposed_token_ids),
            temperature=1.0,
            top_k=None,
            top_p=None,
            backend_preset=speculative_config.backend_preset,
        )
        all_ids = self.pipeline.generate_ids(
            list(context_ids),
            generation_config=greedy_cfg,
            validate_alignment=False,
        )
        target_ids = all_ids[len(context_ids):len(context_ids) + len(proposed_token_ids)]
        return verify_draft_tokens(
            proposed_token_ids,
            target_ids,
            require_exact_match=speculative_config.require_exact_match,
        )


class ParallelTargetVerifier(TargetVerifier):
    def __init__(self, pipeline, verification_config=None):
        self.pipeline = pipeline
        self.verification_config = verification_config

    def verify(self, context_ids, proposed_token_ids, *, state=None, config=None) -> VerificationResult:
        _ = state
        from ops.speculative_verify_ops import (
            ParallelVerificationConfig,
            parallel_verify_tokens,
        )

        vcfg = self.verification_config or ParallelVerificationConfig()
        if config is not None:
            from models.speculative_decoding import SpeculativeConfig as SC
            if isinstance(config, SC):
                vcfg.draft_length = config.draft_length
                vcfg.backend_preset = config.backend_preset
        vcfg = vcfg.validate()

        pipeline_state = None
        stack_cache = None
        position = 0

        if hasattr(self.pipeline, "_state") and self.pipeline._state is not None:
            pipeline_state = self.pipeline._state
            stack_cache = pipeline_state.cache
            position = pipeline_state.position

        if stack_cache is None and hasattr(self.pipeline, "model"):
            embed_test = self.pipeline.model.embed_token_ids([0])
            if _is_mlx_array(embed_test):
                cos, sin = self.pipeline.model._get_rope_tables(128)
            else:
                from models.llama_config import build_rope_tables
                try:
                    cos, sin = build_rope_tables(self.pipeline.llama_config, seq_len=128)
                except Exception:
                    cos, sin = None, None
        else:
            cos, sin = None, None

        result = parallel_verify_tokens(
            context_token_ids=list(context_ids),
            proposed_token_ids=list(proposed_token_ids),
            pipeline=self.pipeline,
            verification_config=vcfg,
        )
        vr = result.to_verification_result(require_exact_match=True)
        vr.metadata["verifier"] = "parallel"
        vr.metadata["verification_path"] = result.metadata.get("verification_path", "decode_loop_staged")
        vr.metadata["draft_length"] = len(proposed_token_ids)
        vr.metadata["accepted_count"] = result.accepted_count
        return vr


def _is_mlx_array(value: Any) -> bool:
    try:
        import mlx.core as _mx
        return type(value).__module__.startswith("mlx")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Speculative generator
# ---------------------------------------------------------------------------

class SpeculativeGenerator:
    def __init__(
        self,
        pipeline,
        draft_proposer: DraftProposer | None = None,
        target_verifier: TargetVerifier | None = None,
        config: SpeculativeConfig | None = None,
    ):
        self.pipeline = pipeline
        self.draft_proposer = draft_proposer or GreedySelfDraftProposer(pipeline)
        self.target_verifier = target_verifier or PipelineTargetVerifier(pipeline)
        self.config = (config or SpeculativeConfig()).validate()

    def generate_ids(
        self,
        input_ids: list[int],
        speculative_config: SpeculativeConfig | None = None,
    ) -> SpeculativeGenerationResult:
        config = (speculative_config or self.config).validate()
        base_seed = config.seed
        input_ids = [int(tid) for tid in input_ids]
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        all_ids = list(input_ids)
        steps: list[SpeculativeStepResult] = []
        total_new = 0
        step_index = 0
        while total_new < config.max_new_tokens:
            remaining = config.max_new_tokens - total_new
            draft_k = min(config.draft_length, remaining)
            step_seed = None if base_seed is None else base_seed + step_index
            proposal = self.draft_proposer.propose(all_ids, draft_k, seed=step_seed)
            if proposal.length() == 0:
                break
            verification = self.target_verifier.verify(all_ids, proposal.token_ids, config=config)
            committed = verification.all_committed_tokens()
            budget = remaining
            actually_committed = committed[:budget]
            all_ids.extend(actually_committed)
            total_new += len(actually_committed)
            step_result = SpeculativeStepResult(
                proposal=proposal,
                verification=verification,
                committed_token_ids=list(actually_committed),
                accepted_count=verification.accepted_count,
                cache_committed=False,
                metadata={"step_index": step_index, "remaining_budget": remaining},
            )
            steps.append(step_result)
            step_index += 1
            if verification.rejected_count > 0 and verification.replacement_token_id is not None:
                pass
        generated_ids = all_ids[len(input_ids):]
        text = None
        if hasattr(self.pipeline, "decode"):
            try:
                text = self.pipeline.decode(all_ids)
            except Exception:
                text = None
        total_proposed = sum(s.proposal.length() for s in steps)
        total_accepted = sum(s.accepted_count for s in steps)
        total_committed = sum(len(s.committed_token_ids) for s in steps)
        return SpeculativeGenerationResult(
            prompt=None,
            prompt_ids=input_ids,
            generated_ids=generated_ids,
            all_ids=all_ids,
            text=text,
            steps=steps,
            metadata={
                "draft_length": config.draft_length,
                "max_new_tokens": config.max_new_tokens,
                "num_steps": len(steps),
                "total_proposed": total_proposed,
                "total_accepted": total_accepted,
                "acceptance_rate": float(total_accepted / total_proposed) if total_proposed > 0 else 0.0,
                "avg_tokens_per_step": float(total_committed / len(steps)) if steps else 0.0,
            },
        )

    def generate_text(
        self,
        prompt: str,
        speculative_config: SpeculativeConfig | None = None,
    ) -> SpeculativeGenerationResult:
        input_ids = self.pipeline.encode(prompt)
        result = self.generate_ids(input_ids, speculative_config=speculative_config)
        result.prompt = prompt
        if hasattr(self.pipeline, "decode"):
            try:
                result.text = self.pipeline.decode(result.all_ids)
            except Exception:
                pass
        return result
