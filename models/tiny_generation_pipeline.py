from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .alignment import AlignmentReport, validate_generation_alignment
from .generation import (
    GenerationConfig,
    ToyGenerationState,
    ToyLlamaStackGenerationModel,
    _optional_llama_prefill_ops,
    _optional_llama_stack_ops,
)
from .llama_config import LlamaLikeConfig
from .prefix_cache import (
    InMemoryPrefixCache,
    PrefixCacheEntry,
    compute_fingerprint,
)
from .quantized_package_io import QuantizedCheckpointPackage
from .sampling import apply_repetition_penalty, greedy_sample, sample_logits
from .tokenization import CharTokenizer, TokenizerProtocol


@dataclass
class TinyGenerationPipelineConfig:
    hidden_size: int = 64
    intermediate_size: int = 128
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    num_hidden_layers: int = 2
    max_position_embeddings: int = 128
    vocab_size: int = 128
    bits: int = 4
    group_size: int = 32
    dtype: str = "float16"
    backend_preset: str = "fused_experimental"
    cache_layout: str = "contiguous"
    use_prefill: bool = True
    use_prefix_cache: bool = False

    def validate(self) -> "TinyGenerationPipelineConfig":
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim, "
                f"got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}, "
                f"head_dim={self.head_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, "
                f"got {self.num_attention_heads}, {self.num_key_value_heads}"
            )
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.cache_layout not in ("contiguous", "paged"):
            raise ValueError("cache_layout must be one of ('contiguous', 'paged')")
        if self.backend_preset not in ("reference", "metal", "tiled", "fused_experimental"):
            raise ValueError("backend_preset must be one of ('reference', 'metal', 'tiled', 'fused_experimental')")
        return self

    def to_llama_config(self) -> LlamaLikeConfig:
        self.validate()
        return LlamaLikeConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=self.vocab_size,
            model_type="tiny_generation_pipeline",
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.validate())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyGenerationPipelineConfig":
        return cls(**data).validate()


@dataclass
class GenerationResult:
    prompt: str
    prompt_ids: list[int]
    generated_ids: list[int]
    all_ids: list[int]
    text: str
    backend_preset: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "all_ids": list(self.all_ids),
            "text": self.text,
            "backend_preset": self.backend_preset,
            "metadata": dict(self.metadata),
        }


@dataclass
class PrefillResult:
    logits: Any
    cache: Any
    next_position: int
    prompt_length: int


class TinyGenerationPipeline:
    """High-level synthetic tiny-model generation pipeline for end-to-end plumbing."""

    def __init__(
        self,
        config: TinyGenerationPipelineConfig | None = None,
        tokenizer: TokenizerProtocol | None = None,
        stack_weights=None,
        generation_config: GenerationConfig | None = None,
    ):
        self.config = (config or TinyGenerationPipelineConfig()).validate()
        user_tokenizer = tokenizer
        self.tokenizer = tokenizer or CharTokenizer()
        self.llama_config = self.config.to_llama_config()
        tokenizer_vocab = int(getattr(self.tokenizer, "vocab_size", self.config.vocab_size))
        if user_tokenizer is None:
            self.vocab_size = tokenizer_vocab
        else:
            self.vocab_size = max(int(self.config.vocab_size), tokenizer_vocab)
        self.llama_config.vocab_size = self.vocab_size
        self.generation_config = (generation_config or GenerationConfig()).validate()
        self.generation_config.backend_preset = self.config.backend_preset
        self._prefill_ops = _optional_llama_prefill_ops()
        if stack_weights is None:
            stack_ops = _optional_llama_stack_ops()
            if stack_ops is None:
                raise RuntimeError("create_random_quantized_llama_stack_weights is not available")
            stack_weights = stack_ops["create_random_quantized_llama_stack_weights"](
                self.llama_config,
                vocab_size=self.vocab_size,
                bits=self.config.bits,
                group_size=self.config.group_size,
                dtype=None,
                seed=0,
                include_embedding=True,
                include_lm_head=True,
            )
        self.stack_weights = stack_weights
        self.model = ToyLlamaStackGenerationModel(
            self.llama_config,
            self.stack_weights,
            tokenizer=self.tokenizer,
            cache_layout=self.config.cache_layout,
            dtype=None,
        )
        self.prefix_cache: InMemoryPrefixCache | None = (
            InMemoryPrefixCache(max_size=64) if self.config.use_prefix_cache else None
        )
        self._state: ToyGenerationState | None = None

    def validate_alignment(self) -> AlignmentReport:
        return validate_generation_alignment(
            tokenizer=self.tokenizer,
            config=self.llama_config,
            stack_weights=self.stack_weights,
            embedding=getattr(self.stack_weights, "embedding", None),
            lm_head=getattr(self.stack_weights, "lm_head", None),
            bits=self.config.bits,
            group_size=self.config.group_size,
        )

    def describe(self) -> dict[str, Any]:
        alignment = self.validate_alignment()
        return {
            "pipeline": "TinyGenerationPipeline",
            "backend_preset": self.config.backend_preset,
            "cache_layout": self.config.cache_layout,
            "use_prefill": self.config.use_prefill,
            "use_prefix_cache": self.config.use_prefix_cache,
            "prefix_cache_size": self.prefix_cache.size if self.prefix_cache else 0,
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "hidden_size": self.llama_config.hidden_size,
            "intermediate_size": self.llama_config.intermediate_size,
            "num_attention_heads": self.llama_config.num_attention_heads,
            "num_key_value_heads": self.llama_config.num_key_value_heads,
            "head_dim": self.llama_config.head_dim,
            "num_hidden_layers": self.llama_config.num_hidden_layers,
            "max_position_embeddings": self.llama_config.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "tokenizer": type(self.tokenizer).__name__,
            "synthetic_weights": True,
            "alignment_ok": alignment.ok,
            "alignment_issue_count": len(alignment.issues),
        }

    def encode(self, prompt: str) -> list[int]:
        return [int(token_id) for token_id in self.tokenizer.encode(prompt)]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode([int(token_id) for token_id in token_ids], stop_at_eos=True)

    def _resolve_generation_config(
        self,
        generation_config: GenerationConfig | None = None,
        *,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
    ) -> GenerationConfig:
        base = generation_config or self.generation_config
        resolved = GenerationConfig(
            max_new_tokens=base.max_new_tokens if max_new_tokens is None else max_new_tokens,
            temperature=1.0 if greedy else (base.temperature if temperature is None else temperature),
            top_k=None if greedy else (base.top_k if top_k is None else top_k),
            top_p=None if greedy else (base.top_p if top_p is None else top_p),
            repetition_penalty=base.repetition_penalty,
            eos_token_id=base.eos_token_id,
            seed=base.seed if seed is None else seed,
            backend_preset=self.config.backend_preset,
        )
        if resolved.eos_token_id is None:
            resolved.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        return resolved.validate()

    def generate_ids(
        self,
        input_ids: list[int],
        generation_config: GenerationConfig | None = None,
        *,
        validate_alignment: bool = True,
        ignore_prefix_cache: bool = False,
    ) -> list[int]:
        generation_config = (generation_config or self.generation_config).validate()
        if validate_alignment:
            self.validate_alignment().raise_for_errors()
        input_ids = [int(token_id) for token_id in input_ids]
        if not self.config.use_prefill:
            return self.model.generate_token_ids(input_ids, generation_config)
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        if self.prefix_cache is not None and not ignore_prefix_cache:
            result = self._prefill_with_cache(input_ids, generation_config)
            if result is not None:
                state, logits = result
            else:
                prefill = self.prefill_prompt(input_ids, generation_config=generation_config)
                state = ToyGenerationState(cache=prefill.cache, position=prefill.next_position, generated_ids=list(input_ids))
                logits = prefill.logits
        else:
            prefill = self.prefill_prompt(input_ids, generation_config=generation_config)
            state = ToyGenerationState(cache=prefill.cache, position=prefill.next_position, generated_ids=list(input_ids))
            logits = prefill.logits
        all_ids = list(input_ids)
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        for step_idx in range(generation_config.max_new_tokens):
            working_logits = logits
            if generation_config.repetition_penalty > 1.0:
                working_logits = apply_repetition_penalty(working_logits, state.generated_ids, penalty=generation_config.repetition_penalty)
            if generation_config.temperature == 1.0 and generation_config.top_k is None and generation_config.top_p is None:
                next_token = greedy_sample(working_logits)
            else:
                sample_seed = None if generation_config.seed is None else generation_config.seed + step_idx
                next_token = sample_logits(
                    working_logits,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    seed=sample_seed,
                )
            next_token = int(next_token)
            all_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits, state = self.model.decode_step(next_token, state, generation_config=generation_config)
        return all_ids

    @staticmethod
    def _clone_stack_cache_safe(stack_cache):
        try:
            from ops.kv_cache_reuse_ops import clone_stack_cache as _csc
        except ImportError:
            return None
        return _csc(stack_cache)

    def _prefill_with_cache(
        self,
        input_ids: list[int],
        generation_config: GenerationConfig,
    ) -> tuple[ToyGenerationState, Any] | None:
        fp = compute_fingerprint(self.llama_config, self.tokenizer)
        match = self.prefix_cache.lookup(input_ids, fp)
        if match.matched:
            cache_clone = self._clone_stack_cache_safe(match.entry.stack_cache)
            if cache_clone is None:
                return None
            suffix = input_ids[match.matched_length:]
            if suffix:
                state = ToyGenerationState(
                    cache=cache_clone,
                    position=match.matched_length,
                    generated_ids=list(input_ids[:match.matched_length]),
                )
                logits = None
            else:
                replay_length = max(0, match.matched_length - 1)
                replay_state = self.reset_cache(B=1)
                try:
                    from ops.kv_cache_reuse_ops import copy_prefix_cache_into
                except ImportError:
                    return None
                replay_state.cache = copy_prefix_cache_into(match.entry.stack_cache, replay_state.cache, replay_length)
                replay_state.position = replay_length
                replay_state.generated_ids = list(input_ids[:replay_length])
                state = replay_state
                suffix = [input_ids[-1]]
                logits = None
            for token_id in suffix:
                logits, state = self.model.decode_step(token_id, state, generation_config=generation_config)
            return state, logits
        state = self.reset_cache(B=1)
        logits, updated_state = self.model.prefill_token_ids(input_ids, state, generation_config=generation_config)
        cached_cache = self._clone_stack_cache_safe(updated_state.cache)
        if cached_cache is not None:
            entry = PrefixCacheEntry(
                fingerprint=fp,
                token_ids=list(input_ids),
                stack_cache=cached_cache,
                metadata={"num_prompt_tokens": len(input_ids)},
            )
            self.prefix_cache.store(entry)
        return updated_state, logits

    def reset_cache(self, B: int = 1) -> ToyGenerationState:
        self._state = self.model.init_state(
            B=B,
            max_seq_len=max(self.llama_config.max_position_embeddings, self.generation_config.max_new_tokens + 1),
        )
        return self._state

    def step(
        self,
        token_id: int,
        state: ToyGenerationState,
        generation_config: GenerationConfig | None = None,
    ):
        return self.model.decode_step(int(token_id), state, generation_config or self.generation_config)

    def prefill_prompt(self, input_ids, state: ToyGenerationState | None = None, generation_config: GenerationConfig | None = None) -> PrefillResult:
        if self._prefill_ops is None:
            raise RuntimeError("The prefill scaffold could not be loaded.")
        token_ids = [int(token_id) for token_id in input_ids]
        if not token_ids:
            raise ValueError("input_ids must contain at least one token")
        if state is None:
            state = self.reset_cache(B=1)
        if state.position != 0:
            raise NotImplementedError("Continuation prefill with a non-zero starting state is not implemented yet.")
        logits, updated_state = self.model.prefill_token_ids(token_ids, state, generation_config=generation_config or self.generation_config)
        return PrefillResult(
            logits=logits,
            cache=updated_state.cache,
            next_position=updated_state.position,
            prompt_length=len(token_ids),
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        validate_alignment: bool = True,
    ) -> GenerationResult:
        prompt_ids = self.encode(prompt)
        generation_config = self._resolve_generation_config(
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        self.generation_config = generation_config
        if validate_alignment:
            self.validate_alignment().raise_for_errors()
        all_ids = self.generate_ids(
            prompt_ids,
            generation_config=generation_config,
            validate_alignment=False,
        )
        generated_ids = all_ids[len(prompt_ids):]
        text = self.decode(all_ids)
        metadata = {
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "vocab_size": self.vocab_size,
            "num_layers": self.llama_config.num_hidden_layers,
            "hidden_size": self.llama_config.hidden_size,
            "backend_preset": self.config.backend_preset,
            "use_prefill": self.config.use_prefill,
            "synthetic_weights": True,
            "alignment_ok": self.validate_alignment().ok,
        }
        return GenerationResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            all_ids=all_ids,
            text=text,
            backend_preset=self.config.backend_preset,
            metadata=metadata,
        )


def _coerce_package(package: QuantizedCheckpointPackage | dict[str, Any] | str | Path) -> QuantizedCheckpointPackage:
    if isinstance(package, QuantizedCheckpointPackage):
        return package
    if isinstance(package, dict):
        return QuantizedCheckpointPackage.from_dict(package)
    if isinstance(package, (str, Path)):
        return QuantizedCheckpointPackage.load_json(package)
    raise TypeError(f"Unsupported package type: {type(package)!r}")


def create_pipeline_from_quantized_package(
    package: QuantizedCheckpointPackage | dict[str, Any] | str | Path,
    *,
    tokenizer: TokenizerProtocol | None = None,
    backend_preset: str | None = None,
) -> TinyGenerationPipeline:
    package_obj = _coerce_package(package).validate(allow_partial=True)
    _ = tokenizer
    _ = backend_preset
    raise NotImplementedError(
        "Quantized package metadata does not contain tensor data. Use synthetic pipeline or future tensor-data package support."
    )
