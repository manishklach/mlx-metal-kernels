from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .generation import GenerationConfig, ToyGenerationState, ToyLlamaStackGenerationModel, _optional_llama_stack_ops
from .llama_config import LlamaLikeConfig
from .quantized_package_io import QuantizedCheckpointPackage
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
        self.tokenizer = tokenizer or CharTokenizer()
        self.llama_config = self.config.to_llama_config()
        self.vocab_size = max(int(self.config.vocab_size), int(getattr(self.tokenizer, "vocab_size", self.config.vocab_size)))
        self.generation_config = (generation_config or GenerationConfig()).validate()
        self.generation_config.backend_preset = self.config.backend_preset
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
        self._state: ToyGenerationState | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "pipeline": "TinyGenerationPipeline",
            "backend_preset": self.config.backend_preset,
            "cache_layout": self.config.cache_layout,
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

    def generate_ids(self, input_ids: list[int], generation_config: GenerationConfig | None = None) -> list[int]:
        return self.model.generate_token_ids([int(token_id) for token_id in input_ids], (generation_config or self.generation_config).validate())

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
        all_ids = self.generate_ids(prompt_ids, generation_config=generation_config)
        generated_ids = all_ids[len(prompt_ids):]
        text = self.decode(all_ids)
        metadata = {
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "vocab_size": self.vocab_size,
            "num_layers": self.llama_config.num_hidden_layers,
            "hidden_size": self.llama_config.hidden_size,
            "backend_preset": self.config.backend_preset,
            "synthetic_weights": True,
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
