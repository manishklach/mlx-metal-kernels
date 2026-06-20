from __future__ import annotations

import math
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional path
    mx = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for the generation scaffold") from exc

from .llama_config import LlamaLikeConfig, tiny_gqa_debug_config
from .quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage
from .quantize_weights import QuantizationConfig, dequantize_quantized_weight, quantize_weight_groupwise
from .sampling import apply_repetition_penalty, greedy_sample, sample_logits, softmax
from .tokenization import CharTokenizer


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:  # noqa: BLE001
        if hasattr(value, "tolist"):
            return np.asarray(value.tolist())
        raise


def _cast_like(value: np.ndarray, template: Any):
    if _is_mlx_array(template):
        return mx.array(value)
    return value


def _optional_llama_ops():
    try:
        from ops.llama_layer_ops import (
            create_random_quantized_llama_layer_weights,
            fused_experimental_backend_config,
            init_llama_layer_cache,
            llama_layer_decode_step,
            metal_backend_config,
            reference_backend_config,
            tiled_backend_config,
        )

        return {
            "create_random_quantized_llama_layer_weights": create_random_quantized_llama_layer_weights,
            "fused_experimental_backend_config": fused_experimental_backend_config,
            "init_llama_layer_cache": init_llama_layer_cache,
            "llama_layer_decode_step": llama_layer_decode_step,
            "metal_backend_config": metal_backend_config,
            "reference_backend_config": reference_backend_config,
            "tiled_backend_config": tiled_backend_config,
        }
    except Exception:  # noqa: BLE001
        return None


def _optional_llama_stack_ops():
    try:
        import ops.llama_stack_ops as stack_module
        from ops.llama_stack_ops import (
            LlamaStackCache,
            create_random_quantized_llama_stack_weights,
            init_llama_stack_cache,
            llama_stack_decode_step,
        )

        return {
            "LlamaStackCache": LlamaStackCache,
            "create_random_quantized_llama_stack_weights": create_random_quantized_llama_stack_weights,
            "init_llama_stack_cache": init_llama_stack_cache,
            "llama_stack_decode_step": llama_stack_decode_step,
            "module": stack_module,
        }
    except Exception:  # noqa: BLE001
        pass
    try:
        stack_path = Path(__file__).resolve().parents[1] / "ops" / "llama_stack_ops.py"
        spec = importlib.util.spec_from_file_location("llama_stack_ops_fallback", stack_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return {
            "LlamaStackCache": module.LlamaStackCache,
            "create_random_quantized_llama_stack_weights": module.create_random_quantized_llama_stack_weights,
            "init_llama_stack_cache": module.init_llama_stack_cache,
            "llama_stack_decode_step": module.llama_stack_decode_step,
            "module": module,
        }
    except Exception:  # noqa: BLE001
        return None


def _backend_config_from_preset(ops, preset: str):
    if ops is None:
        return None
    mapping = {
        "reference": ops["reference_backend_config"],
        "metal": ops["metal_backend_config"],
        "tiled": ops["tiled_backend_config"],
        "fused_experimental": ops["fused_experimental_backend_config"],
    }
    if preset not in mapping:
        raise ValueError(f"backend_preset must be one of {tuple(mapping)}, got {preset}")
    return mapping[preset]()


def _build_rope_tables_numpy(config: LlamaLikeConfig, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if config.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {config.head_dim}")
    positions = np.arange(seq_len, dtype=np.float32)
    inv_freq = 1.0 / (float(config.rope_theta) ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / float(config.head_dim)))
    freqs = positions[:, None] * inv_freq[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _rms_norm_numpy(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _silu_numpy(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@dataclass
class GenerationConfig:
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    eos_token_id: int | None = None
    seed: int | None = None
    backend_preset: str = "fused_experimental"

    def validate(self) -> "GenerationConfig":
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive when provided, got {self.top_k}")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
        return self


@dataclass
class ToyGenerationState:
    cache: Any
    position: int
    generated_ids: list[int]


class ToyLlamaGenerationModel:
    """Single-layer generation scaffold for plumbing, not language quality."""

    def __init__(
        self,
        config,
        layer_weights,
        embedding,
        lm_head,
        *,
        tokenizer=None,
        backend_config=None,
        cache_layout="contiguous",
        dtype=None,
    ):
        self.config = config.validate()
        self.layer_weights = layer_weights
        self.embedding = embedding
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.backend_config = backend_config
        self.cache_layout = cache_layout
        self.dtype = dtype
        self._ops = _optional_llama_ops()
        self._supports_mlx_decode = self._ops is not None and mx is not None and _is_mlx_array(self.embedding)
        if cache_layout != "contiguous" and not self._supports_mlx_decode:
            raise NotImplementedError("The numpy fallback generation scaffold currently supports only contiguous cache layout")
        if self.embedding.shape[1] != self.config.hidden_size:
            raise ValueError(f"embedding must have shape [vocab_size,{self.config.hidden_size}], got {self.embedding.shape}")
        if self.lm_head.shape[0] != self.embedding.shape[0] or self.lm_head.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"lm_head must have shape [{self.embedding.shape[0]},{self.config.hidden_size}] using the documented [vocab_size, hidden_size] convention, got {self.lm_head.shape}"
            )
        self.vocab_size = int(self.embedding.shape[0])
        self._rope_cache: dict[int, tuple[Any, Any]] = {}
        self._dequant_cache: dict[str, np.ndarray] | None = None

    def _get_rope_tables(self, seq_len: int):
        if seq_len not in self._rope_cache:
            if self._supports_mlx_decode:
                from models.llama_config import build_rope_tables

                self._rope_cache[seq_len] = build_rope_tables(self.config, seq_len=seq_len, dtype=mx.float32)
            else:
                self._rope_cache[seq_len] = _build_rope_tables_numpy(self.config, seq_len)
        return self._rope_cache[seq_len]

    def init_state(self, B: int = 1, max_seq_len: int | None = None) -> ToyGenerationState:
        if B != 1:
            raise NotImplementedError("ToyLlamaGenerationModel currently supports only B=1")
        max_seq_len = self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        if self._supports_mlx_decode:
            cache = self._ops["init_llama_layer_cache"](self.config, B, max_seq_len, cache_layout=self.cache_layout, dtype=self.dtype or mx.float16)
        else:
            cache = (
                np.zeros((B, max_seq_len, self.config.num_key_value_heads, self.config.head_dim), dtype=np.float32),
                np.zeros((B, max_seq_len, self.config.num_key_value_heads, self.config.head_dim), dtype=np.float32),
            )
        self._get_rope_tables(max_seq_len + 1)
        return ToyGenerationState(cache=cache, position=0, generated_ids=[])

    def embed_token_ids(self, token_ids):
        token_ids_np = np.atleast_1d(np.asarray(token_ids, dtype=np.int64))
        gathered = _to_numpy(self.embedding)[token_ids_np]
        out = gathered.reshape(1, token_ids_np.shape[0], self.config.hidden_size)
        return _cast_like(out.astype(np.float32, copy=False), self.embedding)

    def logits_from_hidden(self, hidden):
        hidden_np = _to_numpy(hidden).astype(np.float32, copy=False)
        if hidden_np.ndim != 3 or hidden_np.shape[0] != 1:
            raise ValueError(f"hidden must have shape [1,T,{self.config.hidden_size}], got {hidden_np.shape}")
        last_hidden = hidden_np[0, -1]
        logits = last_hidden @ _to_numpy(self.lm_head).astype(np.float32, copy=False).T
        return _cast_like(logits.astype(np.float32, copy=False), self.lm_head)

    def _prefill(self, input_ids: list[int], state: ToyGenerationState, generation_config: GenerationConfig | None = None):
        logits = None
        for token_id in input_ids:
            logits, state = self.decode_step(token_id, state, generation_config=generation_config)
        return logits, state

    def _numpy_weights(self) -> dict[str, np.ndarray]:
        if self._dequant_cache is not None:
            return self._dequant_cache
        weights = self.layer_weights
        self._dequant_cache = {
            "input_ln": _to_numpy(weights.input_layernorm_weight).astype(np.float32, copy=False),
            "post_ln": _to_numpy(weights.post_attention_layernorm_weight).astype(np.float32, copy=False),
            "qkv": _to_numpy(dequantize_quantized_weight(weights.qkv_w, weights.qkv_scales, weights.qkv_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
            "o": _to_numpy(dequantize_quantized_weight(weights.o_w, weights.o_scales, weights.o_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
            "gate": _to_numpy(dequantize_quantized_weight(weights.gate_w, weights.gate_scales, weights.gate_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
            "up": _to_numpy(dequantize_quantized_weight(weights.up_w, weights.up_scales, weights.up_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
            "down": _to_numpy(dequantize_quantized_weight(weights.down_w, weights.down_scales, weights.down_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
        }
        return self._dequant_cache

    def _fallback_decode_hidden(self, embedded, state: ToyGenerationState):
        if state.position >= state.cache[0].shape[1]:
            raise ValueError("Generation state cache is full; increase max_seq_len")
        weights = self._numpy_weights()
        x = _to_numpy(embedded).astype(np.float32, copy=False)[0, 0]
        x_norm = _rms_norm_numpy(x, weights["input_ln"], self.config.rms_norm_eps)
        qkv = x_norm @ weights["qkv"].T
        q_rows = self.config.q_output_dim()
        kv_rows = self.config.kv_output_dim()
        q = qkv[:q_rows].reshape(self.config.num_attention_heads, self.config.head_dim)
        k = qkv[q_rows:q_rows + kv_rows].reshape(self.config.num_key_value_heads, self.config.head_dim)
        v = qkv[q_rows + kv_rows:].reshape(self.config.num_key_value_heads, self.config.head_dim)
        k_cache, v_cache = state.cache
        k_cache[0, state.position] = k
        v_cache[0, state.position] = v
        seen_k = k_cache[0, :state.position + 1]
        seen_v = v_cache[0, :state.position + 1]
        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads
        seen_k = np.repeat(seen_k, repeat_factor, axis=1)
        seen_v = np.repeat(seen_v, repeat_factor, axis=1)
        scores = np.einsum("hd,shd->hs", q, seen_k) / math.sqrt(float(self.config.head_dim))
        probs = _to_numpy(softmax(scores, axis=-1)).astype(np.float32, copy=False)
        attn = np.einsum("hs,shd->hd", probs, seen_v).reshape(self.config.hidden_size)
        attn_proj = attn @ weights["o"].T
        h = x + attn_proj
        mlp_norm = _rms_norm_numpy(h, weights["post_ln"], self.config.rms_norm_eps)
        gate = mlp_norm @ weights["gate"].T
        up = mlp_norm @ weights["up"].T
        mlp = _silu_numpy(gate) * up
        down = mlp @ weights["down"].T
        out = h + down
        return out.reshape(1, 1, self.config.hidden_size).astype(np.float32, copy=False)

    def decode_step(self, token_id, state: ToyGenerationState, generation_config: GenerationConfig | None = None):
        generation_config = (generation_config or GenerationConfig()).validate()
        embedded = self.embed_token_ids([int(token_id)])
        if self._supports_mlx_decode:
            cos, sin = self._get_rope_tables(state.cache[0].shape[1] + 1)
            backend_config = self.backend_config or _backend_config_from_preset(self._ops, generation_config.backend_preset)
            hidden, updated_cache = self._ops["llama_layer_decode_step"](
                embedded,
                self.layer_weights,
                state.cache,
                cos,
                sin,
                state.position,
                self.config,
                backend_config=backend_config,
                cache_layout=self.cache_layout,
            )
            state.cache = updated_cache
        else:
            hidden = self._fallback_decode_hidden(embedded, state)
        state.position += 1
        state.generated_ids.append(int(token_id))
        logits = self.logits_from_hidden(hidden)
        return logits, state

    def generate_token_ids(self, input_ids: list[int], generation_config: GenerationConfig):
        generation_config = generation_config.validate()
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        state = self.init_state(B=1, max_seq_len=max(self.config.max_position_embeddings, len(input_ids) + generation_config.max_new_tokens + 1))
        logits, state = self._prefill([int(token_id) for token_id in input_ids], state, generation_config)
        all_ids = [int(token_id) for token_id in input_ids]
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
            if generation_config.eos_token_id is not None and next_token == generation_config.eos_token_id:
                break
            logits, state = self.decode_step(next_token, state, generation_config=generation_config)
        return all_ids

    def generate_text(self, prompt: str, generation_config: GenerationConfig):
        if self.tokenizer is None:
            raise ValueError("generate_text requires a tokenizer")
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.generate_token_ids(input_ids, generation_config)
        return self.tokenizer.decode(output_ids, stop_at_eos=True)


class ToyLlamaStackGenerationModel:
    """Multi-layer synthetic generation scaffold built on the decode stack."""

    def __init__(
        self,
        config,
        stack_weights,
        *,
        tokenizer=None,
        cache_layout="contiguous",
        dtype=None,
    ):
        self.config = config.validate()
        self.stack_weights = stack_weights
        self.tokenizer = tokenizer
        self.cache_layout = cache_layout
        self.dtype = dtype
        self.embedding = stack_weights.embedding
        self.lm_head = stack_weights.lm_head
        self.vocab_size = int(self.embedding.shape[0]) if self.embedding is not None else 0
        self._stack_ops = _optional_llama_stack_ops()
        if self._stack_ops is None:
            raise RuntimeError("The multi-layer stack scaffold could not be loaded.")

    def init_state(self, B: int = 1, max_seq_len: int | None = None) -> ToyGenerationState:
        if B != 1:
            raise NotImplementedError("ToyLlamaStackGenerationModel currently supports only B=1")
        max_seq_len = self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        cache = self._stack_ops["init_llama_stack_cache"](self.config, B, max_seq_len, cache_layout=self.cache_layout, dtype=self.dtype)
        return ToyGenerationState(cache=cache, position=0, generated_ids=[])

    def embed_token_ids(self, token_ids):
        return self._stack_ops["module"].embed_token_ids(token_ids, self.embedding)

    def logits_from_hidden(self, hidden):
        return self._stack_ops["module"].logits_from_hidden(hidden, self.lm_head)

    def decode_step(self, token_id, state: ToyGenerationState, generation_config: GenerationConfig | None = None):
        generation_config = (generation_config or GenerationConfig()).validate()
        x = self.embed_token_ids([int(token_id)])
        rope_fn = self._stack_ops["module"]._build_rope_tables_numpy
        if _is_mlx_array(x) and mx is not None:
            from models.llama_config import build_rope_tables

            cos, sin = build_rope_tables(self.config, seq_len=state.cache.max_seq_len + 1, dtype=mx.float32)
        else:
            cos, sin = rope_fn(self.config, state.cache.max_seq_len + 1)
        logits, _, updated_cache = self._stack_ops["llama_stack_decode_step"](
            x,
            self.stack_weights,
            state.cache,
            cos,
            sin,
            state.position,
            self.config,
            backend_config=self._stack_ops["module"].LlamaStackBackendConfig(
                layer_backend_preset=generation_config.backend_preset,
                cache_layout=self.cache_layout,
            ),
        )
        state.cache = updated_cache
        state.position += 1
        state.generated_ids.append(int(token_id))
        logits_np = _to_numpy(logits)
        if logits_np.ndim == 3:
            logits = logits_np[0, 0, :]
        elif logits_np.ndim == 2:
            logits = logits_np[0, :]
        else:
            logits = logits_np
        return logits, state

    def generate_token_ids(self, input_ids: list[int], generation_config: GenerationConfig):
        generation_config = generation_config.validate()
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        state = self.init_state(B=1, max_seq_len=max(self.config.max_position_embeddings, len(input_ids) + generation_config.max_new_tokens + 1))
        logits = None
        for token_id in input_ids:
            logits, state = self.decode_step(token_id, state, generation_config)
        all_ids = [int(token_id) for token_id in input_ids]
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
            if generation_config.eos_token_id is not None and next_token == generation_config.eos_token_id:
                break
            logits, state = self.decode_step(next_token, state, generation_config)
        return all_ids

    def generate_text(self, prompt: str, generation_config: GenerationConfig):
        if self.tokenizer is None:
            raise ValueError("generate_text requires a tokenizer")
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.generate_token_ids(input_ids, generation_config)
        return self.tokenizer.decode(output_ids, stop_at_eos=True)


def _synthetic_layer_package(config: LlamaLikeConfig, vocab_size: int, bits: int, group_size: int, seed: int):
    rng = np.random.default_rng(seed)
    quant_config = QuantizationConfig(bits=bits, group_size=group_size)

    def _linear_package(name: str, shape: tuple[int, int]) -> QuantizedLinearPackage:
        weight = rng.normal(size=shape).astype(np.float32)
        quantized = quantize_weight_groupwise(weight, quant_config)
        return QuantizedLinearPackage(
            name=name,
            weight=quantized.packed_weight,
            scales=quantized.scales,
            zeros=quantized.zeros,
            bits=bits,
            group_size=group_size,
            original_shape=shape,
        )

    qkv_rows = config.q_output_dim() + 2 * config.kv_output_dim()
    package = QuantizedLlamaLayerPackage(
        layer_idx=0,
        input_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
        post_attention_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
        qkv=_linear_package("qkv_fused", (qkv_rows, config.hidden_size)),
        o_proj=_linear_package("o_proj", (config.hidden_size, config.hidden_size)),
        gate_proj=_linear_package("gate_proj", (config.intermediate_size, config.hidden_size)),
        up_proj=_linear_package("up_proj", (config.intermediate_size, config.hidden_size)),
        down_proj=_linear_package("down_proj", (config.hidden_size, config.intermediate_size)),
    )
    embedding = rng.normal(size=(vocab_size, config.hidden_size)).astype(np.float32)
    lm_head = rng.normal(size=(vocab_size, config.hidden_size)).astype(np.float32)
    return package.to_kernel_weights(config), embedding, lm_head


def create_synthetic_generation_model(
    *,
    config=None,
    tokenizer=None,
    vocab_size=128,
    bits=4,
    group_size=32,
    dtype=None,
    seed=0,
    backend_preset="fused_experimental",
):
    config = (config or tiny_gqa_debug_config()).validate()
    tokenizer = tokenizer or CharTokenizer()
    vocab_size = max(int(vocab_size), int(getattr(tokenizer, "vocab_size", vocab_size)))
    ops = _optional_llama_ops()
    if ops is not None and mx is not None:
        dtype = mx.float16 if dtype is None else dtype
        weights = ops["create_random_quantized_llama_layer_weights"](config, bits=bits, group_size=group_size, dtype=dtype, seed=seed)
        mx.random.seed(seed + 1)
        embedding = mx.random.normal((vocab_size, config.hidden_size)).astype(dtype)
        lm_head = mx.random.normal((vocab_size, config.hidden_size)).astype(dtype)
        backend_config = _backend_config_from_preset(ops, backend_preset)
    else:
        weights, embedding, lm_head = _synthetic_layer_package(config, vocab_size, bits, group_size, seed)
        backend_config = None
    return ToyLlamaGenerationModel(
        config,
        weights,
        embedding,
        lm_head,
        tokenizer=tokenizer,
        backend_config=backend_config,
        cache_layout="contiguous",
        dtype=dtype,
    )


def create_synthetic_stack_generation_model(
    *,
    config=None,
    tokenizer=None,
    vocab_size=128,
    bits=4,
    group_size=32,
    dtype=None,
    seed=0,
    backend_preset="fused_experimental",
):
    config = (config or tiny_gqa_debug_config()).validate()
    tokenizer = tokenizer or CharTokenizer()
    vocab_size = max(int(vocab_size), int(getattr(tokenizer, "vocab_size", vocab_size)))
    stack_ops = _optional_llama_stack_ops()
    if stack_ops is None:
        raise RuntimeError("The multi-layer stack scaffold could not be loaded.")
    if mx is not None:
        dtype = mx.float16 if dtype is None else dtype
    stack_weights = stack_ops["create_random_quantized_llama_stack_weights"](
        config,
        vocab_size=vocab_size,
        bits=bits,
        group_size=group_size,
        dtype=dtype,
        seed=seed,
        include_embedding=True,
        include_lm_head=True,
    )
    _ = backend_preset
    return ToyLlamaStackGenerationModel(
        config,
        stack_weights,
        tokenizer=tokenizer,
        cache_layout="contiguous",
        dtype=dtype,
    )
