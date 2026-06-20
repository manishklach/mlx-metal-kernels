from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .fused_ops import reference_residual_add, residual_add
from .gqa_ops import reference_gqa_qkv_split_rope
from .mlp_block_ops import quantized_mlp_block, reference_quantized_mlp_block
from .norm_ops import reference_rms_norm, rms_norm
from .paged_kv_ops import allocate_paged_kv_cache
from .quant_ops import pack_q4
from .quantized_decode_block_ops import paged_quantized_decode_block, quantized_decode_block, reference_paged_quantized_decode_block, reference_quantized_decode_block


@dataclass
class LlamaLayerBackendConfig:
    norm_backend: str = "metal"
    qkv_matvec_backend: str = "metal_tiled"
    attention_backend: str = "metal_gqa_threadgroup"
    out_matvec_backend: str = "metal_tiled"
    mlp_backend_preset: str = "fused_experimental"
    cache_backend: str = "metal"
    use_autotune: bool = False


def reference_backend_config() -> LlamaLayerBackendConfig:
    return LlamaLayerBackendConfig(
        norm_backend="reference",
        qkv_matvec_backend="reference",
        attention_backend="reference",
        out_matvec_backend="reference",
        mlp_backend_preset="reference",
        cache_backend="reference",
        use_autotune=False,
    )


def metal_backend_config() -> LlamaLayerBackendConfig:
    return LlamaLayerBackendConfig(
        norm_backend="metal",
        qkv_matvec_backend="metal",
        attention_backend="metal_gqa",
        out_matvec_backend="metal",
        mlp_backend_preset="metal",
        cache_backend="metal",
        use_autotune=False,
    )


def tiled_backend_config() -> LlamaLayerBackendConfig:
    return LlamaLayerBackendConfig(
        norm_backend="metal",
        qkv_matvec_backend="metal_tiled",
        attention_backend="metal_gqa_threadgroup",
        out_matvec_backend="metal_tiled",
        mlp_backend_preset="tiled",
        cache_backend="metal",
        use_autotune=False,
    )


def fused_experimental_backend_config() -> LlamaLayerBackendConfig:
    return LlamaLayerBackendConfig(
        norm_backend="metal",
        qkv_matvec_backend="metal_tiled",
        attention_backend="metal_gqa_threadgroup",
        out_matvec_backend="metal_tiled",
        mlp_backend_preset="fused_experimental",
        cache_backend="metal",
        use_autotune=False,
    )


def _preset_to_backend_config(name: str) -> LlamaLayerBackendConfig:
    mapping = {
        "reference": reference_backend_config,
        "metal": metal_backend_config,
        "tiled": tiled_backend_config,
        "fused_experimental": fused_experimental_backend_config,
    }
    if name not in mapping:
        raise ValueError(f"backend_preset must be one of {tuple(mapping)}, got {name}")
    return mapping[name]()


@dataclass
class LlamaLayerKernelWeights:
    input_layernorm_weight: Any
    post_attention_layernorm_weight: Any
    qkv_w: Any
    qkv_scales: Any | None = None
    qkv_zeros: Any | None = None
    o_w: Any | None = None
    o_scales: Any | None = None
    o_zeros: Any | None = None
    gate_w: Any | None = None
    gate_scales: Any | None = None
    gate_zeros: Any | None = None
    up_w: Any | None = None
    up_scales: Any | None = None
    up_zeros: Any | None = None
    down_w: Any | None = None
    down_scales: Any | None = None
    down_zeros: Any | None = None
    bits: int = 4
    group_size: int = 32

    def validate(self, config) -> "LlamaLayerKernelWeights":
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if getattr(self.input_layernorm_weight, "shape", None) != (config.hidden_size,):
            raise ValueError(f"input_layernorm_weight must have shape {(config.hidden_size,)}, got {getattr(self.input_layernorm_weight, 'shape', None)}")
        if getattr(self.post_attention_layernorm_weight, "shape", None) != (config.hidden_size,):
            raise ValueError(f"post_attention_layernorm_weight must have shape {(config.hidden_size,)}, got {getattr(self.post_attention_layernorm_weight, 'shape', None)}")
        qkv_out = config.q_output_dim() + 2 * config.kv_output_dim()
        expected_qkv_cols = (config.hidden_size + 1) // 2 if self.bits == 4 else config.hidden_size
        if getattr(self.qkv_w, "shape", None) != (qkv_out, expected_qkv_cols):
            raise ValueError(f"qkv_w must have shape {(qkv_out, expected_qkv_cols)}, got {getattr(self.qkv_w, 'shape', None)}")
        expected_out_cols = (config.hidden_size + 1) // 2 if self.bits == 4 else config.hidden_size
        if getattr(self.o_w, "shape", None) != (config.hidden_size, expected_out_cols):
            raise ValueError(f"o_w must have shape {(config.hidden_size, expected_out_cols)}, got {getattr(self.o_w, 'shape', None)}")
        expected_mlp_cols = (config.hidden_size + 1) // 2 if self.bits == 4 else config.hidden_size
        expected_down_cols = (config.intermediate_size + 1) // 2 if self.bits == 4 else config.intermediate_size
        if getattr(self.gate_w, "shape", None) != (config.intermediate_size, expected_mlp_cols):
            raise ValueError(f"gate_w must have shape {(config.intermediate_size, expected_mlp_cols)}, got {getattr(self.gate_w, 'shape', None)}")
        if getattr(self.up_w, "shape", None) != (config.intermediate_size, expected_mlp_cols):
            raise ValueError(f"up_w must have shape {(config.intermediate_size, expected_mlp_cols)}, got {getattr(self.up_w, 'shape', None)}")
        if getattr(self.down_w, "shape", None) != (config.hidden_size, expected_down_cols):
            raise ValueError(f"down_w must have shape {(config.hidden_size, expected_down_cols)}, got {getattr(self.down_w, 'shape', None)}")
        return self

    def shapes(self) -> dict[str, tuple[int, ...] | None]:
        out = {}
        for name, value in self.__dict__.items():
            if value is None:
                out[name] = None
                continue
            shape = getattr(value, "shape", None)
            out[name] = tuple(int(dim) for dim in shape) if shape is not None else None
        return out


def _zero_like_hidden(x):
    return mx.zeros_like(x)


def _normalize_decode_input(x: mx.array, hidden_size: int) -> mx.array:
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != hidden_size:
        raise ValueError(f"x must have shape [B,1,{hidden_size}], got {x.shape}")
    return x


def _validate_cache(cache, config, cache_layout: str):
    if cache_layout == "contiguous":
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError("contiguous cache must be a tuple (K_cache, V_cache)")
        K_cache, V_cache = cache
        expected = (K_cache.shape[0], K_cache.shape[1], config.num_key_value_heads, config.head_dim)
        if K_cache.shape != V_cache.shape:
            raise ValueError(f"K_cache and V_cache must match, got {K_cache.shape}, {V_cache.shape}")
        if K_cache.shape[2] != config.num_key_value_heads or K_cache.shape[3] != config.head_dim:
            raise ValueError(f"cache must have Hkv/head_dim {(config.num_key_value_heads, config.head_dim)}, got {K_cache.shape}")
        return K_cache, V_cache
    if cache_layout == "paged":
        if not isinstance(cache, tuple) or len(cache) != 3:
            raise ValueError("paged cache must be a tuple (K_pages, V_pages, block_table)")
        return cache
    raise ValueError(f"cache_layout must be 'contiguous' or 'paged', got {cache_layout}")


def reference_llama_layer_decode_step(
    x,
    weights: LlamaLayerKernelWeights,
    cache,
    cos,
    sin,
    position,
    config,
    *,
    return_intermediates: bool = False,
    cache_layout: str = "contiguous",
):
    return llama_layer_decode_step(
        x,
        weights,
        cache,
        cos,
        sin,
        position,
        config,
        backend_config=reference_backend_config(),
        cache_layout=cache_layout,
        return_intermediates=return_intermediates,
    )


def llama_layer_decode_step(
    x,
    weights: LlamaLayerKernelWeights,
    cache,
    cos,
    sin,
    position,
    config,
    *,
    backend_config: LlamaLayerBackendConfig | None = None,
    cache_layout: str = "contiguous",
    return_intermediates: bool = False,
):
    config = config.validate()
    weights.validate(config)
    backend_config = backend_config or fused_experimental_backend_config()
    x3d = _normalize_decode_input(x, config.hidden_size)
    if cache_layout == "contiguous":
        K_cache, V_cache = _validate_cache(cache, config, cache_layout)
    else:
        K_cache, V_cache, block_table = _validate_cache(cache, config, cache_layout)

    x_norm = (
        reference_rms_norm(x3d, weights.input_layernorm_weight, eps=config.rms_norm_eps)
        if backend_config.norm_backend == "reference"
        else rms_norm(x3d, weights.input_layernorm_weight, eps=config.rms_norm_eps, backend=backend_config.norm_backend)
    )

    if cache_layout == "contiguous":
        if backend_config.qkv_matvec_backend == "reference" and backend_config.attention_backend == "reference" and backend_config.out_matvec_backend == "reference":
            attn_proj, updated_K, updated_V, qkv, attn_out = reference_quantized_decode_block(
                x_norm,
                weights.qkv_w,
                weights.qkv_scales,
                weights.o_w,
                weights.o_scales,
                K_cache,
                V_cache,
                cos,
                sin,
                position,
                qkv_zeros=weights.qkv_zeros,
                out_zeros=weights.o_zeros,
                bits=weights.bits,
                group_size=weights.group_size,
                H=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                return_intermediates=True,
            )
        else:
            attn_proj, updated_K, updated_V, qkv, attn_out = quantized_decode_block(
                x_norm,
                weights.qkv_w,
                weights.qkv_scales,
                weights.o_w,
                weights.o_scales,
                K_cache,
                V_cache,
                cos,
                sin,
                position,
                qkv_zeros=weights.qkv_zeros,
                out_zeros=weights.o_zeros,
                bits=weights.bits,
                group_size=weights.group_size,
                H=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                matvec_backend=backend_config.qkv_matvec_backend,
                block_backend=backend_config.attention_backend,
                return_intermediates=True,
            )
        updated_cache = (updated_K, updated_V)
    else:
        if backend_config.qkv_matvec_backend == "reference" and backend_config.attention_backend == "reference" and backend_config.out_matvec_backend == "reference":
            attn_proj, updated_K, updated_V, qkv, attn_out = reference_paged_quantized_decode_block(
                x_norm,
                weights.qkv_w,
                weights.qkv_scales,
                weights.o_w,
                weights.o_scales,
                K_cache,
                V_cache,
                block_table,
                cos,
                sin,
                position,
                qkv_zeros=weights.qkv_zeros,
                out_zeros=weights.o_zeros,
                bits=weights.bits,
                group_size=weights.group_size,
                H=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                return_intermediates=True,
            )
        else:
            attn_proj, updated_K, updated_V, qkv, attn_out = paged_quantized_decode_block(
                x_norm,
                weights.qkv_w,
                weights.qkv_scales,
                weights.o_w,
                weights.o_scales,
                K_cache,
                V_cache,
                block_table,
                cos,
                sin,
                position,
                qkv_zeros=weights.qkv_zeros,
                out_zeros=weights.o_zeros,
                bits=weights.bits,
                group_size=weights.group_size,
                H=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                matvec_backend=backend_config.qkv_matvec_backend,
                block_backend=backend_config.attention_backend,
                return_intermediates=True,
            )
        updated_cache = (updated_K, updated_V, block_table)

    h = (
        reference_residual_add(x3d, attn_proj)
        if backend_config.cache_backend == "reference"
        else residual_add(x3d, attn_proj, backend="metal")
    )
    if backend_config.mlp_backend_preset == "reference":
        out, mlp_intermediates = reference_quantized_mlp_block(
            h,
            _zero_like_hidden(h),
            weights.post_attention_layernorm_weight,
            weights.gate_w,
            weights.gate_scales,
            weights.up_w,
            weights.up_scales,
            weights.down_w,
            weights.down_scales,
            gate_zeros=weights.gate_zeros,
            up_zeros=weights.up_zeros,
            down_zeros=weights.down_zeros,
            bits=weights.bits,
            group_size=weights.group_size,
            eps=config.rms_norm_eps,
            return_intermediates=True,
        )
    else:
        out, mlp_intermediates = quantized_mlp_block(
            h,
            _zero_like_hidden(h),
            weights.post_attention_layernorm_weight,
            weights.gate_w,
            weights.gate_scales,
            weights.up_w,
            weights.up_scales,
            weights.down_w,
            weights.down_scales,
            gate_zeros=weights.gate_zeros,
            up_zeros=weights.up_zeros,
            down_zeros=weights.down_zeros,
            bits=weights.bits,
            group_size=weights.group_size,
            eps=config.rms_norm_eps,
            backend_preset=backend_config.mlp_backend_preset,
            return_intermediates=True,
        )

    if not return_intermediates:
        return out, updated_cache
    q, k, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        position_offset=position,
    )
    intermediates = {
        "x_norm": x_norm,
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "attn_out": attn_out,
        "attn_proj": attn_proj,
        "h": h,
        "mlp_norm": mlp_intermediates["normed"],
        "gate": mlp_intermediates["gate"],
        "up": mlp_intermediates["up"],
        "mlp": mlp_intermediates["mlp"],
        "down": mlp_intermediates["down"],
    }
    return out, updated_cache, intermediates


def llama_layer_decode_loop(
    inputs,
    weights,
    cache,
    cos,
    sin,
    config,
    *,
    backend_preset: str = "fused_experimental",
    cache_layout: str = "contiguous",
    T: int | None = None,
    return_final_cache: bool = True,
):
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [B,T,hidden_size], got {inputs.shape}")
    T = inputs.shape[1] if T is None else T
    backend_config = _preset_to_backend_config(backend_preset)
    outputs = []
    running_cache = cache
    for t in range(T):
        out_t, running_cache = llama_layer_decode_step(
            inputs[:, t:t + 1, :],
            weights,
            running_cache,
            cos,
            sin,
            t,
            config,
            backend_config=backend_config,
            cache_layout=cache_layout,
            return_intermediates=False,
        )
        outputs.append(out_t)
    out = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
    if return_final_cache:
        return out, running_cache
    return out


def create_random_quantized_llama_layer_weights(
    config,
    *,
    bits: int = 4,
    group_size: int = 32,
    dtype=mx.float16,
    seed: int = 0,
):
    config = config.validate()
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    mx.random.seed(seed)
    groups_hidden = (config.hidden_size + group_size - 1) // group_size
    groups_intermediate = (config.intermediate_size + group_size - 1) // group_size
    qkv_out = config.q_output_dim() + 2 * config.kv_output_dim()
    q_range = 16 if bits == 4 else 255

    def _q(shape):
        return (mx.random.uniform(shape) * q_range).astype(mx.uint8)

    def _pack(q):
        return pack_q4(q) if bits == 4 else q

    return LlamaLayerKernelWeights(
        input_layernorm_weight=mx.ones((config.hidden_size,), dtype=dtype),
        post_attention_layernorm_weight=mx.ones((config.hidden_size,), dtype=dtype),
        qkv_w=_pack(_q((qkv_out, config.hidden_size))),
        qkv_scales=mx.random.normal((qkv_out, groups_hidden)).astype(mx.float32),
        o_w=_pack(_q((config.hidden_size, config.hidden_size))),
        o_scales=mx.random.normal((config.hidden_size, groups_hidden)).astype(mx.float32),
        gate_w=_pack(_q((config.intermediate_size, config.hidden_size))),
        gate_scales=mx.random.normal((config.intermediate_size, groups_hidden)).astype(mx.float32),
        up_w=_pack(_q((config.intermediate_size, config.hidden_size))),
        up_scales=mx.random.normal((config.intermediate_size, groups_hidden)).astype(mx.float32),
        down_w=_pack(_q((config.hidden_size, config.intermediate_size))),
        down_scales=mx.random.normal((config.hidden_size, groups_intermediate)).astype(mx.float32),
        bits=bits,
        group_size=group_size,
    ).validate(config)


def init_llama_layer_cache(config, B: int, max_seq_len: int, *, cache_layout: str = "contiguous", dtype=mx.float16, page_size: int = 16):
    config = config.validate()
    if cache_layout == "contiguous":
        return (
            mx.zeros((B, max_seq_len, config.num_key_value_heads, config.head_dim), dtype=dtype),
            mx.zeros((B, max_seq_len, config.num_key_value_heads, config.head_dim), dtype=dtype),
        )
    if cache_layout == "paged":
        return (*allocate_paged_kv_cache(B, max_seq_len, config.num_key_value_heads, config.head_dim, page_size, dtype),)
    raise ValueError(f"cache_layout must be 'contiguous' or 'paged', got {cache_layout}")
