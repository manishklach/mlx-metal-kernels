from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx

from ops.activation_ops import swiglu
from ops.autotune_ops import select_backend
from ops.decode_block_ops import decode_block_from_qkv, paged_decode_block_from_qkv
from ops.fused_ops import residual_add
from ops.mlp_block_ops import quantized_mlp_block
from ops.norm_ops import rms_norm
from ops.paged_kv_ops import allocate_paged_kv_cache
from ops.quantized_decode_block_ops import paged_quantized_decode_block, quantized_decode_block
from ops.toy_transformer_ops import make_toy_layer_weights, paged_toy_transformer_decode_layer, toy_transformer_decode_layer

from .llama_config import LlamaLikeConfig, build_rope_tables
from .weight_layouts import fused_qkv_spec


@dataclass
class KernelBackendConfig:
    matvec_backend: str = "metal_tiled"
    attention_backend: str = "metal_threadgroup"
    norm_backend: str = "metal"
    activation_backend: str = "metal"
    use_autotune: bool = False


@dataclass
class LlamaLayerState:
    K_cache: object
    V_cache: object
    K_pages: object | None = None
    V_pages: object | None = None
    block_table: object | None = None


class LlamaLikeKernelAdapter:
    def __init__(self, config: LlamaLikeConfig, backend_config: KernelBackendConfig | None = None, cache_layout: str = "contiguous"):
        self.config = config.validate()
        self.backend_config = backend_config or KernelBackendConfig()
        if cache_layout not in ("contiguous", "paged"):
            raise ValueError(f"cache_layout must be 'contiguous' or 'paged', got {cache_layout}")
        self.cache_layout = cache_layout

    def validate_supported(self) -> None:
        self.config.validate()

    def init_cache(self, B: int, dtype=mx.float16):
        self.validate_supported()
        if B <= 0:
            raise ValueError("B must be positive")
        states: list[LlamaLayerState] = []
        H = self.config.num_key_value_heads
        D = self.config.head_dim
        MAX_S = self.config.max_position_embeddings
        for _ in range(self.config.num_hidden_layers):
            if self.cache_layout == "contiguous":
                K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
                V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
                states.append(LlamaLayerState(K_cache=K_cache, V_cache=V_cache))
            else:
                page_size = 16
                K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, page_size, dtype)
                states.append(LlamaLayerState(K_cache=None, V_cache=None, K_pages=K_pages, V_pages=V_pages, block_table=block_table))
        return states

    def build_rope_tables(self, seq_len: int | None = None, dtype=mx.float32):
        return build_rope_tables(self.config, seq_len=seq_len, dtype=dtype)

    def describe(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "cache_layout": self.cache_layout,
            "backend_config": asdict(self.backend_config),
            "fused_qkv_shape": fused_qkv_spec(self.config).expected_shape(),
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "kv_groups": self.config.kv_groups(),
            "fused_qkv_output_dim": self.config.fused_qkv_output_dim(),
            "mlp_shapes": {
                "gate_proj": (self.config.intermediate_size, self.config.hidden_size),
                "up_proj": (self.config.intermediate_size, self.config.hidden_size),
                "down_proj": (self.config.hidden_size, self.config.intermediate_size),
            },
            "gqa_supported": True,
            "gqa_prefill_supported": True,
        }

    def choose_backend(self, op_name: str, shape: dict, dtype) -> str:
        default = self._default_backend(op_name)
        dtype_name = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
        if self.backend_config.use_autotune:
            return select_backend(op_name, shape, dtype_name, default_backend=default)
        return default

    def decode_layer_from_fused_qkv(
        self,
        x,
        qkv,
        layer_state: LlamaLayerState,
        cos,
        sin,
        position,
        *,
        attn_norm_weight,
        ffn_norm_weight,
        o_proj,
        gate_proj,
        up_proj,
        down_proj,
        scale=None,
    ):
        self.validate_supported()
        x3d = x if x.ndim == 3 else x.reshape(x.shape[0], 1, x.shape[-1])
        if self.cache_layout == "contiguous":
            attn_out, updated_K, updated_V = decode_block_from_qkv(
                qkv,
                layer_state.K_cache,
                layer_state.V_cache,
                cos,
                sin,
                position,
                H=self.config.num_attention_heads,
                D=self.config.head_dim,
                scale=scale,
                backend="metal",
            )
        else:
            attn_out, updated_K, updated_V = paged_decode_block_from_qkv(
                qkv,
                layer_state.K_pages,
                layer_state.V_pages,
                layer_state.block_table,
                cos,
                sin,
                position,
                H=self.config.num_attention_heads,
                D=self.config.head_dim,
                scale=scale,
                backend="metal",
            )
        flat_attn = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1)
        post_attn = residual_add(_linear(flat_attn, o_proj), x3d)
        ffn_in = rms_norm(post_attn, ffn_norm_weight, eps=self.config.rms_norm_eps, backend=self.backend_config.norm_backend)
        hidden = swiglu(_linear(ffn_in, gate_proj), _linear(ffn_in, up_proj), backend=self.backend_config.activation_backend)
        out = residual_add(_linear(hidden, down_proj), post_attn)
        if self.cache_layout == "contiguous":
            return out, LlamaLayerState(K_cache=updated_K, V_cache=updated_V)
        return out, LlamaLayerState(K_cache=None, V_cache=None, K_pages=updated_K, V_pages=updated_V, block_table=layer_state.block_table)

    def decode_layer_quantized_from_fused_qkv(
        self,
        x,
        layer_state: LlamaLayerState,
        cos,
        sin,
        position,
        *,
        attn_norm_weight,
        ffn_norm_weight,
        qkv_w,
        qkv_scales,
        out_w,
        out_scales,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        bits=4,
        group_size=32,
    ):
        self.validate_supported()
        matvec_backend = self.choose_backend(
            "q4_matvec_decode" if bits == 4 else "q8_matvec_decode",
            {"B": x.shape[0], "K": self.config.hidden_size, "N": self.config.hidden_size, "group_size": group_size},
            x.dtype,
        )
        x3d = x if x.ndim == 3 else x.reshape(x.shape[0], 1, x.shape[-1])
        attn_in = rms_norm(x3d, attn_norm_weight, eps=self.config.rms_norm_eps, backend=self.backend_config.norm_backend)
        if self.cache_layout == "contiguous":
            attn_proj, updated_K, updated_V = quantized_decode_block(
                attn_in,
                qkv_w,
                qkv_scales,
                out_w,
                out_scales,
                layer_state.K_cache,
                layer_state.V_cache,
                cos,
                sin,
                position,
                bits=bits,
                group_size=group_size,
                H=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                matvec_backend=matvec_backend,
                block_backend="metal" if not self.config.is_gqa() else "reference",
            )
            post_attn = residual_add(attn_proj, x3d)
            out = self.run_quantized_mlp_block(
                post_attn,
                mx.zeros_like(post_attn),
                ffn_norm_weight,
                gate_w,
                gate_scales,
                up_w,
                up_scales,
                down_w,
                down_scales,
                bits=bits,
                group_size=group_size,
            )
            return out, LlamaLayerState(K_cache=updated_K, V_cache=updated_V)
        attn_proj, updated_K, updated_V = paged_quantized_decode_block(
            attn_in,
            qkv_w,
            qkv_scales,
            out_w,
            out_scales,
            layer_state.K_pages,
            layer_state.V_pages,
            layer_state.block_table,
            cos,
            sin,
            position,
            bits=bits,
            group_size=group_size,
            H=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            matvec_backend=matvec_backend,
            block_backend="metal" if not self.config.is_gqa() else "reference",
        )
        post_attn = residual_add(attn_proj, x3d)
        out = self.run_quantized_mlp_block(
            post_attn,
            mx.zeros_like(post_attn),
            ffn_norm_weight,
            gate_w,
            gate_scales,
            up_w,
            up_scales,
            down_w,
            down_scales,
            bits=bits,
            group_size=group_size,
        )
        return out, LlamaLayerState(K_cache=None, V_cache=None, K_pages=updated_K, V_pages=updated_V, block_table=layer_state.block_table)

    def run_quantized_mlp_block(
        self,
        x,
        residual,
        norm_weight,
        gate_w,
        gate_scales,
        up_w,
        up_scales,
        down_w,
        down_scales,
        *,
        gate_zeros=None,
        up_zeros=None,
        down_zeros=None,
        bits=4,
        group_size=32,
        return_intermediates=False,
    ):
        matvec_backend = self.choose_backend(
            "q4_matvec_decode" if bits == 4 else "q8_matvec_decode",
            {
                "B": x.shape[0],
                "S": x.shape[1],
                "K": self.config.hidden_size,
                "N": self.config.intermediate_size,
                "group_size": group_size,
            },
            x.dtype,
        )
        return quantized_mlp_block(
            x,
            residual,
            norm_weight,
            gate_w,
            gate_scales,
            up_w,
            up_scales,
            down_w,
            down_scales,
            gate_zeros=gate_zeros,
            up_zeros=up_zeros,
            down_zeros=down_zeros,
            bits=bits,
            group_size=group_size,
            eps=self.config.rms_norm_eps,
            norm_backend=self.backend_config.norm_backend,
            matvec_backend=matvec_backend,
            activation_backend=self.backend_config.activation_backend,
            residual_backend="metal",
            return_intermediates=return_intermediates,
        )

    def make_demo_quantized_weights(self, *, bits: int = 4, group_size: int = 32):
        self.validate_supported()
        return make_toy_layer_weights(
            self.config.hidden_size,
            self.config.intermediate_size,
            bits=bits,
            group_size=group_size,
            num_attention_heads=self.config.num_attention_heads,
            head_dim=self.config.head_dim,
        )

    def _default_backend(self, op_name: str) -> str:
        if op_name in ("q4_matvec_decode", "q8_matvec_decode"):
            return self.backend_config.matvec_backend
        if op_name == "fast_attention":
            return "threadgroup" if self.backend_config.attention_backend == "metal_threadgroup" else self.backend_config.attention_backend
        if op_name in ("decode_attention", "paged_decode_attention"):
            return self.backend_config.attention_backend
        return self.backend_config.attention_backend


def _linear(x, weight):
    return mx.matmul(x.astype(mx.float32), weight.astype(mx.float32).transpose(1, 0)).astype(x.dtype)
