from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .quantize_weights import _materialize_symmetric_zeros, _shape_tuple


@dataclass
class QuantizedLinearPackage:
    name: str
    weight: Any
    scales: Any
    zeros: Any | None
    bits: int
    group_size: int
    original_shape: tuple[int, int]

    def shapes(self) -> dict[str, tuple[int, ...] | None]:
        return {
            "weight": _shape_tuple(self.weight),
            "scales": _shape_tuple(self.scales),
            "zeros": _shape_tuple(self.zeros),
            "original_shape": self.original_shape,
        }

    def kernel_zeros(self):
        return self.zeros if self.zeros is not None else _materialize_symmetric_zeros(self.scales, self.bits)


@dataclass
class QuantizedLlamaLayerPackage:
    layer_idx: int
    input_layernorm_weight: Any
    post_attention_layernorm_weight: Any
    qkv: QuantizedLinearPackage
    o_proj: QuantizedLinearPackage
    gate_proj: QuantizedLinearPackage
    up_proj: QuantizedLinearPackage
    down_proj: QuantizedLinearPackage

    def shapes(self) -> dict[str, Any]:
        return {
            "input_layernorm_weight": _shape_tuple(self.input_layernorm_weight),
            "post_attention_layernorm_weight": _shape_tuple(self.post_attention_layernorm_weight),
            "qkv": self.qkv.shapes(),
            "o_proj": self.o_proj.shapes(),
            "gate_proj": self.gate_proj.shapes(),
            "up_proj": self.up_proj.shapes(),
            "down_proj": self.down_proj.shapes(),
        }

    @staticmethod
    def _kernel_weight_class():
        try:
            from ops.llama_layer_ops import LlamaLayerKernelWeights

            return LlamaLayerKernelWeights
        except Exception:  # noqa: BLE001
            @dataclass
            class ShapeOnlyLlamaLayerKernelWeights:
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

                def validate(self, config):
                    if self.bits not in (4, 8):
                        raise ValueError(f"bits must be 4 or 8, got {self.bits}")
                    if self.group_size <= 0:
                        raise ValueError(f"group_size must be positive, got {self.group_size}")
                    if _shape_tuple(self.input_layernorm_weight) != (config.hidden_size,):
                        raise ValueError("input_layernorm_weight has wrong shape")
                    if _shape_tuple(self.post_attention_layernorm_weight) != (config.hidden_size,):
                        raise ValueError("post_attention_layernorm_weight has wrong shape")
                    qkv_out = config.q_output_dim() + 2 * config.kv_output_dim()
                    expected_hidden_cols = (config.hidden_size + 1) // 2 if self.bits == 4 else config.hidden_size
                    expected_intermediate_cols = (config.intermediate_size + 1) // 2 if self.bits == 4 else config.intermediate_size
                    if _shape_tuple(self.qkv_w) != (qkv_out, expected_hidden_cols):
                        raise ValueError("qkv_w has wrong shape")
                    if _shape_tuple(self.o_w) != (config.hidden_size, expected_hidden_cols):
                        raise ValueError("o_w has wrong shape")
                    if _shape_tuple(self.gate_w) != (config.intermediate_size, expected_hidden_cols):
                        raise ValueError("gate_w has wrong shape")
                    if _shape_tuple(self.up_w) != (config.intermediate_size, expected_hidden_cols):
                        raise ValueError("up_w has wrong shape")
                    if _shape_tuple(self.down_w) != (config.hidden_size, expected_intermediate_cols):
                        raise ValueError("down_w has wrong shape")
                    return self

                def shapes(self) -> dict[str, tuple[int, ...] | None]:
                    out = {}
                    for name, value in self.__dict__.items():
                        out[name] = _shape_tuple(value) if value is not None else None
                    return out

            ShapeOnlyLlamaLayerKernelWeights.__name__ = "LlamaLayerKernelWeights"
            ShapeOnlyLlamaLayerKernelWeights.__qualname__ = "LlamaLayerKernelWeights"
            ShapeOnlyLlamaLayerKernelWeights.__module__ = "ops.llama_layer_ops"
            return ShapeOnlyLlamaLayerKernelWeights

    def to_kernel_weights(self, config):
        kernel_weight_cls = self._kernel_weight_class()
        weights = kernel_weight_cls(
            input_layernorm_weight=self.input_layernorm_weight,
            post_attention_layernorm_weight=self.post_attention_layernorm_weight,
            qkv_w=self.qkv.weight,
            qkv_scales=self.qkv.scales,
            qkv_zeros=self.qkv.kernel_zeros(),
            o_w=self.o_proj.weight,
            o_scales=self.o_proj.scales,
            o_zeros=self.o_proj.kernel_zeros(),
            gate_w=self.gate_proj.weight,
            gate_scales=self.gate_proj.scales,
            gate_zeros=self.gate_proj.kernel_zeros(),
            up_w=self.up_proj.weight,
            up_scales=self.up_proj.scales,
            up_zeros=self.up_proj.kernel_zeros(),
            down_w=self.down_proj.weight,
            down_scales=self.down_proj.scales,
            down_zeros=self.down_proj.kernel_zeros(),
            bits=self.qkv.bits,
            group_size=self.qkv.group_size,
        )
        return weights.validate(config)
