from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoint_adapter import CheckpointAdapter
from .quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage
from .quantize_weights import (
    QuantizationConfig,
    QuantizedWeight,
    dequantize_quantized_weight,
    quantization_error,
    quantize_weight_groupwise,
)


@dataclass
class QuantizationReport:
    ok: bool
    quantized_tensors: list[str]
    skipped_tensors: list[str]
    errors: list[str]
    metrics: dict[str, dict] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


class CheckpointQuantizer:
    def __init__(self, checkpoint_adapter: CheckpointAdapter, quant_config: QuantizationConfig):
        self.checkpoint_adapter = checkpoint_adapter
        self.quant_config = quant_config.validate()
        self._quantized_tensors: list[str] = []
        self._skipped_tensors: list[str] = []
        self._errors: list[str] = []
        self._metrics: dict[str, dict] = {}

    def _record_quantized(self, name: str, original, quantized: QuantizedWeight) -> QuantizedWeight:
        dequantized = dequantize_quantized_weight(
            quantized.packed_weight,
            quantized.scales,
            quantized.zeros,
            bits=quantized.bits,
            group_size=quantized.group_size,
        )
        self._metrics[name] = quantization_error(original, dequantized)
        if name not in self._quantized_tensors:
            self._quantized_tensors.append(name)
        return quantized

    def quantize_tensor(self, name: str) -> QuantizedWeight:
        try:
            tensor = self.checkpoint_adapter.load_tensor(name)
            quantized = quantize_weight_groupwise(tensor, self.quant_config)
            quantized.name = name
            return self._record_quantized(name, tensor, quantized)
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"{name}: {exc}")
            raise

    def quantize_linear_weight(self, name: str) -> QuantizedWeight:
        try:
            tensor = self.checkpoint_adapter.load_tensor(name)
            shape = getattr(tensor, "shape", None)
            if shape is None or len(shape) != 2:
                self._skipped_tensors.append(name)
                raise ValueError(f"{name} must be a rank-2 linear weight, got {shape}")
            quantized = quantize_weight_groupwise(tensor, self.quant_config)
            quantized.name = name
            return self._record_quantized(name, tensor, quantized)
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"{name}: {exc}")
            raise

    def quantize_fused_qkv_for_layer(self, layer_idx: int) -> QuantizedWeight:
        name = f"model.layers.{layer_idx}.self_attn.qkv_proj.fused_weight"
        try:
            fused = self.checkpoint_adapter.fuse_qkv_for_layer(layer_idx)
            quantized = quantize_weight_groupwise(fused, self.quant_config)
            quantized.name = name
            return self._record_quantized(name, fused, quantized)
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"{name}: {exc}")
            raise

    def _linear_package(self, name: str, quantized: QuantizedWeight) -> QuantizedLinearPackage:
        return QuantizedLinearPackage(
            name=name,
            weight=quantized.packed_weight,
            scales=quantized.scales,
            zeros=quantized.zeros,
            bits=quantized.bits,
            group_size=quantized.group_size,
            original_shape=quantized.original_shape,
        )

    def quantize_layer(self, layer_idx: int) -> QuantizedLlamaLayerPackage:
        names = self.checkpoint_adapter.layer_names(layer_idx)
        qkv = self.quantize_fused_qkv_for_layer(layer_idx)
        o_proj = self.quantize_linear_weight(names.o_proj)
        gate_proj = self.quantize_linear_weight(names.gate_proj)
        up_proj = self.quantize_linear_weight(names.up_proj)
        down_proj = self.quantize_linear_weight(names.down_proj)
        input_ln = self.checkpoint_adapter.load_tensor(names.input_layernorm)
        post_ln = self.checkpoint_adapter.load_tensor(names.post_attention_layernorm)
        self._skipped_tensors.extend([names.input_layernorm, names.post_attention_layernorm])
        return QuantizedLlamaLayerPackage(
            layer_idx=layer_idx,
            input_layernorm_weight=input_ln,
            post_attention_layernorm_weight=post_ln,
            qkv=self._linear_package("qkv_fused", qkv),
            o_proj=self._linear_package(names.o_proj, o_proj),
            gate_proj=self._linear_package(names.gate_proj, gate_proj),
            up_proj=self._linear_package(names.up_proj, up_proj),
            down_proj=self._linear_package(names.down_proj, down_proj),
        )

    def quantize_layers(self, layer_indices=None) -> list[QuantizedLlamaLayerPackage]:
        if layer_indices is None:
            layer_indices = range(self.checkpoint_adapter.config.num_hidden_layers)
        return [self.quantize_layer(layer_idx) for layer_idx in layer_indices]

    def report(self) -> QuantizationReport:
        return QuantizationReport(
            ok=not self._errors,
            quantized_tensors=list(dict.fromkeys(self._quantized_tensors)),
            skipped_tensors=list(dict.fromkeys(self._skipped_tensors)),
            errors=list(self._errors),
            metrics=dict(self._metrics),
        )
