from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoint_manifest import CheckpointManifest
from .checkpoint_mapping import LayerTensorNames, infer_model_family, llama_layer_tensor_names, validate_llama_checkpoint_shapes
from .llama_config import LlamaLikeConfig
from .qkv_fusion import fuse_qkv_weights, fused_qkv_shape
from .quant_packaging import llama_quantized_layer_specs
from .tensor_store import InMemoryTensorStore, ManifestTensorStore, TensorStore


@dataclass
class CheckpointAdapterConfig:
    model_family: str = "llama"
    fuse_qkv: bool = True
    require_all_layers: bool = True
    allow_missing_lm_head: bool = True
    allow_missing_embeddings: bool = True
    quantized: bool = False
    bits: int = 4
    group_size: int = 32


@dataclass
class AdapterIssue:
    severity: str
    tensor: str | None
    message: str


@dataclass
class AdapterReport:
    ok: bool
    issues: list[AdapterIssue] = field(default_factory=list)
    derived_tensors: list[str] = field(default_factory=list)
    layer_count: int = 0

    def errors(self) -> list[AdapterIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[AdapterIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def raise_for_errors(self) -> None:
        errors = self.errors()
        if errors:
            joined = "; ".join(f"{issue.tensor or '<global>'}: {issue.message}" for issue in errors)
            raise ValueError(joined)


class CheckpointAdapter:
    def __init__(self, config: LlamaLikeConfig, tensor_store: TensorStore, adapter_config: CheckpointAdapterConfig | None = None):
        self.config = config.validate()
        self.tensor_store = tensor_store
        self.adapter_config = adapter_config or CheckpointAdapterConfig()

    def tensor_names(self) -> list[str]:
        return self.tensor_store.keys()

    def layer_names(self, layer_idx: int) -> LayerTensorNames:
        return llama_layer_tensor_names(layer_idx)

    def expected_layer_shapes(self, layer_idx: int) -> dict[str, tuple[int, ...]]:
        _ = self.layer_names(layer_idx)
        return {
            "q_proj": (self.config.q_output_dim(), self.config.hidden_size),
            "k_proj": (self.config.kv_output_dim(), self.config.hidden_size),
            "v_proj": (self.config.kv_output_dim(), self.config.hidden_size),
            "qkv_fused": fused_qkv_shape(self.config),
            "o_proj": (self.config.hidden_size, self.config.q_output_dim()),
            "gate_proj": (self.config.intermediate_size, self.config.hidden_size),
            "up_proj": (self.config.intermediate_size, self.config.hidden_size),
            "down_proj": (self.config.hidden_size, self.config.intermediate_size),
            "input_layernorm": (self.config.hidden_size,),
            "post_attention_layernorm": (self.config.hidden_size,),
        }

    def layer_shapes(self, layer_idx: int) -> dict[str, tuple[int, ...] | None]:
        names = self.layer_names(layer_idx)
        out: dict[str, tuple[int, ...] | None] = {}
        for logical_name, tensor_name in names.as_dict().items():
            out[logical_name] = self.tensor_store.get_shape(tensor_name) if self.tensor_store.has(tensor_name) else None
        out["qkv_fused"] = self.get_fused_qkv_shape(layer_idx)
        return out

    def can_load_tensor(self, name: str) -> bool:
        return self.tensor_store.has(name)

    def load_tensor(self, name: str):
        return self.tensor_store.load(name)

    def _manifest_from_store(self) -> CheckpointManifest:
        tensors = {
            name: {"name": name, "shape": list(self.tensor_store.get_shape(name)), "dtype": self.tensor_store.get_dtype(name)}
            for name in self.tensor_names()
        }
        model_type = self.adapter_config.model_family or self.config.model_type
        return CheckpointManifest.from_dict({"model_type": model_type, "tensors": tensors})

    def validate(self) -> AdapterReport:
        manifest = self._manifest_from_store()
        report = validate_llama_checkpoint_shapes(manifest, self.config, require_all_layers=self.adapter_config.require_all_layers)
        issues = [AdapterIssue(issue.severity, issue.tensor, issue.message) for issue in report.issues]
        if self.adapter_config.allow_missing_embeddings:
            for name in ("model.embed_tokens.weight", "lm_head.weight"):
                if not self.tensor_store.has(name):
                    issues.append(AdapterIssue("warning", name, "optional tensor is missing"))
        ok = not any(issue.severity == "error" for issue in issues)
        derived = []
        if self.adapter_config.fuse_qkv:
            derived = [f"model.layers.{layer_idx}.self_attn.qkv_proj.fused_weight" for layer_idx in range(self.config.num_hidden_layers)]
        return AdapterReport(ok=ok, issues=issues, derived_tensors=derived, layer_count=self.config.num_hidden_layers)

    def fuse_qkv_for_layer(self, layer_idx: int):
        if isinstance(self.tensor_store, ManifestTensorStore):
            raise NotImplementedError("ManifestTensorStore is shape-only; fused QKV requires loadable tensor data")
        names = self.layer_names(layer_idx)
        return fuse_qkv_weights(
            self.tensor_store.load(names.q_proj),
            self.tensor_store.load(names.k_proj),
            self.tensor_store.load(names.v_proj),
        )

    def get_fused_qkv_shape(self, layer_idx: int) -> tuple[int, int]:
        _ = layer_idx
        return fused_qkv_shape(self.config)

    def quantized_specs_for_layer(self, layer_idx: int):
        return llama_quantized_layer_specs(
            self.config,
            layer_idx=layer_idx,
            bits=self.adapter_config.bits,
            group_size=self.adapter_config.group_size,
        )

    def describe(self) -> dict[str, Any]:
        family = self.adapter_config.model_family
        if family == "llama" and isinstance(self.tensor_store, ManifestTensorStore):
            family = infer_model_family(self.tensor_store.manifest)
        desc = {
            "model_family": family,
            "hidden_size": self.config.hidden_size,
            "layers": self.config.num_hidden_layers,
            "heads": self.config.num_attention_heads,
            "kv_heads": self.config.num_key_value_heads,
            "gqa": self.config.is_gqa(),
            "tensor_count": len(self.tensor_names()),
            "fuse_qkv": self.adapter_config.fuse_qkv,
            "quantized": self.adapter_config.quantized,
        }
        if self.adapter_config.quantized:
            desc["bits"] = self.adapter_config.bits
            desc["group_size"] = self.adapter_config.group_size
        return desc


def adapter_from_manifest_path(config: LlamaLikeConfig, manifest_path, adapter_config: CheckpointAdapterConfig | None = None) -> CheckpointAdapter:
    manifest = CheckpointManifest.load_json(Path(manifest_path))
    return CheckpointAdapter(config, ManifestTensorStore(manifest), adapter_config=adapter_config)


def adapter_from_in_memory_tensors(config: LlamaLikeConfig, tensors: dict[str, Any], adapter_config: CheckpointAdapterConfig | None = None) -> CheckpointAdapter:
    return CheckpointAdapter(config, InMemoryTensorStore(tensors), adapter_config=adapter_config)
