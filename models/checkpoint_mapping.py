from __future__ import annotations

from dataclasses import dataclass

from .checkpoint_manifest import CheckpointManifest
from .llama_config import LlamaLikeConfig


@dataclass
class TensorNamePattern:
    logical_name: str
    checkpoint_name: str
    required: bool = True


@dataclass
class LayerTensorNames:
    layer_idx: int
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str
    gate_proj: str
    up_proj: str
    down_proj: str
    input_layernorm: str
    post_attention_layernorm: str

    def as_dict(self) -> dict[str, str]:
        return {
            "q_proj": self.q_proj,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
            "o_proj": self.o_proj,
            "gate_proj": self.gate_proj,
            "up_proj": self.up_proj,
            "down_proj": self.down_proj,
            "input_layernorm": self.input_layernorm,
            "post_attention_layernorm": self.post_attention_layernorm,
        }


@dataclass
class ValidationIssue:
    severity: str
    tensor: str
    message: str


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue]

    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def raise_for_errors(self) -> None:
        if self.errors():
            joined = "; ".join(f"{issue.tensor}: {issue.message}" for issue in self.errors())
            raise ValueError(joined)


def llama_layer_tensor_names(layer_idx, prefix="model.layers") -> LayerTensorNames:
    stem = f"{prefix}.{layer_idx}"
    return LayerTensorNames(
        layer_idx=layer_idx,
        q_proj=f"{stem}.self_attn.q_proj.weight",
        k_proj=f"{stem}.self_attn.k_proj.weight",
        v_proj=f"{stem}.self_attn.v_proj.weight",
        o_proj=f"{stem}.self_attn.o_proj.weight",
        gate_proj=f"{stem}.mlp.gate_proj.weight",
        up_proj=f"{stem}.mlp.up_proj.weight",
        down_proj=f"{stem}.mlp.down_proj.weight",
        input_layernorm=f"{stem}.input_layernorm.weight",
        post_attention_layernorm=f"{stem}.post_attention_layernorm.weight",
    )


def mistral_layer_tensor_names(layer_idx, prefix="model.layers") -> LayerTensorNames:
    return llama_layer_tensor_names(layer_idx, prefix=prefix)


def build_llama_name_map(num_layers, prefix="model.layers") -> dict[int, LayerTensorNames]:
    return {layer_idx: llama_layer_tensor_names(layer_idx, prefix=prefix) for layer_idx in range(num_layers)}


def infer_model_family(manifest: CheckpointManifest) -> str:
    model_type = manifest.model_type.lower()
    if "mistral" in model_type:
        return "mistral"
    if "llama" in model_type:
        return "llama"
    if any(name.startswith("model.layers.") for name in manifest.tensor_names()):
        return "llama"
    return "unknown"


def resolve_required_tensors(manifest: CheckpointManifest, config: LlamaLikeConfig, *, include_embeddings: bool = False) -> dict[str, str]:
    required = {}
    for layer_idx, names in build_llama_name_map(config.num_hidden_layers).items():
        for logical_name, checkpoint_name in names.as_dict().items():
            if manifest.has(checkpoint_name):
                required[f"layers.{layer_idx}.{logical_name}"] = checkpoint_name
    if include_embeddings:
        for name in ("model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"):
            if manifest.has(name):
                required[name] = name
    return required


def missing_required_tensors(manifest: CheckpointManifest, config: LlamaLikeConfig, *, include_embeddings: bool = False) -> list[str]:
    missing = []
    for layer_idx, names in build_llama_name_map(config.num_hidden_layers).items():
        for checkpoint_name in names.as_dict().values():
            if not manifest.has(checkpoint_name):
                missing.append(checkpoint_name)
    if include_embeddings:
        for name in ("model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"):
            if not manifest.has(name):
                missing.append(name)
    return missing


def extra_tensors(manifest: CheckpointManifest, expected_names) -> list[str]:
    expected = set(expected_names)
    return sorted(name for name in manifest.tensor_names() if name not in expected)


def _expected_layer_shapes(config: LlamaLikeConfig) -> dict[str, tuple[int, ...]]:
    return {
        "q_proj": (config.q_output_dim(), config.hidden_size),
        "k_proj": (config.kv_output_dim(), config.hidden_size),
        "v_proj": (config.kv_output_dim(), config.hidden_size),
        "o_proj": (config.hidden_size, config.q_output_dim()),
        "gate_proj": (config.intermediate_size, config.hidden_size),
        "up_proj": (config.intermediate_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.intermediate_size),
        "input_layernorm": (config.hidden_size,),
        "post_attention_layernorm": (config.hidden_size,),
    }


def validate_llama_layer_shapes(manifest: CheckpointManifest, config: LlamaLikeConfig, *, layer_idx) -> ValidationReport:
    names = llama_layer_tensor_names(layer_idx)
    expected = _expected_layer_shapes(config)
    issues: list[ValidationIssue] = []
    for logical_name, checkpoint_name in names.as_dict().items():
        tensor = manifest.get(checkpoint_name)
        if tensor is None:
            issues.append(ValidationIssue("error", checkpoint_name, "missing required tensor"))
            continue
        if tuple(tensor.shape) != expected[logical_name]:
            issues.append(
                ValidationIssue(
                    "error",
                    checkpoint_name,
                    f"expected shape {expected[logical_name]}, got {tensor.shape}",
                )
            )
    return ValidationReport(ok=not issues, issues=issues)


def validate_llama_checkpoint_shapes(
    manifest: CheckpointManifest,
    config: LlamaLikeConfig,
    *,
    require_all_layers=True,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for layer_idx in range(config.num_hidden_layers):
        report = validate_llama_layer_shapes(manifest, config, layer_idx=layer_idx)
        issues.extend(report.issues)
    if not require_all_layers:
        issues = [issue for issue in issues if "missing required tensor" not in issue.message]
    return ValidationReport(ok=not any(issue.severity == "error" for issue in issues), issues=issues)
