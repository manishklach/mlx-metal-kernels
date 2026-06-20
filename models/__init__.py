from .checkpoint_manifest import CheckpointManifest, TensorInfo
from .checkpoint_mapping import (
    LayerTensorNames,
    TensorNamePattern,
    ValidationIssue,
    ValidationReport,
    build_llama_name_map,
    extra_tensors,
    infer_model_family,
    llama_layer_tensor_names,
    missing_required_tensors,
    mistral_layer_tensor_names,
    resolve_required_tensors,
    validate_llama_checkpoint_shapes,
    validate_llama_layer_shapes,
)
from .llama_config import (
    LlamaLikeConfig,
    build_rope_tables,
    llama_7b_like,
    llama_8b_like,
    tiny_debug_config,
    tiny_gqa_debug_config,
)
from .qkv_fusion import (
    build_fused_qkv_manifest_entries,
    create_fused_qkv_manifest,
    fuse_qkv_shapes,
    fuse_qkv_weights,
    fused_qkv_shape,
    split_fused_qkv_shape,
    split_fused_qkv_weight,
)
from .quant_packaging import (
    QuantizedTensorSpec,
    llama_quantized_layer_specs,
    q4_packed_shape,
    q8_packed_shape,
    quantized_linear_spec,
)
from .weight_layouts import LayerWeightSpec, LinearWeightSpec, fused_qkv_spec, llama_layer_weight_specs, validate_weight_shapes

try:
    from .model_adapter import KernelBackendConfig, LlamaLayerState, LlamaLikeKernelAdapter
except ImportError:  # pragma: no cover - allows shape-only helpers without MLX installed
    KernelBackendConfig = None
    LlamaLayerState = None
    LlamaLikeKernelAdapter = None

__all__ = [
    "CheckpointManifest",
    "LayerTensorNames",
    "LayerWeightSpec",
    "LinearWeightSpec",
    "LlamaLikeConfig",
    "QuantizedTensorSpec",
    "TensorInfo",
    "TensorNamePattern",
    "ValidationIssue",
    "ValidationReport",
    "build_fused_qkv_manifest_entries",
    "build_llama_name_map",
    "build_rope_tables",
    "create_fused_qkv_manifest",
    "extra_tensors",
    "fuse_qkv_shapes",
    "fuse_qkv_weights",
    "fused_qkv_spec",
    "fused_qkv_shape",
    "infer_model_family",
    "llama_7b_like",
    "llama_8b_like",
    "llama_layer_weight_specs",
    "llama_layer_tensor_names",
    "llama_quantized_layer_specs",
    "missing_required_tensors",
    "mistral_layer_tensor_names",
    "q4_packed_shape",
    "q8_packed_shape",
    "quantized_linear_spec",
    "resolve_required_tensors",
    "split_fused_qkv_shape",
    "split_fused_qkv_weight",
    "tiny_debug_config",
    "tiny_gqa_debug_config",
    "validate_llama_checkpoint_shapes",
    "validate_llama_layer_shapes",
    "validate_weight_shapes",
]

if KernelBackendConfig is not None:
    __all__.extend(["KernelBackendConfig", "LlamaLayerState", "LlamaLikeKernelAdapter"])
