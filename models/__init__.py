from .checkpoint_adapter import (
    AdapterIssue,
    AdapterReport,
    CheckpointAdapter,
    CheckpointAdapterConfig,
    adapter_from_in_memory_tensors,
    adapter_from_manifest_path,
)
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
from .checkpoint_quantizer import CheckpointQuantizer, QuantizationReport
from .layer_weight_adapter import LayerWeightAdapter, LayerWeights
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
from .quantize_weights import (
    QuantizationConfig,
    QuantizedWeight,
    dequantize_quantized_weight,
    quantization_error,
    quantize_weight_groupwise,
)
from .quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage
from .tensor_store import InMemoryTensorStore, ManifestTensorStore, SafeTensorsTensorStore, TensorStore
from .weight_layouts import LayerWeightSpec, LinearWeightSpec, fused_qkv_spec, llama_layer_weight_specs, validate_weight_shapes

try:
    from .model_adapter import KernelBackendConfig, LlamaLayerState, LlamaLikeKernelAdapter
except ImportError:  # pragma: no cover - allows shape-only helpers without MLX installed
    KernelBackendConfig = None
    LlamaLayerState = None
    LlamaLikeKernelAdapter = None

__all__ = [
    "AdapterIssue",
    "AdapterReport",
    "CheckpointAdapter",
    "CheckpointAdapterConfig",
    "CheckpointManifest",
    "CheckpointQuantizer",
    "InMemoryTensorStore",
    "LayerTensorNames",
    "LayerWeightAdapter",
    "LayerWeightSpec",
    "LayerWeights",
    "LinearWeightSpec",
    "LlamaLikeConfig",
    "ManifestTensorStore",
    "QuantizationConfig",
    "QuantizationReport",
    "QuantizedLinearPackage",
    "QuantizedLlamaLayerPackage",
    "QuantizedTensorSpec",
    "QuantizedWeight",
    "SafeTensorsTensorStore",
    "TensorInfo",
    "TensorNamePattern",
    "TensorStore",
    "ValidationIssue",
    "ValidationReport",
    "adapter_from_in_memory_tensors",
    "adapter_from_manifest_path",
    "build_fused_qkv_manifest_entries",
    "build_llama_name_map",
    "build_rope_tables",
    "create_fused_qkv_manifest",
    "dequantize_quantized_weight",
    "extra_tensors",
    "fuse_qkv_shapes",
    "fuse_qkv_weights",
    "fused_qkv_shape",
    "fused_qkv_spec",
    "infer_model_family",
    "llama_7b_like",
    "llama_8b_like",
    "llama_layer_tensor_names",
    "llama_layer_weight_specs",
    "llama_quantized_layer_specs",
    "missing_required_tensors",
    "mistral_layer_tensor_names",
    "q4_packed_shape",
    "q8_packed_shape",
    "quantization_error",
    "quantize_weight_groupwise",
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
