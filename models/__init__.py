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
from .checkpoint_converter import CheckpointConverter, CheckpointConverterConfig, CheckpointConverterReport
from .checkpoint_quantizer import CheckpointQuantizer, QuantizationReport
from .generation import (
    GenerationConfig,
    ToyGenerationState,
    ToyLlamaGenerationModel,
    ToyLlamaStackGenerationModel,
    create_synthetic_generation_model,
    create_synthetic_stack_generation_model,
)
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
from .quantized_package_io import (
    QuantizedCheckpointPackage,
    QuantizedLayerMetadata,
    QuantizedTensorMetadata,
    package_from_quantized_layers,
)
from .sampling import apply_repetition_penalty, greedy_sample, sample_logits, softmax, top_k_filter, top_p_filter
from .tensor_store import InMemoryTensorStore, ManifestTensorStore, SafeTensorsTensorStore, TensorStore
from .tiny_generation_pipeline import (
    GenerationResult,
    PrefillResult,
    TinyGenerationPipeline,
    TinyGenerationPipelineConfig,
    create_pipeline_from_quantized_package,
)
from .tokenization import CharTokenizer, TokenizerProtocol, WhitespaceTokenizer
from .tokenizer_adapters import (
    HFTokenizerAdapter,
    OptionalDependencyError,
    SentencePieceTokenizerAdapter,
    TokenizerAdapterFactory,
    TokenizerInfo,
    describe_tokenizer,
    load_tokenizer_for_generation,
)
from .weight_layouts import LayerWeightSpec, LinearWeightSpec, fused_qkv_spec, llama_layer_weight_specs, validate_weight_shapes

try:
    from .model_adapter import KernelBackendConfig, LlamaLayerState, LlamaLikeKernelAdapter
except ImportError:  # pragma: no cover - allows shape-only helpers without MLX installed
    KernelBackendConfig = None
    LlamaLayerState = None
    LlamaLikeKernelAdapter = None

__all__ = [
    "describe_tokenizer",
    "HFTokenizerAdapter",
    "CheckpointConverter",
    "CheckpointConverterConfig",
    "CheckpointConverterReport",
    "AdapterIssue",
    "AdapterReport",
    "apply_repetition_penalty",
    "CharTokenizer",
    "CheckpointAdapter",
    "CheckpointAdapterConfig",
    "CheckpointManifest",
    "CheckpointQuantizer",
    "create_fused_qkv_manifest",
    "create_synthetic_generation_model",
    "create_synthetic_stack_generation_model",
    "dequantize_quantized_weight",
    "extra_tensors",
    "fuse_qkv_shapes",
    "fuse_qkv_weights",
    "fused_qkv_shape",
    "fused_qkv_spec",
    "GenerationConfig",
    "GenerationResult",
    "greedy_sample",
    "InMemoryTensorStore",
    "infer_model_family",
    "LayerTensorNames",
    "LayerWeightAdapter",
    "LayerWeightSpec",
    "LayerWeights",
    "LinearWeightSpec",
    "llama_7b_like",
    "llama_8b_like",
    "llama_layer_tensor_names",
    "llama_layer_weight_specs",
    "llama_quantized_layer_specs",
    "LlamaLikeConfig",
    "load_tokenizer_for_generation",
    "ManifestTensorStore",
    "missing_required_tensors",
    "OptionalDependencyError",
    "mistral_layer_tensor_names",
    "q4_packed_shape",
    "q8_packed_shape",
    "quantization_error",
    "quantized_linear_spec",
    "QuantizationConfig",
    "QuantizationReport",
    "SentencePieceTokenizerAdapter",
    "QuantizedCheckpointPackage",
    "QuantizedLayerMetadata",
    "QuantizedLinearPackage",
    "QuantizedLlamaLayerPackage",
    "QuantizedTensorMetadata",
    "QuantizedTensorSpec",
    "QuantizedWeight",
    "quantize_weight_groupwise",
    "resolve_required_tensors",
    "SafeTensorsTensorStore",
    "sample_logits",
    "softmax",
    "split_fused_qkv_shape",
    "split_fused_qkv_weight",
    "TensorInfo",
    "TensorNamePattern",
    "TensorStore",
    "TinyGenerationPipeline",
    "TinyGenerationPipelineConfig",
    "tiny_debug_config",
    "tiny_gqa_debug_config",
    "TokenizerAdapterFactory",
    "TokenizerInfo",
    "TokenizerProtocol",
    "top_k_filter",
    "top_p_filter",
    "ToyGenerationState",
    "ToyLlamaGenerationModel",
    "ToyLlamaStackGenerationModel",
    "ValidationIssue",
    "ValidationReport",
    "validate_llama_checkpoint_shapes",
    "validate_llama_layer_shapes",
    "validate_weight_shapes",
    "WhitespaceTokenizer",
    "adapter_from_in_memory_tensors",
    "adapter_from_manifest_path",
    "build_fused_qkv_manifest_entries",
    "build_llama_name_map",
    "build_rope_tables",
    "create_pipeline_from_quantized_package",
    "package_from_quantized_layers",
]

if KernelBackendConfig is not None:
    __all__.extend(["KernelBackendConfig", "LlamaLayerState", "LlamaLikeKernelAdapter"])
    "PrefillResult",
