from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (
    CharTokenizer,
    TinyGenerationPipelineConfig,
    package_from_quantized_layers,
    quantize_weight_groupwise,
    validate_config_against_package,
    validate_generation_alignment,
    validate_tokenizer_against_config,
)
from models.quantize_weights import QuantizationConfig
from models.quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage
from models.quantized_package_io import QuantizedTensorMetadata

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for alignment_demo.py") from exc


def _quantized_linear(name: str, shape: tuple[int, int], *, bits: int, group_size: int, seed: int) -> QuantizedLinearPackage:
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=shape).astype(np.float32)
    quantized = quantize_weight_groupwise(weight, QuantizationConfig(bits=bits, group_size=group_size))
    return QuantizedLinearPackage(
        name=name,
        weight=quantized.packed_weight,
        scales=quantized.scales,
        zeros=quantized.zeros,
        bits=bits,
        group_size=group_size,
        original_shape=shape,
    )


def _tensor_metadata(name: str, role: str, shape: tuple[int, int], *, bits: int, group_size: int, seed: int) -> QuantizedTensorMetadata:
    linear = _quantized_linear(name, shape, bits=bits, group_size=group_size, seed=seed)
    return QuantizedTensorMetadata(
        name=name,
        role=role,
        bits=linear.bits,
        group_size=linear.group_size,
        original_shape=shape,
        packed_shape=tuple(int(dim) for dim in linear.weight.shape),
        scales_shape=tuple(int(dim) for dim in linear.scales.shape),
        zeros_shape=tuple(int(dim) for dim in linear.zeros.shape) if linear.zeros is not None else None,
    )


def build_synthetic_package():
    tokenizer = CharTokenizer()
    pipeline_config = TinyGenerationPipelineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=32,
        vocab_size=tokenizer.vocab_size,
        bits=4,
        group_size=32,
        backend_preset="reference",
    ).validate()
    config = pipeline_config.to_llama_config()
    layers = []
    for layer_idx in range(config.num_hidden_layers):
        layers.append(
            QuantizedLlamaLayerPackage(
                layer_idx=layer_idx,
                input_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
                post_attention_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
                qkv=_quantized_linear("qkv", (config.q_output_dim() + 2 * config.kv_output_dim(), config.hidden_size), bits=pipeline_config.bits, group_size=pipeline_config.group_size, seed=layer_idx + 1),
                o_proj=_quantized_linear("o_proj", (config.hidden_size, config.hidden_size), bits=pipeline_config.bits, group_size=pipeline_config.group_size, seed=layer_idx + 10),
                gate_proj=_quantized_linear("gate_proj", (config.intermediate_size, config.hidden_size), bits=pipeline_config.bits, group_size=pipeline_config.group_size, seed=layer_idx + 20),
                up_proj=_quantized_linear("up_proj", (config.intermediate_size, config.hidden_size), bits=pipeline_config.bits, group_size=pipeline_config.group_size, seed=layer_idx + 30),
                down_proj=_quantized_linear("down_proj", (config.hidden_size, config.intermediate_size), bits=pipeline_config.bits, group_size=pipeline_config.group_size, seed=layer_idx + 40),
            )
        )
    package = package_from_quantized_layers(
        config,
        layers,
        bits=pipeline_config.bits,
        group_size=pipeline_config.group_size,
        model_type=config.model_type,
    )
    package.global_tensors["embedding"] = _tensor_metadata(
        "embedding",
        "embedding",
        (pipeline_config.vocab_size, config.hidden_size),
        bits=pipeline_config.bits,
        group_size=pipeline_config.group_size,
        seed=100,
    )
    package.global_tensors["lm_head"] = _tensor_metadata(
        "lm_head",
        "lm_head",
        (pipeline_config.vocab_size, config.hidden_size),
        bits=pipeline_config.bits,
        group_size=pipeline_config.group_size,
        seed=101,
    )
    return tokenizer, pipeline_config, config, package


def main() -> None:
    tokenizer, pipeline_config, config, package = build_synthetic_package()
    matching_vocab = tokenizer.vocab_size
    embedding = np.zeros((matching_vocab, config.hidden_size), dtype=np.float32)
    lm_head = np.zeros((matching_vocab, config.hidden_size), dtype=np.float32)

    print("Tokenizer vs config")
    report = validate_tokenizer_against_config(tokenizer, config, embedding=embedding, lm_head=lm_head)
    print(report.pretty_print())
    print()

    print("Config vs package")
    package.config["vocab_size"] = matching_vocab
    report = validate_config_against_package(config, package)
    print(report.pretty_print())
    print()

    print("Full generation alignment")
    report = validate_generation_alignment(
        tokenizer=tokenizer,
        config=config,
        package=package,
        embedding=embedding,
        lm_head=lm_head,
        bits=pipeline_config.bits,
        group_size=pipeline_config.group_size,
    )
    print(report.pretty_print())
    print()

    print("Intentional mismatch")
    bad_lm_head = np.zeros((matching_vocab + 3, config.hidden_size), dtype=np.float32)
    mismatch_report = validate_generation_alignment(
        tokenizer=tokenizer,
        config=config,
        package=package,
        embedding=embedding,
        lm_head=bad_lm_head,
        bits=pipeline_config.bits,
        group_size=pipeline_config.group_size,
    )
    print(mismatch_report.pretty_print())


if __name__ == "__main__":
    main()
