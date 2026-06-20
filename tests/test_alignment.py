from __future__ import annotations

import math

import pytest

from models import (
    AlignmentIssue,
    AlignmentReport,
    CharTokenizer,
    LlamaLikeConfig,
    QuantizedCheckpointPackage,
    QuantizedLayerMetadata,
    QuantizedTensorMetadata,
    package_from_quantized_layers,
    q4_packed_shape,
    quantize_weight_groupwise,
    tokenizer_alignment_info,
    validate_config_against_package,
    validate_generation_alignment,
    validate_quantization_alignment,
    validate_tokenizer_against_config,
)
from models.quantize_weights import QuantizationConfig
from models.quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage

np = pytest.importorskip("numpy")


class FakeTokenizer:
    vocab_size = 8
    bos_token_id = 1
    eos_token_id = 99
    pad_token_id = None
    unk_token_id = 3

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [1, 2]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True, stop_at_eos: bool | None = None) -> str:
        return "x"


def _tiny_config(**overrides) -> LlamaLikeConfig:
    kwargs = dict(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=32,
        vocab_size=96,
        model_type="alignment_test_model",
    )
    kwargs.update(overrides)
    return LlamaLikeConfig(**kwargs).validate()


def _quantized_linear(name: str, shape: tuple[int, int], *, bits: int = 4, group_size: int = 32, seed: int = 0) -> QuantizedLinearPackage:
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


def _global_tensor_meta(name: str, role: str, shape: tuple[int, int], *, bits: int = 4, group_size: int = 32, seed: int = 0) -> QuantizedTensorMetadata:
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


def _synthetic_package(config: LlamaLikeConfig | None = None, *, bits: int = 4, group_size: int = 32) -> QuantizedCheckpointPackage:
    config = config or _tiny_config()
    layers = []
    for layer_idx in range(config.num_hidden_layers):
        layer = QuantizedLlamaLayerPackage(
            layer_idx=layer_idx,
            input_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
            post_attention_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
            qkv=_quantized_linear("qkv", (config.q_output_dim() + 2 * config.kv_output_dim(), config.hidden_size), bits=bits, group_size=group_size, seed=layer_idx + 1),
            o_proj=_quantized_linear("o_proj", (config.hidden_size, config.hidden_size), bits=bits, group_size=group_size, seed=layer_idx + 10),
            gate_proj=_quantized_linear("gate_proj", (config.intermediate_size, config.hidden_size), bits=bits, group_size=group_size, seed=layer_idx + 20),
            up_proj=_quantized_linear("up_proj", (config.intermediate_size, config.hidden_size), bits=bits, group_size=group_size, seed=layer_idx + 30),
            down_proj=_quantized_linear("down_proj", (config.hidden_size, config.intermediate_size), bits=bits, group_size=group_size, seed=layer_idx + 40),
        )
        layers.append(layer)
    package = package_from_quantized_layers(config, layers, bits=bits, group_size=group_size, model_type=config.model_type)
    package.global_tensors["embedding"] = _global_tensor_meta("embedding", "embedding", (config.vocab_size, config.hidden_size), bits=bits, group_size=group_size, seed=100)
    package.global_tensors["lm_head"] = _global_tensor_meta("lm_head", "lm_head", (config.vocab_size, config.hidden_size), bits=bits, group_size=group_size, seed=101)
    return package


def test_alignment_report_ok_with_no_errors():
    report = AlignmentReport(ok=False, issues=[])
    assert report.ok is True
    assert report.errors() == []


def test_alignment_report_filters_by_severity():
    report = AlignmentReport(
        ok=True,
        issues=[
            AlignmentIssue("error", "A", "bad"),
            AlignmentIssue("warning", "B", "warn"),
            AlignmentIssue("info", "C", "info"),
        ],
    )
    assert len(report.errors()) == 1
    assert len(report.warnings()) == 1
    assert len(report.infos()) == 1


def test_alignment_report_raise_for_errors_only_raises_on_errors():
    ok_report = AlignmentReport(ok=True, issues=[AlignmentIssue("warning", "W", "warn")])
    ok_report.raise_for_errors()
    bad_report = AlignmentReport(ok=True, issues=[AlignmentIssue("error", "E", "bad")])
    with pytest.raises(ValueError, match="Alignment status: error"):
        bad_report.raise_for_errors()


def test_tokenizer_alignment_info_works_on_char_tokenizer():
    info = tokenizer_alignment_info(CharTokenizer())
    assert info["vocab_size"] is not None
    assert info["bos_token_id"] is not None
    assert info["tokenizer_type"] == "CharTokenizer"


def test_validate_tokenizer_against_config_passes_for_matching_vocab_embedding_and_lm_head():
    tokenizer = CharTokenizer()
    config = _tiny_config(vocab_size=tokenizer.vocab_size)
    embedding = np.zeros((tokenizer.vocab_size, config.hidden_size), dtype=np.float32)
    lm_head = np.zeros((tokenizer.vocab_size, config.hidden_size), dtype=np.float32)
    report = validate_tokenizer_against_config(tokenizer, config, embedding=embedding, lm_head=lm_head)
    assert report.ok
    assert report.errors() == []


def test_validate_tokenizer_against_config_catches_vocab_mismatch():
    tokenizer = CharTokenizer()
    config = _tiny_config(vocab_size=tokenizer.vocab_size + 7)
    report = validate_tokenizer_against_config(tokenizer, config)
    assert not report.ok
    assert any(issue.code == "VOCAB_SIZE_MISMATCH" for issue in report.errors())


def test_validate_tokenizer_against_config_catches_token_id_out_of_range():
    config = _tiny_config(vocab_size=FakeTokenizer.vocab_size)
    report = validate_tokenizer_against_config(FakeTokenizer(), config)
    assert not report.ok
    assert any(issue.code == "TOKEN_ID_OUT_OF_RANGE" for issue in report.errors())


def test_validate_config_against_package_passes_for_matching_synthetic_package():
    config = _tiny_config()
    report = validate_config_against_package(config, _synthetic_package(config))
    assert report.ok


def test_validate_config_against_package_catches_hidden_size_mismatch():
    config = _tiny_config()
    package = _synthetic_package(config)
    package.config["hidden_size"] = config.hidden_size + 8
    report = validate_config_against_package(config, package)
    assert not report.ok
    assert any(issue.code == "HIDDEN_SIZE_MISMATCH" for issue in report.errors())


def test_validate_config_against_package_catches_qkv_shape_mismatch():
    config = _tiny_config()
    package = _synthetic_package(config)
    qkv = package.layers[0].tensors["qkv"]
    package.layers[0].tensors["qkv"] = QuantizedTensorMetadata(
        name=qkv.name,
        role=qkv.role,
        bits=qkv.bits,
        group_size=qkv.group_size,
        original_shape=(qkv.original_shape[0] + 1, qkv.original_shape[1]),
        packed_shape=qkv.packed_shape,
        scales_shape=qkv.scales_shape,
        zeros_shape=qkv.zeros_shape,
    )
    report = validate_config_against_package(config, package)
    assert not report.ok
    assert any(issue.field == "layers[0].qkv.original_shape" for issue in report.errors())


def test_validate_quantization_alignment_passes_for_q4_package():
    report = validate_quantization_alignment(_synthetic_package(bits=4))
    assert report.ok


def test_validate_quantization_alignment_catches_q4_packed_shape_mismatch():
    config = _tiny_config()
    package = _synthetic_package(config, bits=4)
    tensor = package.layers[0].tensors["o_proj"]
    package.layers[0].tensors["o_proj"] = QuantizedTensorMetadata(
        name=tensor.name,
        role=tensor.role,
        bits=tensor.bits,
        group_size=tensor.group_size,
        original_shape=tensor.original_shape,
        packed_shape=(tensor.packed_shape[0], tensor.packed_shape[1] + 1),
        scales_shape=tensor.scales_shape,
        zeros_shape=tensor.zeros_shape,
    )
    report = validate_quantization_alignment(package)
    assert not report.ok
    assert any(issue.code == "PACKED_SHAPE_MISMATCH" for issue in report.errors())


def test_validate_generation_alignment_combines_reports_and_summary_fields():
    tokenizer = CharTokenizer()
    config = _tiny_config(vocab_size=tokenizer.vocab_size)
    package = _synthetic_package(config)
    report = validate_generation_alignment(
        tokenizer=tokenizer,
        config=config,
        package=package,
        embedding=np.zeros((tokenizer.vocab_size, config.hidden_size), dtype=np.float32),
        lm_head=np.zeros((tokenizer.vocab_size, config.hidden_size), dtype=np.float32),
        bits=4,
        group_size=32,
    )
    assert report.summary["has_tokenizer"] is True
    assert report.summary["has_package"] is True
    assert report.summary["bits"] == 4
    assert report.ok
