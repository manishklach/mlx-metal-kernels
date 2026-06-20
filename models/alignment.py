from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .llama_config import LlamaLikeConfig
from .quantized_package_io import QuantizedCheckpointPackage, QuantizedTensorMetadata


@dataclass
class AlignmentIssue:
    severity: str
    code: str
    message: str
    field: str | None = None
    expected: Any | None = None
    actual: Any | None = None


@dataclass
class AlignmentReport:
    ok: bool
    issues: list[AlignmentIssue]
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ok = not any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> list[AlignmentIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[AlignmentIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def infos(self) -> list[AlignmentIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError(self.pretty_print())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [asdict(issue) for issue in self.issues],
            "summary": dict(self.summary),
        }

    def pretty_print(self) -> str:
        lines = [f"Alignment status: {'ok' if self.ok else 'error'}"]
        if self.summary:
            summary_items = ", ".join(f"{key}={value!r}" for key, value in sorted(self.summary.items()))
            lines.append(f"Summary: {summary_items}")
        if not self.issues:
            lines.append("No alignment issues.")
            return "\n".join(lines)
        for issue in self.issues:
            detail = f"[{issue.severity}] {issue.code}: {issue.message}"
            if issue.field is not None:
                detail += f" (field={issue.field})"
            if issue.expected is not None or issue.actual is not None:
                detail += f" expected={issue.expected!r} actual={issue.actual!r}"
            lines.append(detail)
        return "\n".join(lines)


def tokenizer_alignment_info(tokenizer) -> dict[str, Any]:
    if tokenizer is None:
        return {
            "vocab_size": None,
            "bos_token_id": None,
            "eos_token_id": None,
            "pad_token_id": None,
            "unk_token_id": None,
            "tokenizer_type": None,
        }
    return {
        "vocab_size": _safe_int(getattr(tokenizer, "vocab_size", None)),
        "bos_token_id": _safe_int(getattr(tokenizer, "bos_token_id", None)),
        "eos_token_id": _safe_int(getattr(tokenizer, "eos_token_id", None)),
        "pad_token_id": _safe_int(getattr(tokenizer, "pad_token_id", None)),
        "unk_token_id": _safe_int(getattr(tokenizer, "unk_token_id", None)),
        "tokenizer_type": type(tokenizer).__name__,
    }


def validate_config_against_package(config: LlamaLikeConfig, package: QuantizedCheckpointPackage) -> AlignmentReport:
    config = config.validate()
    issues: list[AlignmentIssue] = []
    package_config = dict(getattr(package, "config", {}) or {})

    if not getattr(package, "format_version", None):
        issues.append(_issue("error", "UNSUPPORTED_PACKAGE_FORMAT", "package format_version is missing", field="format_version"))

    _compare_value(issues, "hidden_size", config.hidden_size, package_config.get("hidden_size"), "HIDDEN_SIZE_MISMATCH")
    _compare_value(
        issues,
        "intermediate_size",
        config.intermediate_size,
        package_config.get("intermediate_size"),
        "INTERMEDIATE_SIZE_MISMATCH",
    )
    _compare_value(
        issues,
        "num_attention_heads",
        config.num_attention_heads,
        package_config.get("num_attention_heads"),
        "PACKAGE_CONFIG_MISMATCH",
        message="package num_attention_heads does not match config",
    )
    _compare_value(
        issues,
        "num_key_value_heads",
        config.num_key_value_heads,
        package_config.get("num_key_value_heads"),
        "KV_HEAD_MISMATCH",
    )
    _compare_value(issues, "head_dim", config.head_dim, package_config.get("head_dim"), "HEAD_DIM_MISMATCH")
    _compare_value(
        issues,
        "max_position_embeddings",
        config.max_position_embeddings,
        package_config.get("max_position_embeddings"),
        "PACKAGE_CONFIG_MISMATCH",
        message="package max_position_embeddings does not match config",
    )
    _compare_value(
        issues,
        "model_type",
        config.model_type,
        package.model_type or package_config.get("model_type"),
        "PACKAGE_CONFIG_MISMATCH",
        message="package model_type does not match config",
    )

    package_num_layers = package_config.get("num_hidden_layers")
    explicit_partial = bool(package.metadata.get("partial") or package.metadata.get("allow_partial") or package_config.get("partial"))
    if package_num_layers is not None and int(package_num_layers) != config.num_hidden_layers:
        if explicit_partial:
            issues.append(
                _issue(
                    "info",
                    "OPTIONAL_COMPONENT_MISSING",
                    "package declares a partial layer set; skipping strict num_hidden_layers equality",
                    field="config.num_hidden_layers",
                    expected=config.num_hidden_layers,
                    actual=package_num_layers,
                )
            )
        else:
            issues.append(
                _issue(
                    "error",
                    "LAYER_COUNT_MISMATCH",
                    "package config num_hidden_layers does not match runtime config",
                    field="config.num_hidden_layers",
                    expected=config.num_hidden_layers,
                    actual=package_num_layers,
                )
            )
    actual_layer_count = len(package.layers)
    if actual_layer_count and actual_layer_count != config.num_hidden_layers:
        if explicit_partial:
            issues.append(
                _issue(
                    "info",
                    "OPTIONAL_COMPONENT_MISSING",
                    "package contains an explicit partial layer set",
                    field="layers",
                    expected=config.num_hidden_layers,
                    actual=actual_layer_count,
                )
            )
        else:
            issues.append(
                _issue(
                    "error",
                    "LAYER_COUNT_MISMATCH",
                    "package layer metadata count does not match config num_hidden_layers",
                    field="layers",
                    expected=config.num_hidden_layers,
                    actual=actual_layer_count,
                )
            )

    expected_shapes = {
        "qkv": (config.q_output_dim() + 2 * config.kv_output_dim(), config.hidden_size),
        "o_proj": (config.hidden_size, config.hidden_size),
        "gate_proj": (config.intermediate_size, config.hidden_size),
        "up_proj": (config.intermediate_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.intermediate_size),
    }
    for layer in package.layers:
        for tensor_name, expected_shape in expected_shapes.items():
            tensor = layer.tensors.get(tensor_name)
            if tensor is None:
                issues.append(
                    _issue(
                        "error",
                        "OPTIONAL_COMPONENT_MISSING" if explicit_partial else "PACKAGE_CONFIG_MISMATCH",
                        f"layer {layer.layer_idx} is missing required tensor metadata {tensor_name!r}",
                        field=f"layers[{layer.layer_idx}].{tensor_name}",
                    )
                )
                continue
            if tuple(tensor.original_shape) != tuple(expected_shape):
                issues.append(
                    _issue(
                        "error",
                        _shape_code_for_tensor(tensor_name),
                        f"layer {layer.layer_idx} tensor {tensor_name!r} original_shape does not match config",
                        field=f"layers[{layer.layer_idx}].{tensor_name}.original_shape",
                        expected=expected_shape,
                        actual=tuple(tensor.original_shape),
                    )
                )

    summary = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "num_layers": config.num_hidden_layers,
        "package_format_version": getattr(package, "format_version", None),
    }
    return AlignmentReport(ok=True, issues=issues, summary=summary)


def validate_tokenizer_against_package(tokenizer, package: QuantizedCheckpointPackage) -> AlignmentReport:
    issues: list[AlignmentIssue] = []
    info = tokenizer_alignment_info(tokenizer)
    vocab_size = info["vocab_size"]
    package_vocab_size = _safe_int(package.config.get("vocab_size")) if getattr(package, "config", None) else None

    if vocab_size is not None and package_vocab_size is not None and vocab_size != package_vocab_size:
        issues.append(
            _issue(
                "error",
                "VOCAB_SIZE_MISMATCH",
                "tokenizer vocab_size does not match package config vocab_size",
                field="vocab_size",
                expected=package_vocab_size,
                actual=vocab_size,
            )
        )

    _validate_token_ids(issues, info, vocab_size)
    _warn_for_missing_specials(issues, info)

    embedding = package.global_tensors.get("embedding")
    lm_head = package.global_tensors.get("lm_head")
    if embedding is not None and vocab_size is not None and _first_dim(embedding.original_shape) != vocab_size:
        issues.append(
            _issue(
                "error",
                "VOCAB_SIZE_MISMATCH",
                "embedding vocab dimension does not match tokenizer vocab_size",
                field="global_tensors.embedding.original_shape",
                expected=(vocab_size, _shape_tuple(embedding.original_shape)[1] if len(embedding.original_shape) > 1 else None),
                actual=tuple(embedding.original_shape),
            )
        )
    if lm_head is not None and vocab_size is not None and _first_dim(lm_head.original_shape) != vocab_size:
        issues.append(
            _issue(
                "error",
                "VOCAB_SIZE_MISMATCH",
                "lm_head vocab dimension does not match tokenizer vocab_size",
                field="global_tensors.lm_head.original_shape",
                expected=(vocab_size, _shape_tuple(lm_head.original_shape)[1] if len(lm_head.original_shape) > 1 else None),
                actual=tuple(lm_head.original_shape),
            )
        )

    return AlignmentReport(ok=True, issues=issues, summary={"vocab_size": vocab_size, "tokenizer_type": info["tokenizer_type"]})


def validate_tokenizer_against_config(
    tokenizer,
    config: LlamaLikeConfig,
    *,
    embedding=None,
    lm_head=None,
) -> AlignmentReport:
    config = config.validate()
    issues: list[AlignmentIssue] = []
    info = tokenizer_alignment_info(tokenizer)
    vocab_size = info["vocab_size"]
    if vocab_size is not None and config.vocab_size is not None and vocab_size != config.vocab_size:
        issues.append(
            _issue(
                "error",
                "VOCAB_SIZE_MISMATCH",
                "tokenizer vocab_size does not match config vocab_size",
                field="vocab_size",
                expected=config.vocab_size,
                actual=vocab_size,
            )
        )

    _validate_token_ids(issues, info, vocab_size)
    if info["eos_token_id"] is None:
        issues.append(
            _issue(
                "warning",
                "MISSING_EOS_TOKEN",
                "tokenizer has no eos_token_id; generation stopping may need an explicit eos override",
                field="eos_token_id",
            )
        )
    if info["bos_token_id"] is None:
        issues.append(
            _issue(
                "warning",
                "MISSING_BOS_TOKEN",
                "tokenizer has no bos_token_id; callers should decide whether prompts need an explicit BOS token",
                field="bos_token_id",
            )
        )

    _check_embedding_shape(issues, embedding, vocab_size or config.vocab_size, config.hidden_size, "embedding")
    _check_embedding_shape(issues, lm_head, vocab_size or config.vocab_size, config.hidden_size, "lm_head")

    return AlignmentReport(
        ok=True,
        issues=issues,
        summary={"vocab_size": vocab_size, "hidden_size": config.hidden_size, "tokenizer_type": info["tokenizer_type"]},
    )


def validate_quantization_alignment(
    package: QuantizedCheckpointPackage,
    *,
    bits: int | None = None,
    group_size: int | None = None,
) -> AlignmentReport:
    issues: list[AlignmentIssue] = []
    package_bits = _safe_int(package.quantization.get("bits")) if getattr(package, "quantization", None) else None
    package_group_size = _safe_int(package.quantization.get("group_size")) if getattr(package, "quantization", None) else None

    if bits is not None and package_bits is not None and bits != package_bits:
        issues.append(
            _issue(
                "error",
                "QUANT_BITS_MISMATCH",
                "requested bits do not match package quantization bits",
                field="quantization.bits",
                expected=bits,
                actual=package_bits,
            )
        )
    if group_size is not None and package_group_size is not None and group_size != package_group_size:
        issues.append(
            _issue(
                "error",
                "GROUP_SIZE_MISMATCH",
                "requested group_size does not match package quantization group_size",
                field="quantization.group_size",
                expected=group_size,
                actual=package_group_size,
            )
        )

    for field_name, tensor in _iter_quantized_tensors(package):
        if tensor.bits == 0:
            continue
        if package_bits is not None and tensor.bits != package_bits:
            issues.append(
                _issue(
                    "error",
                    "QUANT_BITS_MISMATCH",
                    f"tensor {tensor.name!r} bits do not match package-level bits",
                    field=f"{field_name}.bits",
                    expected=package_bits,
                    actual=tensor.bits,
                )
            )
        if package_group_size is not None and tensor.group_size != package_group_size:
            issues.append(
                _issue(
                    "error",
                    "GROUP_SIZE_MISMATCH",
                    f"tensor {tensor.name!r} group_size does not match package-level group_size",
                    field=f"{field_name}.group_size",
                    expected=package_group_size,
                    actual=tensor.group_size,
                )
            )
        if len(tensor.original_shape) < 2:
            continue
        out_dim, in_dim = int(tensor.original_shape[0]), int(tensor.original_shape[1])
        expected_packed = (out_dim, math.ceil(in_dim / 2)) if tensor.bits == 4 else (out_dim, in_dim)
        if tuple(tensor.packed_shape) != expected_packed:
            issues.append(
                _issue(
                    "error",
                    "PACKED_SHAPE_MISMATCH",
                    f"tensor {tensor.name!r} packed_shape does not match the repo's q{tensor.bits} convention",
                    field=f"{field_name}.packed_shape",
                    expected=expected_packed,
                    actual=tuple(tensor.packed_shape),
                )
            )
        expected_groups = math.ceil(in_dim / tensor.group_size)
        expected_scales = (out_dim, expected_groups)
        if tuple(tensor.scales_shape) != expected_scales:
            issues.append(
                _issue(
                    "error",
                    "SCALES_SHAPE_MISMATCH",
                    f"tensor {tensor.name!r} scales_shape does not match the expected groupwise layout",
                    field=f"{field_name}.scales_shape",
                    expected=expected_scales,
                    actual=tuple(tensor.scales_shape),
                )
            )
        if tensor.zeros_shape is not None and tuple(tensor.zeros_shape) != expected_scales:
            issues.append(
                _issue(
                    "error",
                    "ZEROS_SHAPE_MISMATCH",
                    f"tensor {tensor.name!r} zeros_shape does not match the expected groupwise layout",
                    field=f"{field_name}.zeros_shape",
                    expected=expected_scales,
                    actual=tuple(tensor.zeros_shape),
                )
            )

    return AlignmentReport(
        ok=True,
        issues=issues,
        summary={"bits": package_bits, "group_size": package_group_size, "tensor_count": package.tensor_count()},
    )


def validate_generation_alignment(
    *,
    tokenizer=None,
    config: LlamaLikeConfig | None = None,
    package: QuantizedCheckpointPackage | None = None,
    stack_weights=None,
    embedding=None,
    lm_head=None,
    bits: int | None = None,
    group_size: int | None = None,
) -> AlignmentReport:
    reports: list[AlignmentReport] = []
    effective_config = config.validate() if config is not None else None
    effective_embedding = embedding if embedding is not None else getattr(stack_weights, "embedding", None)
    effective_lm_head = lm_head if lm_head is not None else getattr(stack_weights, "lm_head", None)

    if tokenizer is not None and effective_config is not None:
        reports.append(
            validate_tokenizer_against_config(
                tokenizer,
                effective_config,
            )
        )
    if tokenizer is not None and package is not None:
        reports.append(validate_tokenizer_against_package(tokenizer, package))
    if effective_config is not None and package is not None:
        reports.append(validate_config_against_package(effective_config, package))
    if package is not None:
        reports.append(validate_quantization_alignment(package, bits=bits, group_size=group_size))
    if effective_config is not None and stack_weights is not None:
        reports.append(_validate_stack_weights_against_config(effective_config, stack_weights))
    if effective_config is not None:
        reports.append(
            _validate_embedding_and_lm_head(
                effective_config,
                embedding=effective_embedding,
                lm_head=effective_lm_head,
                tokenizer=tokenizer,
            )
        )

    issues: list[AlignmentIssue] = []
    for report in reports:
        issues.extend(report.issues)

    info = tokenizer_alignment_info(tokenizer)
    summary = {
        "vocab_size": info["vocab_size"] or (effective_config.vocab_size if effective_config is not None else None),
        "hidden_size": effective_config.hidden_size if effective_config is not None else None,
        "num_layers": effective_config.num_hidden_layers if effective_config is not None else (len(getattr(stack_weights, "layers", [])) if stack_weights is not None else None),
        "num_attention_heads": effective_config.num_attention_heads if effective_config is not None else None,
        "num_key_value_heads": effective_config.num_key_value_heads if effective_config is not None else None,
        "bits": bits if bits is not None else _safe_int(getattr(package, "quantization", {}).get("bits")) if package is not None else None,
        "group_size": group_size if group_size is not None else _safe_int(getattr(package, "quantization", {}).get("group_size")) if package is not None else None,
        "has_tokenizer": tokenizer is not None,
        "has_package": package is not None,
        "has_stack_weights": stack_weights is not None,
        "has_embedding": effective_embedding is not None,
        "has_lm_head": effective_lm_head is not None,
    }
    return AlignmentReport(ok=True, issues=issues, summary=summary)


def _validate_stack_weights_against_config(config: LlamaLikeConfig, stack_weights) -> AlignmentReport:
    issues: list[AlignmentIssue] = []
    layers = list(getattr(stack_weights, "layers", []) or [])
    if len(layers) != config.num_hidden_layers:
        issues.append(
            _issue(
                "error",
                "LAYER_COUNT_MISMATCH",
                "stack_weights layer count does not match config num_hidden_layers",
                field="stack_weights.layers",
                expected=config.num_hidden_layers,
                actual=len(layers),
            )
        )
    final_norm_shape = _shape_tuple(getattr(stack_weights, "final_norm_weight", None))
    if final_norm_shape is not None and final_norm_shape != (config.hidden_size,):
        issues.append(
            _issue(
                "error",
                "HIDDEN_SIZE_MISMATCH",
                "stack_weights final_norm_weight shape does not match hidden_size",
                field="stack_weights.final_norm_weight",
                expected=(config.hidden_size,),
                actual=final_norm_shape,
            )
        )
    for layer_idx, layer in enumerate(layers):
        layer_bits = _safe_int(getattr(layer, "bits", None))
        layer_group_size = _safe_int(getattr(layer, "group_size", None))
        if layer_bits is not None and layer_bits not in (4, 8):
            issues.append(
                _issue(
                    "error",
                    "QUANT_BITS_MISMATCH",
                    f"layer {layer_idx} uses unsupported quantization bits",
                    field=f"stack_weights.layers[{layer_idx}].bits",
                    expected="4 or 8",
                    actual=layer_bits,
                )
            )
        if layer_group_size is not None and layer_group_size <= 0:
            issues.append(
                _issue(
                    "error",
                    "GROUP_SIZE_MISMATCH",
                    f"layer {layer_idx} group_size must be positive",
                    field=f"stack_weights.layers[{layer_idx}].group_size",
                    expected="> 0",
                    actual=layer_group_size,
                )
            )
    return AlignmentReport(ok=True, issues=issues, summary={"num_layers": len(layers)})


def _validate_embedding_and_lm_head(config: LlamaLikeConfig, *, embedding=None, lm_head=None, tokenizer=None) -> AlignmentReport:
    issues: list[AlignmentIssue] = []
    info = tokenizer_alignment_info(tokenizer)
    expected_vocab = info["vocab_size"] or config.vocab_size

    if embedding is None:
        issues.append(
            _issue(
                "warning",
                "MISSING_EMBEDDING",
                "no embedding tensor is attached to the generation alignment context",
                field="embedding",
            )
        )
    else:
        _check_embedding_shape(issues, embedding, expected_vocab, config.hidden_size, "embedding")

    if lm_head is None:
        issues.append(
            _issue(
                "warning",
                "MISSING_LM_HEAD",
                "no lm_head tensor is attached to the generation alignment context",
                field="lm_head",
            )
        )
    else:
        _check_embedding_shape(issues, lm_head, expected_vocab, config.hidden_size, "lm_head")

    return AlignmentReport(ok=True, issues=issues, summary={"expected_vocab_size": expected_vocab, "hidden_size": config.hidden_size})


def _iter_quantized_tensors(package: QuantizedCheckpointPackage):
    for name, tensor in package.global_tensors.items():
        yield f"global_tensors.{name}", tensor
    for layer in package.layers:
        for name, tensor in layer.tensors.items():
            yield f"layers[{layer.layer_idx}].tensors.{name}", tensor


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    return None


def _first_dim(shape: tuple[int, ...] | list[int] | None) -> int | None:
    if not shape:
        return None
    return int(shape[0])


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    field: str | None = None,
    expected: Any | None = None,
    actual: Any | None = None,
) -> AlignmentIssue:
    return AlignmentIssue(
        severity=severity,
        code=code,
        message=message,
        field=field,
        expected=expected,
        actual=actual,
    )


def _compare_value(
    issues: list[AlignmentIssue],
    field: str,
    expected: Any,
    actual: Any,
    code: str,
    *,
    message: str | None = None,
) -> None:
    if actual is None:
        return
    if actual != expected:
        issues.append(
            _issue(
                "error",
                code,
                message or f"package {field} does not match config",
                field=field,
                expected=expected,
                actual=actual,
            )
        )


def _validate_token_ids(issues: list[AlignmentIssue], info: dict[str, Any], vocab_size: int | None) -> None:
    if vocab_size is None:
        return
    for field in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = info.get(field)
        if token_id is None:
            continue
        if token_id < 0 or token_id >= vocab_size:
            issues.append(
                _issue(
                    "error",
                    "TOKEN_ID_OUT_OF_RANGE",
                    f"{field} must be within [0, vocab_size)",
                    field=field,
                    expected=f"[0, {vocab_size})",
                    actual=token_id,
                )
            )


def _warn_for_missing_specials(issues: list[AlignmentIssue], info: dict[str, Any]) -> None:
    if info["eos_token_id"] is None:
        issues.append(_issue("warning", "MISSING_EOS_TOKEN", "tokenizer has no eos_token_id", field="eos_token_id"))
    if info["bos_token_id"] is None:
        issues.append(_issue("warning", "MISSING_BOS_TOKEN", "tokenizer has no bos_token_id", field="bos_token_id"))
    if info["pad_token_id"] is None:
        issues.append(_issue("warning", "OPTIONAL_COMPONENT_MISSING", "tokenizer has no pad_token_id", field="pad_token_id"))


def _check_embedding_shape(
    issues: list[AlignmentIssue],
    tensor: Any,
    expected_vocab_size: int | None,
    hidden_size: int,
    field_name: str,
) -> None:
    if tensor is None:
        return
    shape = _shape_tuple(tensor)
    if shape is None or len(shape) != 2:
        issues.append(
            _issue(
                "error",
                "PACKAGE_CONFIG_MISMATCH",
                f"{field_name} must use the documented [vocab_size, hidden_size] convention",
                field=field_name,
                expected="[vocab_size, hidden_size]",
                actual=shape,
            )
        )
        return
    if expected_vocab_size is not None and shape[0] != expected_vocab_size:
        issues.append(
            _issue(
                "error",
                "VOCAB_SIZE_MISMATCH",
                f"{field_name} vocab dimension does not match the expected vocab_size",
                field=field_name,
                expected=expected_vocab_size,
                actual=shape[0],
            )
        )
    if shape[1] != hidden_size:
        issues.append(
            _issue(
                "error",
                "HIDDEN_SIZE_MISMATCH",
                f"{field_name} hidden dimension does not match config hidden_size",
                field=field_name,
                expected=hidden_size,
                actual=shape[1],
            )
        )


def _shape_code_for_tensor(tensor_name: str) -> str:
    return {
        "qkv": "HEAD_DIM_MISMATCH",
        "o_proj": "HIDDEN_SIZE_MISMATCH",
        "gate_proj": "INTERMEDIATE_SIZE_MISMATCH",
        "up_proj": "INTERMEDIATE_SIZE_MISMATCH",
        "down_proj": "INTERMEDIATE_SIZE_MISMATCH",
    }.get(tensor_name, "PACKAGE_CONFIG_MISMATCH")
