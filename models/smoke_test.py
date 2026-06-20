from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .alignment import tokenizer_alignment_info, validate_config_against_package, validate_generation_alignment
from .quantized_package_io import QuantizedCheckpointPackage
from .tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig
from .tokenization import CharTokenizer
from .tokenizer_adapters import OptionalDependencyError, TokenizerAdapterFactory


@dataclass
class SmokeTestConfig:
    package_path: str | None = None
    tokenizer_path: str | None = None
    tokenizer_kind: str | None = None
    prompt: str = "Hello"
    max_new_tokens: int = 4
    backend_preset: str = "fused_experimental"
    bits: int | None = None
    group_size: int | None = None
    dry_run: bool = True
    synthetic_fallback: bool = False
    require_tensor_data: bool = False
    validate_alignment: bool = True
    use_prefill: bool = True
    seed: int = 0

    def validate(self) -> SmokeTestConfig:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.backend_preset not in ("reference", "metal", "tiled", "fused_experimental"):
            raise ValueError(
                "backend_preset must be one of ('reference', 'metal', 'tiled', 'fused_experimental')"
            )
        if self.bits is not None and self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8 when provided, got {self.bits}")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError(f"group_size must be positive when provided, got {self.group_size}")
        return self


@dataclass
class SmokeTestIssue:
    severity: str
    code: str
    message: str
    field: str | None = None


@dataclass
class SmokeTestReport:
    ok: bool
    mode: str
    issues: list[SmokeTestIssue]
    alignment_report: Any | None = None
    package_summary: dict[str, Any] = field(default_factory=dict)
    tokenizer_summary: dict[str, Any] = field(default_factory=dict)
    generation_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ok = not any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> list[SmokeTestIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[SmokeTestIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def infos(self) -> list[SmokeTestIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError(self.pretty_print())

    def to_dict(self) -> dict[str, Any]:
        alignment_report = self.alignment_report.to_dict() if hasattr(self.alignment_report, "to_dict") else self.alignment_report
        return {
            "ok": self.ok,
            "mode": self.mode,
            "issues": [asdict(issue) for issue in self.issues],
            "alignment_report": alignment_report,
            "package_summary": dict(self.package_summary),
            "tokenizer_summary": dict(self.tokenizer_summary),
            "generation_summary": dict(self.generation_summary),
        }

    def pretty_print(self) -> str:
        lines = [f"Smoke test mode: {self.mode}", f"Smoke test status: {'ok' if self.ok else 'error'}"]
        if self.package_summary:
            lines.append(f"Package summary: {self.package_summary}")
        if self.tokenizer_summary:
            lines.append(f"Tokenizer summary: {self.tokenizer_summary}")
        if self.generation_summary:
            lines.append(f"Generation summary: {self.generation_summary}")
        if self.alignment_report is not None and hasattr(self.alignment_report, "pretty_print"):
            lines.append("Alignment report:")
            lines.append(self.alignment_report.pretty_print())
        if not self.issues:
            lines.append("No smoke test issues.")
            return "\n".join(lines)
        for issue in self.issues:
            detail = f"[{issue.severity}] {issue.code}: {issue.message}"
            if issue.field is not None:
                detail += f" (field={issue.field})"
            lines.append(detail)
        return "\n".join(lines)


def inspect_package_executability(
    package: QuantizedCheckpointPackage,
    package_path: str | Path | None = None,
    *,
    check_checksums: bool = False,
) -> dict[str, Any]:
    from .tensor_data_io import compute_file_checksum

    base_dir = Path(package_path).resolve().parent if package_path is not None else None
    has_layers = bool(package.layers)
    has_global_tensors = bool(package.global_tensors)
    has_embedding_metadata = "embedding" in package.global_tensors
    has_lm_head_metadata = "lm_head" in package.global_tensors
    missing_tensor_data: list[str] = []
    present_tensor_data: list[str] = []
    checksum_mismatches: list[str] = []
    validated_checksums = 0

    for field_name, tensor in _iter_package_tensors(package):
        is_norm = tensor.bits == 0
        if is_norm:
            attr_names = ("data_file",)
        else:
            attr_names = ("data_file", "scales_file", "zeros_file")
        for attr_name in attr_names:
            rel_path = getattr(tensor, attr_name, None)
            if rel_path is None:
                if is_norm and attr_name != "data_file":
                    continue
                missing_tensor_data.append(f"{field_name}.{attr_name}")
                continue
            resolved = Path(rel_path)
            if not resolved.is_absolute() and base_dir is not None:
                resolved = base_dir / resolved
            if resolved.exists():
                present_tensor_data.append(str(resolved))
                if check_checksums and attr_name == "data_file" and tensor.checksum is not None:
                    try:
                        actual = compute_file_checksum(resolved)
                        validated_checksums += 1
                        if actual != tensor.checksum:
                            checksum_mismatches.append(
                                f"{field_name}.data_file: expected {tensor.checksum}, got {actual}"
                            )
                    except Exception as exc:
                        checksum_mismatches.append(
                            f"{field_name}.data_file: checksum error: {exc}"
                        )
            else:
                missing_tensor_data.append(f"{field_name}.{attr_name}:{resolved}")

    tensor_data_files_present = len(present_tensor_data) > 0 and len(missing_tensor_data) == 0
    return {
        "has_metadata": bool(package.format_version),
        "has_layers": has_layers,
        "has_global_tensors": has_global_tensors,
        "has_embedding_metadata": has_embedding_metadata,
        "has_lm_head_metadata": has_lm_head_metadata,
        "tensor_data_files_present": tensor_data_files_present,
        "missing_tensor_data": missing_tensor_data,
        "present_tensor_data": present_tensor_data,
        "checksum_mismatches": checksum_mismatches,
        "validated_checksums": validated_checksums,
        "executable": bool(has_layers and tensor_data_files_present and not checksum_mismatches),
    }


def load_optional_local_tokenizer(path=None, kind=None, fallback_char: bool = False):
    if path is None:
        if kind in ("char", "whitespace"):
            return TokenizerAdapterFactory.from_file(None, kind=kind)
        return CharTokenizer() if fallback_char else None
    resolved_kind = None if kind in (None, "auto") else kind
    return TokenizerAdapterFactory.from_file(path, kind=resolved_kind)


def run_local_smoke_test(config: SmokeTestConfig) -> SmokeTestReport:
    config = config.validate()
    mode = "dry_run" if config.dry_run else "generation"
    issues: list[SmokeTestIssue] = []
    package = None
    package_summary: dict[str, Any] = {}
    tokenizer_summary: dict[str, Any] = {}
    generation_summary: dict[str, Any] = {}
    alignment_report = None
    effective_tokenizer = None
    effective_llama_config = None
    executability = None

    if config.package_path is not None:
        pkg_path = Path(config.package_path)
        if not pkg_path.exists():
            _add_issue(
                issues,
                "error" if not config.synthetic_fallback else "warning",
                "PACKAGE_MISSING",
                f"package file not found: {pkg_path}",
                field="package_path",
            )
        else:
            try:
                package = QuantizedCheckpointPackage.load_json(str(pkg_path))
                package.validate(allow_partial=True)
                package_summary = package.summary()
                executability = inspect_package_executability(package, pkg_path)
                package_summary["executability"] = executability
                if not executability["executable"]:
                    _add_issue(
                        issues,
                        "warning" if config.dry_run or config.synthetic_fallback else "error",
                        "PACKAGE_METADATA_ONLY",
                        "package metadata is present but executable tensor data is not fully available",
                        field="package_path",
                    )
            except Exception as exc:  # noqa: BLE001
                _add_issue(
                    issues,
                    "error",
                    "UNSUPPORTED_PACKAGE_FORMAT",
                    f"failed to load package metadata: {exc}",
                    field="package_path",
                )
    elif not config.synthetic_fallback:
        _add_issue(
            issues,
            "warning" if config.dry_run else "error",
            "PACKAGE_MISSING",
            "no package_path was provided",
            field="package_path",
        )

    try:
        effective_tokenizer = load_optional_local_tokenizer(
            config.tokenizer_path,
            kind=config.tokenizer_kind,
            fallback_char=config.synthetic_fallback and config.tokenizer_path is None,
        )
    except OptionalDependencyError as exc:
        _add_issue(
            issues,
            "error",
            "OPTIONAL_DEPENDENCY_MISSING",
            str(exc),
            field="tokenizer_path",
        )
    except Exception as exc:  # noqa: BLE001
        _add_issue(
            issues,
            "error",
            "TOKENIZER_MISSING",
            f"failed to load tokenizer: {exc}",
            field="tokenizer_path",
        )

    if effective_tokenizer is None:
        if config.synthetic_fallback:
            effective_tokenizer = CharTokenizer()
            _add_issue(
                issues,
                "info",
                "SYNTHETIC_FALLBACK_USED",
                "using CharTokenizer because synthetic_fallback was requested",
                field="tokenizer_path",
            )
        else:
            _add_issue(
                issues,
                "warning" if config.dry_run and not config.require_tensor_data else "error",
                "TOKENIZER_MISSING",
                "no local tokenizer was provided",
                field="tokenizer_path",
            )

    if effective_tokenizer is not None:
        tokenizer_summary = tokenizer_alignment_info(effective_tokenizer)

    if package is not None and getattr(package, "config", None):
        try:
            effective_llama_config = _config_from_package(package)
        except Exception as exc:  # noqa: BLE001
            _add_issue(
                issues,
                "error",
                "UNSUPPORTED_PACKAGE_FORMAT",
                f"package config could not be converted into LlamaLikeConfig: {exc}",
                field="config",
            )

    if config.validate_alignment:
        if package is not None and effective_llama_config is not None:
            if effective_tokenizer is not None:
                alignment_report = validate_generation_alignment(
                    tokenizer=effective_tokenizer,
                    config=effective_llama_config,
                    package=package,
                    bits=config.bits,
                    group_size=config.group_size,
                )
            else:
                alignment_report = validate_config_against_package(effective_llama_config, package)
            if alignment_report is not None and getattr(alignment_report, "ok", True) is False:
                _add_issue(
                    issues,
                    "error",
                    "ALIGNMENT_FAILED",
                    "alignment validation reported one or more errors",
                    field="alignment_report",
                )

    if config.dry_run:
        _add_issue(
            issues,
            "info",
            "DRY_RUN_ONLY",
            "dry-run mode does not attempt generation",
        )
        return SmokeTestReport(
            ok=True,
            mode=mode,
            issues=issues,
            alignment_report=alignment_report,
            package_summary=package_summary,
            tokenizer_summary=tokenizer_summary,
            generation_summary=generation_summary,
        )

    if package is not None and executability is not None and not executability["executable"]:
        if config.require_tensor_data:
            _add_issue(
                issues,
                "error",
                "TENSOR_DATA_MISSING",
                "tensor data is required for execution but the package is metadata-only or incomplete",
                field="package_path",
            )
            _add_issue(
                issues,
                "info",
                "GENERATION_SKIPPED",
                "generation was skipped because tensor data is missing",
            )
            return SmokeTestReport(
                ok=True,
                mode=mode,
                issues=issues,
                alignment_report=alignment_report,
                package_summary=package_summary,
                tokenizer_summary=tokenizer_summary,
                generation_summary=generation_summary,
            )
        if not config.synthetic_fallback:
            _add_issue(
                issues,
                "error",
                "TENSOR_DATA_MISSING",
                "metadata-only package cannot run generation without --synthetic-fallback",
                field="package_path",
            )
            _add_issue(
                issues,
                "info",
                "GENERATION_SKIPPED",
                "generation was skipped because package tensor-data loading is not implemented",
            )
            return SmokeTestReport(
                ok=True,
                mode=mode,
                issues=issues,
                alignment_report=alignment_report,
                package_summary=package_summary,
                tokenizer_summary=tokenizer_summary,
                generation_summary=generation_summary,
            )

    if package is not None and executability is not None and executability["executable"]:
        _add_issue(
            issues,
            "error",
            "TENSOR_DATA_MISSING",
            "Quantized package tensor-data loading is not implemented yet.",
            field="package_path",
        )
        _add_issue(
            issues,
            "info",
            "GENERATION_SKIPPED",
            "generation was skipped because package tensor-data loading is not implemented",
        )
        return SmokeTestReport(
            ok=True,
            mode=mode,
            issues=issues,
            alignment_report=alignment_report,
            package_summary=package_summary,
            tokenizer_summary=tokenizer_summary,
            generation_summary=generation_summary,
        )

    if not config.synthetic_fallback:
        _add_issue(
            issues,
            "error",
            "GENERATION_SKIPPED",
            "generation requested without executable tensor data or synthetic fallback",
        )
        return SmokeTestReport(
            ok=True,
            mode=mode,
            issues=issues,
            alignment_report=alignment_report,
            package_summary=package_summary,
            tokenizer_summary=tokenizer_summary,
            generation_summary=generation_summary,
        )

    _add_issue(
        issues,
        "warning",
        "SYNTHETIC_FALLBACK_USED",
        "using synthetic TinyGenerationPipeline with random weights; output is not meaningful language",
    )
    pipeline_config = _pipeline_config_from_context(
        effective_llama_config,
        tokenizer=effective_tokenizer or CharTokenizer(),
        backend_preset=config.backend_preset,
        bits=config.bits,
        group_size=config.group_size,
        use_prefill=config.use_prefill,
    )
    pipeline = TinyGenerationPipeline(
        config=pipeline_config,
        tokenizer=effective_tokenizer or CharTokenizer(),
    )
    result = pipeline.generate(
        config.prompt,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        greedy=True,
        validate_alignment=True,
    )
    generation_summary = {
        "generated_ids": list(result.generated_ids),
        "text": result.text,
        "num_generated_tokens": len(result.generated_ids),
        "backend_preset": result.backend_preset,
    }
    _add_issue(
        issues,
        "info",
        "GENERATION_SUCCEEDED",
        "synthetic fallback generation completed successfully",
    )
    return SmokeTestReport(
        ok=True,
        mode=mode,
        issues=issues,
        alignment_report=alignment_report,
        package_summary=package_summary,
        tokenizer_summary=tokenizer_summary,
        generation_summary=generation_summary,
    )


def _iter_package_tensors(package: QuantizedCheckpointPackage):
    for name, tensor in package.global_tensors.items():
        yield f"global_tensors.{name}", tensor
    for layer in package.layers:
        for name, tensor in layer.tensors.items():
            yield f"layers[{layer.layer_idx}].tensors.{name}", tensor


def _config_from_package(package: QuantizedCheckpointPackage):
    from .llama_config import LlamaLikeConfig

    return LlamaLikeConfig.from_dict(dict(package.config))


def _pipeline_config_from_context(
    llama_config,
    *,
    tokenizer,
    backend_preset: str,
    bits: int | None,
    group_size: int | None,
    use_prefill: bool,
) -> TinyGenerationPipelineConfig:
    if llama_config is None:
        fallback = TinyGenerationPipelineConfig(
            backend_preset=backend_preset,
            bits=bits if bits is not None else 4,
            group_size=group_size if group_size is not None else 32,
            use_prefill=use_prefill,
            vocab_size=int(getattr(tokenizer, "vocab_size", TinyGenerationPipelineConfig().vocab_size)),
        ).validate()
        return fallback
    vocab_size = max(int(getattr(tokenizer, "vocab_size", 0) or 0), int(getattr(llama_config, "vocab_size", 0) or 0))
    return TinyGenerationPipelineConfig(
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_attention_heads=llama_config.num_attention_heads,
        num_key_value_heads=llama_config.num_key_value_heads,
        head_dim=llama_config.head_dim,
        num_hidden_layers=llama_config.num_hidden_layers,
        max_position_embeddings=llama_config.max_position_embeddings,
        vocab_size=vocab_size,
        bits=bits if bits is not None else 4,
        group_size=group_size if group_size is not None else 32,
        backend_preset=backend_preset,
        use_prefill=use_prefill,
    ).validate()


def _add_issue(issues: list[SmokeTestIssue], severity: str, code: str, message: str, field: str | None = None) -> None:
    issues.append(SmokeTestIssue(severity=severity, code=code, message=message, field=field))
