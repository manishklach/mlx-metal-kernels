from __future__ import annotations

import pytest

from models import (
    CharTokenizer,
    QuantizedCheckpointPackage,
    SmokeTestConfig,
    SmokeTestIssue,
    SmokeTestReport,
    TinyGenerationPipelineConfig,
    inspect_package_executability,
    run_local_smoke_test,
)


def _metadata_only_package() -> QuantizedCheckpointPackage:
    config = TinyGenerationPipelineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=32,
        vocab_size=CharTokenizer().vocab_size,
        backend_preset="reference",
    ).to_llama_config()
    return QuantizedCheckpointPackage(
        model_type=config.model_type,
        config=config.to_dict(),
        quantization={"bits": 4, "group_size": 32},
        layers=[],
        metadata={"test": "metadata-only"},
    )


def test_smoke_test_config_validates():
    config = SmokeTestConfig(max_new_tokens=2, backend_preset="reference").validate()
    assert config.max_new_tokens == 2


def test_smoke_test_report_filters_by_severity():
    report = SmokeTestReport(
        ok=True,
        mode="dry_run",
        issues=[
            SmokeTestIssue("error", "A", "bad"),
            SmokeTestIssue("warning", "B", "warn"),
            SmokeTestIssue("info", "C", "info"),
        ],
    )
    assert len(report.errors()) == 1
    assert len(report.warnings()) == 1
    assert len(report.infos()) == 1


def test_inspect_package_executability_returns_false_for_metadata_only_package(tmp_path):
    package = _metadata_only_package()
    path = tmp_path / "package.json"
    package.save_json(path)
    info = inspect_package_executability(package, path)
    assert info["executable"] is False
    assert info["tensor_data_files_present"] is False


def test_run_local_smoke_test_with_synthetic_fallback_and_dry_run_succeeds():
    report = run_local_smoke_test(
        SmokeTestConfig(
            dry_run=True,
            synthetic_fallback=True,
            backend_preset="reference",
        )
    )
    assert report.ok
    assert any(issue.code == "DRY_RUN_ONLY" for issue in report.infos())


def test_run_local_smoke_test_with_synthetic_fallback_and_no_dry_run_generates():
    report = run_local_smoke_test(
        SmokeTestConfig(
            dry_run=False,
            synthetic_fallback=True,
            prompt="Hi",
            max_new_tokens=2,
            backend_preset="reference",
        )
    )
    assert report.ok
    assert report.generation_summary["num_generated_tokens"] == 2
    assert any(issue.code == "GENERATION_SUCCEEDED" for issue in report.infos())


def test_run_local_smoke_test_metadata_only_package_with_require_tensor_data_reports_missing(tmp_path):
    path = tmp_path / "package.json"
    _metadata_only_package().save_json(path)
    report = run_local_smoke_test(
        SmokeTestConfig(
            package_path=str(path),
            dry_run=False,
            require_tensor_data=True,
            synthetic_fallback=False,
            backend_preset="reference",
        )
    )
    assert not report.ok
    assert any(issue.code == "TENSOR_DATA_MISSING" for issue in report.errors())


def test_run_local_smoke_test_with_missing_package_reports_package_missing():
    report = run_local_smoke_test(
        SmokeTestConfig(
            package_path="missing.json",
            dry_run=True,
            synthetic_fallback=False,
            backend_preset="reference",
        )
    )
    assert not report.ok
    assert any(issue.code == "PACKAGE_MISSING" for issue in report.errors())


def test_tokenizer_missing_produces_clear_warning_or_error(tmp_path):
    path = tmp_path / "package.json"
    _metadata_only_package().save_json(path)
    report = run_local_smoke_test(
        SmokeTestConfig(
            package_path=str(path),
            dry_run=True,
            synthetic_fallback=False,
            backend_preset="reference",
        )
    )
    assert any(issue.code == "TOKENIZER_MISSING" for issue in report.issues)


def test_alignment_failure_is_surfaced_for_mismatched_package_and_tokenizer(tmp_path):
    package = _metadata_only_package()
    package.config["vocab_size"] = 5
    path = tmp_path / "package.json"
    package.save_json(path)
    report = run_local_smoke_test(
        SmokeTestConfig(
            package_path=str(path),
            dry_run=True,
            synthetic_fallback=True,
            backend_preset="reference",
        )
    )
    assert not report.ok
    assert any(issue.code == "ALIGNMENT_FAILED" for issue in report.errors())


def test_smoke_test_report_to_dict_works():
    report = run_local_smoke_test(
        SmokeTestConfig(
            dry_run=True,
            synthetic_fallback=True,
            backend_preset="reference",
        )
    )
    payload = report.to_dict()
    assert payload["mode"] == "dry_run"
    assert isinstance(payload["issues"], list)
