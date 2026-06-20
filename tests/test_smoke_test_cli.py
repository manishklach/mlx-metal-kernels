from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from models import CharTokenizer, QuantizedCheckpointPackage, TinyGenerationPipelineConfig

_CLI_PATH = str(Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_local_model.py")


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


def _run_cli(*args, expect_code=0):
    result = subprocess.run(
        [sys.executable, _CLI_PATH, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expect_code is not None:
        assert result.returncode == expect_code, f"CLI failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
    return result


def test_smoke_test_cli_synthetic_fallback_dry_run_exits_zero():
    result = _run_cli("--synthetic-fallback", "--dry-run", "--backend-preset", "reference")
    assert "DRY_RUN_ONLY" in result.stdout


def test_smoke_test_cli_synthetic_fallback_no_dry_run_exits_zero():
    result = _run_cli(
        "--synthetic-fallback",
        "--no-dry-run",
        "--prompt",
        "Hi",
        "--max-new-tokens",
        "2",
        "--backend-preset",
        "reference",
    )
    assert "GENERATION_SUCCEEDED" in result.stdout


def test_smoke_test_cli_missing_package_exits_non_zero():
    result = _run_cli("--package", "missing.json", "--dry-run", expect_code=1)
    assert "PACKAGE_MISSING" in result.stdout


def test_smoke_test_cli_json_output_is_parseable():
    result = _run_cli("--synthetic-fallback", "--dry-run", "--json", "--backend-preset", "reference")
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["ok"] is True


def test_smoke_test_cli_metadata_only_package_with_require_tensor_data_reports_missing(tmp_path):
    path = tmp_path / "package.json"
    _metadata_only_package().save_json(path)
    result = _run_cli(
        "--package",
        str(path),
        "--no-dry-run",
        "--require-tensor-data",
        "--tokenizer-kind",
        "char",
        "--backend-preset",
        "reference",
        expect_code=1,
    )
    assert "TENSOR_DATA_MISSING" in result.stdout
