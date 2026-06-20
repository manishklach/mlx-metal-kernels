from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_CLI_PATH = str(Path(__file__).resolve().parents[1] / "scripts" / "convert_checkpoint.py")
_INSPECT_PATH = str(Path(__file__).resolve().parents[1] / "scripts" / "inspect_quantized_package.py")


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


def _run_inspect(*args, expect_code=0):
    result = subprocess.run(
        [sys.executable, _INSPECT_PATH, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expect_code is not None:
        assert result.returncode == expect_code, f"inspect failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
    return result


class TestConvertCheckpointCLI:
    def test_synthetic_demo_creates_output(self, tmp_path):
        out = tmp_path / "package.json"
        result = _run_cli("--synthetic-demo", "--output", str(out))
        assert out.exists()

    def test_synthetic_demo_exit_zero(self, tmp_path):
        out = tmp_path / "package.json"
        result = _run_cli("--synthetic-demo", "--output", str(out))
        assert result.returncode == 0

    def test_synthetic_demo_with_bits_8(self, tmp_path):
        out = tmp_path / "package.json"
        _run_cli("--synthetic-demo", "--bits", "8", "--output", str(out))
        assert out.exists()

    def test_inspect_valid_package(self, tmp_path):
        out = tmp_path / "package.json"
        _run_cli("--synthetic-demo", "--output", str(out))
        result = _run_inspect(str(out))
        assert "Format version" in result.stdout
        assert "Num layers" in result.stdout

    def test_inspect_with_verbose(self, tmp_path):
        out = tmp_path / "package.json"
        _run_cli("--synthetic-demo", "--output", str(out))
        result = _run_inspect(str(out), "--verbose")
        assert "Layer" in result.stdout

    def test_no_args_exits_error(self):
        result = _run_cli(expect_code=1)
        assert result.returncode == 1

    def test_synthetic_demo_with_layers_selection(self, tmp_path):
        out = tmp_path / "package.json"
        result = _run_cli("--synthetic-demo", "--layers", "0", "--output", str(out))
        assert result.returncode == 0

    def test_inspect_missing_file_exits_error(self):
        result = _run_inspect("/nonexistent/path.json", expect_code=1)
        assert result.returncode == 1
