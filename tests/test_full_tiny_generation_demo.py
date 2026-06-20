from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_tiny_generation_demo_exits_zero():
    script = ROOT / "examples" / "full_tiny_generation_demo.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--prompt", "Hi", "--max-new-tokens", "2", "--greedy", "--backend-preset", "reference"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.lower()
    assert "synthetic random weights" in stdout
    assert "generated ids:" in stdout


def test_full_tiny_generation_with_package_demo_exits_zero():
    script = ROOT / "examples" / "full_tiny_generation_with_package_demo.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.lower()
    assert "package fallback:" in stdout
    assert "metadata-only package" in stdout
