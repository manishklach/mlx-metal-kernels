from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "backend_registry.py"
    spec = importlib.util.spec_from_file_location("backend_registry_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_list_ops_includes_expected_ops():
    module = _load_module()
    ops = module.list_ops()
    assert "fast_attention" in ops
    assert "decode_attention" in ops
    assert "paged_decode_attention" in ops
    assert "q4_matvec_decode" in ops
    assert "q8_matvec_decode" in ops


def test_get_candidate_backends_and_validate():
    module = _load_module()
    candidates = module.get_candidate_backends("fast_attention")
    assert "baseline" in candidates
    assert "simdgroup_d64" in candidates
    assert module.validate_backend("fast_attention", "baseline") == "baseline"
    with pytest.raises(ValueError, match="Unsupported backend"):
        module.validate_backend("fast_attention", "not_a_backend")


def test_filter_backends_for_shape_d64_and_d128():
    module = _load_module()
    backends = ["baseline", "baseline_d64", "baseline_d128", "simdgroup_d64"]
    d64 = module.filter_backends_for_shape("fast_attention", {"D": 64}, "float16", backends)
    assert "baseline_d64" in d64
    assert "simdgroup_d64" in d64
    assert "baseline_d128" not in d64

    d128 = module.filter_backends_for_shape("fast_attention", {"D": 128}, "float16", backends)
    assert "baseline_d128" in d128
    assert "baseline_d64" not in d128
    assert "simdgroup_d64" not in d128


def test_filter_backends_for_shape_removes_simdgroup_for_bf16():
    module = _load_module()
    backends = ["baseline", "simdgroup_d64"]
    filtered = module.filter_backends_for_shape("fast_attention", {"D": 64}, "bfloat16", backends)
    assert "baseline" in filtered
    assert "simdgroup_d64" not in filtered
