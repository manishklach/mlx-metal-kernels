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
    assert "quantized_mlp_block" in ops
    assert "gqa_attention" in ops
    assert "llama_layer_decode" in ops
    assert "llama_stack_decode" in ops


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


def test_filter_backends_for_shape_quantized_mlp_fused_experimental():
    module = _load_module()
    backends = ["reference", "tiled", "fused_experimental"]
    filtered_fp16 = module.filter_backends_for_shape("quantized_mlp_block", {"bits": 4}, "float16", backends)
    assert "fused_experimental" in filtered_fp16

    filtered_bf16 = module.filter_backends_for_shape("quantized_mlp_block", {"bits": 4}, "bfloat16", backends)
    assert "fused_experimental" not in filtered_bf16


def test_filter_backends_for_shape_gqa_attention():
    module = _load_module()
    backends = ["reference", "metal_gqa", "metal_gqa_threadgroup"]
    filtered = module.filter_backends_for_shape("gqa_attention", {"Hq": 4, "Hkv": 2, "D": 64}, "float16", backends)
    assert "metal_gqa" in filtered
    assert "metal_gqa_threadgroup" in filtered

    invalid = module.filter_backends_for_shape("gqa_attention", {"Hq": 3, "Hkv": 2, "D": 64}, "float16", backends)
    assert invalid == ["reference"]


def test_filter_backends_for_shape_llama_layer_decode():
    module = _load_module()
    backends = ["reference", "metal", "tiled", "fused_experimental"]
    filtered = module.filter_backends_for_shape("llama_layer_decode", {"bits": 4, "D": 64}, "float16", backends)
    assert "fused_experimental" in filtered
    filtered_bf16 = module.filter_backends_for_shape("llama_layer_decode", {"bits": 4, "D": 64}, "bfloat16", backends)
    assert "fused_experimental" not in filtered_bf16


def test_filter_backends_for_shape_llama_stack_decode():
    module = _load_module()
    backends = ["reference", "metal", "tiled", "fused_experimental"]
    filtered = module.filter_backends_for_shape("llama_stack_decode", {"bits": 4, "cache": "contiguous"}, "float16", backends)
    assert "fused_experimental" in filtered
    filtered_bf16 = module.filter_backends_for_shape("llama_stack_decode", {"bits": 4, "cache": "contiguous"}, "bfloat16", backends)
    assert "fused_experimental" not in filtered_bf16
    filtered_paged = module.filter_backends_for_shape("llama_stack_decode", {"bits": 4, "cache": "paged"}, "float16", backends)
    assert filtered_paged == []
