from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "ops" / "autotune_ops.py"
    spec = importlib.util.spec_from_file_location("autotune_ops_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_tuning_key_is_deterministic():
    module = _load_module()
    shape = {"B": 1, "D": 64, "H": 8}
    extra = {"length": 128, "group_size": 32}
    key1 = module.make_tuning_key("decode_attention", shape, "float16", extra=extra)
    key2 = module.make_tuning_key("decode_attention", dict(reversed(list(shape.items()))), "float16", extra=dict(reversed(list(extra.items()))))
    assert key1 == key2


def test_save_and_load_roundtrip(tmp_path):
    module = _load_module()
    path = tmp_path / "autotune.json"
    payload = {"version": 1, "entries": {"k": {"best_backend": "metal"}}}
    module.save_autotune_results(payload, path=path)
    loaded = module.load_autotune_results(path=path)
    assert loaded == payload


def test_record_and_lookup_best_backend(tmp_path):
    module = _load_module()
    path = tmp_path / "autotune.json"
    shape = {"B": 1, "MAX_S": 128, "H": 8, "D": 64, "length": 128}
    module.record_best_backend(
        "decode_attention",
        shape,
        "float16",
        "metal_threadgroup",
        {"metal_threadgroup": {"mean_ms": 1.23}, "metal": {"mean_ms": 1.56}},
        path=path,
        extra={"length": 128},
    )
    assert module.lookup_best_backend("decode_attention", shape, "float16", path=path, extra={"length": 128}) == "metal_threadgroup"


def test_select_backend_prefers_saved_then_default(tmp_path):
    module = _load_module()
    path = tmp_path / "autotune.json"
    shape = {"B": 1, "S": 64, "H": 4, "D": 64}
    assert module.select_backend("fast_attention", shape, "float16", path=path) == "baseline"
    module.record_best_backend(
        "fast_attention",
        shape,
        "float16",
        "threadgroup",
        {"threadgroup": {"mean_ms": 0.91}},
        path=path,
    )
    assert module.select_backend("fast_attention", shape, "float16", path=path) == "threadgroup"


def test_select_backend_require_tuned_raises(tmp_path):
    module = _load_module()
    with pytest.raises(KeyError, match="No saved autotune backend"):
        module.select_backend("q4_matvec_decode", {"B": 1, "K": 512, "N": 512, "group_size": 32}, "float16", path=tmp_path / "missing.json", require_tuned=True)


def test_explain_backend_choice_returns_useful_text(tmp_path):
    module = _load_module()
    path = tmp_path / "autotune.json"
    shape = {"B": 1, "K": 512, "N": 512, "group_size": 32}
    default_text = module.explain_backend_choice("q4_matvec_decode", shape, "float16", path=path)
    assert "conservative default" in default_text
    module.record_best_backend("q4_matvec_decode", shape, "float16", "metal_tiled", {"metal_tiled": {"mean_ms": 0.5}}, path=path, extra={"bits": 4, "group_size": 32})
    tuned_text = module.explain_backend_choice("q4_matvec_decode", shape, "float16", path=path, extra={"bits": 4, "group_size": 32})
    assert "autotuned backend" in tuned_text
