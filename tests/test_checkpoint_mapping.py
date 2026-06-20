import pytest

from models.checkpoint_manifest import CheckpointManifest
from models.checkpoint_mapping import (
    build_llama_name_map,
    llama_layer_tensor_names,
    missing_required_tensors,
    validate_llama_checkpoint_shapes,
    validate_llama_layer_shapes,
)
from models.llama_config import tiny_debug_config, tiny_gqa_debug_config


def _make_manifest(config, *, num_layers=1, bad_q_shape=None):
    tensors = {}
    for layer_idx in range(num_layers):
        names = llama_layer_tensor_names(layer_idx)
        tensors[names.q_proj] = {"shape": list(bad_q_shape or (config.q_output_dim(), config.hidden_size)), "dtype": "float16"}
        tensors[names.k_proj] = {"shape": [config.kv_output_dim(), config.hidden_size], "dtype": "float16"}
        tensors[names.v_proj] = {"shape": [config.kv_output_dim(), config.hidden_size], "dtype": "float16"}
        tensors[names.o_proj] = {"shape": [config.hidden_size, config.q_output_dim()], "dtype": "float16"}
        tensors[names.gate_proj] = {"shape": [config.intermediate_size, config.hidden_size], "dtype": "float16"}
        tensors[names.up_proj] = {"shape": [config.intermediate_size, config.hidden_size], "dtype": "float16"}
        tensors[names.down_proj] = {"shape": [config.hidden_size, config.intermediate_size], "dtype": "float16"}
        tensors[names.input_layernorm] = {"shape": [config.hidden_size], "dtype": "float16"}
        tensors[names.post_attention_layernorm] = {"shape": [config.hidden_size], "dtype": "float16"}
    return CheckpointManifest.from_dict({"model_type": "llama_like", "tensors": tensors})


def test_llama_layer_tensor_names_expected():
    names = llama_layer_tensor_names(0)
    assert names.q_proj == "model.layers.0.self_attn.q_proj.weight"
    assert names.down_proj == "model.layers.0.mlp.down_proj.weight"


def test_build_llama_name_map_for_two_layers():
    mapping = build_llama_name_map(2)
    assert 0 in mapping and 1 in mapping
    assert mapping[1].v_proj.endswith("layers.1.self_attn.v_proj.weight")


def test_missing_required_tensors_detects_missing_q_proj():
    cfg = tiny_debug_config()
    manifest = _make_manifest(cfg)
    del manifest.tensors["model.layers.0.self_attn.q_proj.weight"]
    missing = missing_required_tensors(manifest, cfg)
    assert "model.layers.0.self_attn.q_proj.weight" in missing


def test_validate_llama_layer_shapes_passes_for_tiny():
    cfg = tiny_debug_config()
    report = validate_llama_layer_shapes(_make_manifest(cfg), cfg, layer_idx=0)
    assert report.ok is True


def test_validate_llama_layer_shapes_catches_wrong_q_shape():
    cfg = tiny_debug_config()
    report = validate_llama_layer_shapes(_make_manifest(cfg, bad_q_shape=(1, 1)), cfg, layer_idx=0)
    assert report.ok is False
    assert report.errors()


def test_validate_llama_checkpoint_shapes_good_manifest():
    cfg = tiny_debug_config()
    report = validate_llama_checkpoint_shapes(_make_manifest(cfg, num_layers=cfg.num_hidden_layers), cfg)
    assert report.ok is True


def test_gqa_manifest_uses_hq_and_hkv_dims():
    cfg = tiny_gqa_debug_config()
    report = validate_llama_layer_shapes(_make_manifest(cfg), cfg, layer_idx=0)
    assert report.ok is True
