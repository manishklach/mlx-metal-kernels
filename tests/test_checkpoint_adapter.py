import pytest

from models import (
    CheckpointAdapter,
    CheckpointAdapterConfig,
    CheckpointManifest,
    ManifestTensorStore,
    tiny_debug_config,
    tiny_gqa_debug_config,
)

np = pytest.importorskip("numpy")


def _layer_tensors(config, layer_idx):
    stem = f"model.layers.{layer_idx}"
    return {
        f"{stem}.self_attn.q_proj.weight": np.zeros((config.q_output_dim(), config.hidden_size), dtype=np.float16),
        f"{stem}.self_attn.k_proj.weight": np.zeros((config.kv_output_dim(), config.hidden_size), dtype=np.float16),
        f"{stem}.self_attn.v_proj.weight": np.zeros((config.kv_output_dim(), config.hidden_size), dtype=np.float16),
        f"{stem}.self_attn.o_proj.weight": np.zeros((config.hidden_size, config.q_output_dim()), dtype=np.float16),
        f"{stem}.mlp.gate_proj.weight": np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float16),
        f"{stem}.mlp.up_proj.weight": np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float16),
        f"{stem}.mlp.down_proj.weight": np.zeros((config.hidden_size, config.intermediate_size), dtype=np.float16),
        f"{stem}.input_layernorm.weight": np.zeros((config.hidden_size,), dtype=np.float16),
        f"{stem}.post_attention_layernorm.weight": np.zeros((config.hidden_size,), dtype=np.float16),
    }


def _all_tensors(config):
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        tensors.update(_layer_tensors(config, layer_idx))
    return tensors


def test_checkpoint_adapter_describe_contains_expected_fields():
    cfg = tiny_debug_config()
    adapter = CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg)))
    desc = adapter.describe()
    assert desc["hidden_size"] == cfg.hidden_size
    assert desc["layers"] == cfg.num_hidden_layers
    assert desc["heads"] == cfg.num_attention_heads
    assert desc["kv_heads"] == cfg.num_key_value_heads
    assert desc["fuse_qkv"] is True


def test_checkpoint_adapter_validate_ok_for_complete_tensors():
    cfg = tiny_debug_config()
    adapter = CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg)))
    report = adapter.validate()
    assert report.ok is True
    assert report.layer_count == cfg.num_hidden_layers


def test_checkpoint_adapter_validate_catches_wrong_q_proj_shape():
    cfg = tiny_debug_config()
    tensors = _all_tensors(cfg)
    tensors["model.layers.0.self_attn.q_proj.weight"] = np.zeros((1, 1), dtype=np.float16)
    adapter = CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(tensors))
    report = adapter.validate()
    assert report.ok is False
    assert report.errors()


def test_checkpoint_adapter_layer_names_and_shapes():
    cfg = tiny_debug_config()
    adapter = CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg)))
    names = adapter.layer_names(0)
    shapes = adapter.expected_layer_shapes(0)
    assert names.q_proj.endswith("q_proj.weight")
    assert shapes["q_proj"] == (cfg.q_output_dim(), cfg.hidden_size)
    assert shapes["k_proj"] == (cfg.kv_output_dim(), cfg.hidden_size)


def test_checkpoint_adapter_get_fused_qkv_shape_for_mha_and_gqa():
    mha = CheckpointAdapter(tiny_debug_config(), __import__("models").InMemoryTensorStore(_all_tensors(tiny_debug_config())))
    gqa_cfg = tiny_gqa_debug_config()
    gqa = CheckpointAdapter(gqa_cfg, __import__("models").InMemoryTensorStore(_all_tensors(gqa_cfg)))
    assert mha.get_fused_qkv_shape(0) == (tiny_debug_config().fused_qkv_output_dim(), tiny_debug_config().hidden_size)
    assert gqa.get_fused_qkv_shape(0) == (gqa_cfg.fused_qkv_output_dim(), gqa_cfg.hidden_size)


def test_checkpoint_adapter_fuse_qkv_for_layer_returns_expected_shape():
    cfg = tiny_debug_config()
    adapter = CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg)))
    fused = adapter.fuse_qkv_for_layer(0)
    assert tuple(fused.shape) == adapter.get_fused_qkv_shape(0)


def test_checkpoint_adapter_manifest_only_fusion_raises():
    cfg = tiny_debug_config()
    manifest = CheckpointManifest.from_dict(
        {"model_type": "llama_like", "tensors": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in _all_tensors(cfg).items()}}
    )
    adapter = CheckpointAdapter(cfg, ManifestTensorStore(manifest))
    with pytest.raises(NotImplementedError, match="shape-only"):
        adapter.fuse_qkv_for_layer(0)


def test_checkpoint_adapter_quantized_specs_for_layer():
    cfg = tiny_gqa_debug_config()
    adapter = CheckpointAdapter(
        cfg,
        __import__("models").InMemoryTensorStore(_all_tensors(cfg)),
        adapter_config=CheckpointAdapterConfig(quantized=True, bits=4, group_size=32),
    )
    specs = adapter.quantized_specs_for_layer(0)
    assert set(specs) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_checkpoint_adapter_missing_tensor_produces_error():
    cfg = tiny_debug_config()
    tensors = _all_tensors(cfg)
    del tensors["model.layers.0.self_attn.k_proj.weight"]
    adapter = CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(tensors))
    report = adapter.validate()
    assert report.ok is False
    assert any(issue.tensor == "model.layers.0.self_attn.k_proj.weight" for issue in report.errors())
