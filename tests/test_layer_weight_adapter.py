import pytest

from models import CheckpointAdapter, CheckpointManifest, LayerWeightAdapter, ManifestTensorStore, tiny_debug_config

np = pytest.importorskip("numpy")


def _all_tensors(config):
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        stem = f"model.layers.{layer_idx}"
        tensors.update(
            {
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
        )
    return tensors


def test_layer_weight_adapter_required_names_and_shape_summary():
    cfg = tiny_debug_config()
    adapter = LayerWeightAdapter(CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg))))
    required = adapter.required_tensor_names(0)
    summary = adapter.layer_shape_summary(0)
    assert set(required) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "input_layernorm", "post_attention_layernorm"}
    assert summary["qkv_fused"] == (cfg.fused_qkv_output_dim(), cfg.hidden_size)


def test_layer_weight_adapter_load_layer_with_fused_qkv():
    cfg = tiny_debug_config()
    adapter = LayerWeightAdapter(CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg))))
    layer = adapter.load_layer(0, fuse_qkv=True, load_tensors=True)
    assert layer.has_fused_qkv() is True
    assert "qkv_fused" in layer.available_names()
    assert layer.shapes()["qkv_fused"] == (cfg.fused_qkv_output_dim(), cfg.hidden_size)


def test_layer_weight_adapter_manifest_only_load_raises():
    cfg = tiny_debug_config()
    manifest = CheckpointManifest.from_dict(
        {"model_type": "llama_like", "tensors": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in _all_tensors(cfg).items()}}
    )
    adapter = LayerWeightAdapter(CheckpointAdapter(cfg, ManifestTensorStore(manifest)))
    with pytest.raises(NotImplementedError):
        adapter.load_layer(0, load_tensors=True)


def test_layer_weight_adapter_shape_only_mode_works():
    cfg = tiny_debug_config()
    adapter = LayerWeightAdapter(CheckpointAdapter(cfg, __import__("models").InMemoryTensorStore(_all_tensors(cfg))))
    layer = adapter.load_layer(0, load_tensors=False)
    assert layer.layer_idx == 0
    assert layer.available_names() == []
