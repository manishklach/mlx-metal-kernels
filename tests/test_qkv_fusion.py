import pytest

mx = pytest.importorskip("mlx.core")

from models import create_fused_qkv_manifest
from models.checkpoint_manifest import CheckpointManifest
from models.llama_config import tiny_debug_config, tiny_gqa_debug_config
from models.qkv_fusion import (
    build_fused_qkv_manifest_entries,
    fuse_qkv_shapes,
    fuse_qkv_weights,
    split_fused_qkv_shape,
    split_fused_qkv_weight,
)


def _manifest_for_config(config):
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.self_attn"
        tensors[f"{prefix}.q_proj.weight"] = {"shape": [config.q_output_dim(), config.hidden_size], "dtype": "float16"}
        tensors[f"{prefix}.k_proj.weight"] = {"shape": [config.kv_output_dim(), config.hidden_size], "dtype": "float16"}
        tensors[f"{prefix}.v_proj.weight"] = {"shape": [config.kv_output_dim(), config.hidden_size], "dtype": "float16"}
    return CheckpointManifest.from_dict(
        {
            "model_type": "llama_like",
            "tensors": tensors,
        }
    )


def test_fuse_qkv_shapes_for_mha():
    cfg = tiny_debug_config()
    assert fuse_qkv_shapes((cfg.hidden_size, cfg.hidden_size), (cfg.hidden_size, cfg.hidden_size), (cfg.hidden_size, cfg.hidden_size)) == (
        3 * cfg.hidden_size,
        cfg.hidden_size,
    )


def test_fuse_qkv_shapes_for_gqa():
    cfg = tiny_gqa_debug_config()
    fused = fuse_qkv_shapes(
        (cfg.q_output_dim(), cfg.hidden_size),
        (cfg.kv_output_dim(), cfg.hidden_size),
        (cfg.kv_output_dim(), cfg.hidden_size),
    )
    assert fused == (cfg.fused_qkv_output_dim(), cfg.hidden_size)


def test_split_fused_qkv_shape():
    cfg = tiny_gqa_debug_config()
    q_shape, k_shape, v_shape = split_fused_qkv_shape((cfg.fused_qkv_output_dim(), cfg.hidden_size), cfg)
    assert q_shape == (cfg.q_output_dim(), cfg.hidden_size)
    assert k_shape == (cfg.kv_output_dim(), cfg.hidden_size)
    assert v_shape == (cfg.kv_output_dim(), cfg.hidden_size)


def test_fuse_and_split_weights_roundtrip():
    cfg = tiny_gqa_debug_config()
    q_w = mx.random.normal((cfg.q_output_dim(), cfg.hidden_size)).astype(mx.float16)
    k_w = mx.random.normal((cfg.kv_output_dim(), cfg.hidden_size)).astype(mx.float16)
    v_w = mx.random.normal((cfg.kv_output_dim(), cfg.hidden_size)).astype(mx.float16)
    fused = fuse_qkv_weights(q_w, k_w, v_w)
    q_split, k_split, v_split = split_fused_qkv_weight(fused, cfg)
    mx.eval(fused, q_split, k_split, v_split)
    assert q_split.shape == q_w.shape
    assert k_split.shape == k_w.shape
    assert v_split.shape == v_w.shape


def test_mismatched_input_dim_raises():
    with pytest.raises(ValueError, match="input dims must match"):
        fuse_qkv_shapes((4, 8), (4, 7), (4, 8))


def test_fused_manifest_entries_and_create_manifest():
    cfg = tiny_gqa_debug_config()
    manifest = _manifest_for_config(cfg)
    fused = build_fused_qkv_manifest_entries(manifest, cfg, layer_idx=0)
    assert fused.shape == (cfg.fused_qkv_output_dim(), cfg.hidden_size)
    derived = create_fused_qkv_manifest(manifest, cfg)
    assert derived.has("model.layers.0.self_attn.qkv_proj.fused_weight")
