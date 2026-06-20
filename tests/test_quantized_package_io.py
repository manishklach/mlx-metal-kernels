from __future__ import annotations

import json

import pytest

from models.quantized_package_io import (
    QuantizedCheckpointPackage,
    QuantizedLayerMetadata,
    QuantizedTensorMetadata,
    package_from_quantized_layers,
)

np = pytest.importorskip("numpy")


def _sample_tensor_meta(**overrides) -> QuantizedTensorMetadata:
    kwargs = dict(
        name="layers.0.qkv",
        role="qkv",
        bits=4,
        group_size=32,
        original_shape=(128, 64),
        packed_shape=(128, 32),
        scales_shape=(128, 2),
    )
    kwargs.update(overrides)
    return QuantizedTensorMetadata(**kwargs)


def _sample_layer_meta(layer_idx=0) -> QuantizedLayerMetadata:
    return QuantizedLayerMetadata(
        layer_idx=layer_idx,
        tensors={
            "input_layernorm": _sample_tensor_meta(name=f"layers.{layer_idx}.input_layernorm", role="norm", original_shape=(64,), packed_shape=(64,), scales_shape=(0,), bits=0, group_size=0),
            "post_attention_layernorm": _sample_tensor_meta(name=f"layers.{layer_idx}.post_attention_layernorm", role="norm", original_shape=(64,), packed_shape=(64,), scales_shape=(0,), bits=0, group_size=0),
            "qkv": _sample_tensor_meta(name=f"layers.{layer_idx}.qkv", role="qkv", original_shape=(128, 64), packed_shape=(128, 32), scales_shape=(128, 2)),
            "o_proj": _sample_tensor_meta(name=f"layers.{layer_idx}.o_proj", role="o_proj", original_shape=(64, 128), packed_shape=(64, 64), scales_shape=(64, 4)),
            "gate_proj": _sample_tensor_meta(name=f"layers.{layer_idx}.gate_proj", role="gate_proj", original_shape=(128, 64), packed_shape=(128, 32), scales_shape=(128, 2)),
            "up_proj": _sample_tensor_meta(name=f"layers.{layer_idx}.up_proj", role="up_proj", original_shape=(128, 64), packed_shape=(128, 32), scales_shape=(128, 2)),
            "down_proj": _sample_tensor_meta(name=f"layers.{layer_idx}.down_proj", role="down_proj", original_shape=(64, 128), packed_shape=(64, 64), scales_shape=(64, 4)),
        },
    )


def _sample_package(layers=2) -> QuantizedCheckpointPackage:
    return QuantizedCheckpointPackage(
        format_version="0.1.0",
        model_type="llama_like",
        config={"hidden_size": 64, "num_hidden_layers": 2},
        quantization={"bits": 4, "group_size": 32, "symmetric": True},
        layers=[_sample_layer_meta(i) for i in range(layers)],
    )


class TestQuantizedTensorMetadata:
    def test_to_dict_from_dict_roundtrip(self):
        meta = _sample_tensor_meta()
        restored = QuantizedTensorMetadata.from_dict(meta.to_dict())
        assert restored == meta

    def test_to_dict_from_dict_with_zeros(self):
        meta = _sample_tensor_meta(zeros_shape=(128, 2))
        restored = QuantizedTensorMetadata.from_dict(meta.to_dict())
        assert restored.zeros_shape == (128, 2)

    def test_to_dict_from_dict_none_zeros(self):
        meta = _sample_tensor_meta(zeros_shape=None)
        restored = QuantizedTensorMetadata.from_dict(meta.to_dict())
        assert restored.zeros_shape is None

    def test_to_dict_includes_lists_for_shapes(self):
        d = _sample_tensor_meta().to_dict()
        assert isinstance(d["original_shape"], list)
        assert isinstance(d["packed_shape"], list)
        assert isinstance(d["scales_shape"], list)


class TestQuantizedLayerMetadata:
    def test_to_dict_from_dict_roundtrip(self):
        layer = _sample_layer_meta(layer_idx=1)
        restored = QuantizedLayerMetadata.from_dict(layer.to_dict())
        assert restored.layer_idx == 1
        assert set(restored.tensors) == set(layer.tensors)
        for key in layer.tensors:
            assert restored.tensors[key] == layer.tensors[key]

    def test_layer_idx_preserved(self):
        layer = _sample_layer_meta(layer_idx=5)
        assert layer.to_dict()["layer_idx"] == 5


class TestQuantizedCheckpointPackage:
    def test_to_dict_from_dict_roundtrip(self):
        pkg = _sample_package(layers=2)
        restored = QuantizedCheckpointPackage.from_dict(pkg.to_dict())
        assert restored.format_version == pkg.format_version
        assert restored.model_type == pkg.model_type
        assert len(restored.layers) == 2
        assert restored.layers[0].layer_idx == 0
        assert restored.layers[1].layer_idx == 1

    def test_save_load_json_roundtrip(self, tmp_path):
        pkg = _sample_package(layers=2)
        path = tmp_path / "package.json"
        pkg.save_json(str(path))
        assert path.exists()
        loaded = QuantizedCheckpointPackage.load_json(str(path))
        assert loaded.format_version == pkg.format_version
        assert loaded.num_layers() == 2
        assert loaded.tensor_count() > 0

    def test_validate_valid_package(self):
        pkg = _sample_package(layers=2)
        pkg.validate()

    def test_validate_empty_format_version(self):
        pkg = _sample_package()
        pkg.format_version = ""
        with pytest.raises(ValueError, match="format_version"):
            pkg.validate()

    def test_validate_empty_model_type(self):
        pkg = _sample_package()
        pkg.model_type = ""
        with pytest.raises(ValueError, match="model_type"):
            pkg.validate()

    def test_validate_no_layers(self):
        pkg = _sample_package(layers=0)
        with pytest.raises(ValueError, match="at least one layer"):
            pkg.validate()

    def test_validate_no_layers_allow_partial(self):
        pkg = _sample_package(layers=0)
        pkg.validate(allow_partial=True)

    def test_validate_invalid_bits(self):
        pkg = _sample_package()
        pkg.layers[0].tensors["qkv"].bits = 3
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            pkg.validate()

    def test_validate_invalid_group_size(self):
        pkg = _sample_package()
        pkg.layers[0].tensors["qkv"].group_size = -1
        with pytest.raises(ValueError, match="group_size must be positive"):
            pkg.validate()

    def test_validate_missing_required_tensor_key(self):
        pkg = _sample_package()
        del pkg.layers[0].tensors["o_proj"]
        with pytest.raises(ValueError, match="missing required tensor key"):
            pkg.validate()

    def test_validate_non_positive_shape_dim(self):
        pkg = _sample_package()
        pkg.layers[0].tensors["qkv"].original_shape = (128, 0)
        with pytest.raises(ValueError, match="non-positive"):
            pkg.validate()

    def test_validate_duplicate_layer_idx(self):
        pkg = _sample_package(layers=2)
        pkg.layers.append(pkg.layers[0])
        with pytest.raises(ValueError, match="duplicate layer_idx"):
            pkg.validate()

    def test_validate_non_contiguous_indices(self):
        pkg = _sample_package(layers=1)
        pkg.layers.append(_sample_layer_meta(layer_idx=3))
        with pytest.raises(ValueError, match="contiguous"):
            pkg.validate()

    def test_num_layers(self):
        assert _sample_package(layers=3).num_layers() == 3
        assert _sample_package(layers=0).num_layers() == 0

    def test_tensor_count(self):
        pkg = _sample_package(layers=2)
        expected = 2 * 7
        assert pkg.tensor_count() == expected

    def test_summary_returns_useful_dict(self):
        pkg = _sample_package(layers=2)
        s = pkg.summary()
        assert s["num_layers"] == 2
        assert s["tensor_count"] == 14
        assert "per_layer" in s
        assert "0" in s["per_layer"]
        assert "1" in s["per_layer"]


class TestPackageFromQuantizedLayers:
    def _make_quantized_layers(self, config, bits=4, group_size=32):
        from models import CheckpointAdapter, CheckpointQuantizer, InMemoryTensorStore, QuantizationConfig

        rng = np.random.default_rng(42)
        tensors = {}
        for li in range(config.num_hidden_layers):
            stem = f"model.layers.{li}"
            tensors[f"{stem}.self_attn.q_proj.weight"] = rng.normal(size=(config.q_output_dim(), config.hidden_size)).astype(np.float16)
            tensors[f"{stem}.self_attn.k_proj.weight"] = rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16)
            tensors[f"{stem}.self_attn.v_proj.weight"] = rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16)
            tensors[f"{stem}.self_attn.o_proj.weight"] = rng.normal(size=(config.hidden_size, config.q_output_dim())).astype(np.float16)
            tensors[f"{stem}.mlp.gate_proj.weight"] = rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16)
            tensors[f"{stem}.mlp.up_proj.weight"] = rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16)
            tensors[f"{stem}.mlp.down_proj.weight"] = rng.normal(size=(config.hidden_size, config.intermediate_size)).astype(np.float16)
            tensors[f"{stem}.input_layernorm.weight"] = np.ones((config.hidden_size,), dtype=np.float16)
            tensors[f"{stem}.post_attention_layernorm.weight"] = np.ones((config.hidden_size,), dtype=np.float16)
        store = InMemoryTensorStore(tensors)
        from models.checkpoint_adapter import CheckpointAdapterConfig
        adapter = CheckpointAdapter(config, store, adapter_config=CheckpointAdapterConfig(fuse_qkv=True))
        quantizer = CheckpointQuantizer(adapter, QuantizationConfig(bits=bits, group_size=group_size))
        return quantizer.quantize_layers()

    def test_creates_expected_layer_count(self):
        from models.llama_config import tiny_gqa_debug_config
        config = tiny_gqa_debug_config()
        q_layers = self._make_quantized_layers(config)
        pkg = package_from_quantized_layers(config, q_layers, bits=4, group_size=32)
        assert pkg.num_layers() == config.num_hidden_layers
        assert pkg.tensor_count() == config.num_hidden_layers * 7

    def test_gqa_qkv_shape_in_metadata(self):
        from models.llama_config import tiny_gqa_debug_config
        config = tiny_gqa_debug_config()
        q_layers = self._make_quantized_layers(config)
        pkg = package_from_quantized_layers(config, q_layers, bits=4, group_size=32)
        qkv_meta = pkg.layers[0].tensors["qkv"]
        expected_rows = config.q_output_dim() + 2 * config.kv_output_dim()
        assert qkv_meta.original_shape[0] == expected_rows

    def test_package_config_includes_hidden_size(self):
        from models.llama_config import tiny_gqa_debug_config
        config = tiny_gqa_debug_config()
        q_layers = self._make_quantized_layers(config)
        pkg = package_from_quantized_layers(config, q_layers, bits=4, group_size=32)
        assert pkg.config["hidden_size"] == config.hidden_size

    def test_bits_group_size_propagated(self):
        from models.llama_config import tiny_gqa_debug_config
        config = tiny_gqa_debug_config()
        q_layers = self._make_quantized_layers(config, bits=8, group_size=64)
        pkg = package_from_quantized_layers(config, q_layers, bits=8, group_size=64)
        assert pkg.quantization["bits"] == 8
        assert pkg.quantization["group_size"] == 64

    def test_full_roundtrip_save_load_validate(self, tmp_path):
        from models.llama_config import tiny_debug_config
        config = tiny_debug_config()
        q_layers = self._make_quantized_layers(config)
        pkg = package_from_quantized_layers(config, q_layers, bits=4, group_size=32)
        path = tmp_path / "roundtrip.json"
        pkg.save_json(str(path))
        loaded = QuantizedCheckpointPackage.load_json(str(path))
        loaded.validate()
        assert loaded.num_layers() == config.num_hidden_layers
