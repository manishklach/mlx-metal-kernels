from __future__ import annotations

import pytest

from models.checkpoint_converter import CheckpointConverter, CheckpointConverterConfig

np = pytest.importorskip("numpy")


def _build_adapter(config, fuse_qkv=True, seed=42, num_layers=None):
    from models.checkpoint_adapter import CheckpointAdapter, CheckpointAdapterConfig
    from models.tensor_store import InMemoryTensorStore

    nlayers = num_layers or config.num_hidden_layers
    rng = np.random.default_rng(seed)
    tensors = {}
    for li in range(nlayers):
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
    return CheckpointAdapter(config, InMemoryTensorStore(tensors), adapter_config=CheckpointAdapterConfig(fuse_qkv=fuse_qkv))


class TestCheckpointConverter:
    def test_convert_single_layer(self):
        from models.llama_config import tiny_debug_config

        config = tiny_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        package, report = converter.convert()
        assert report.ok
        assert package is not None
        assert package.num_layers() == config.num_hidden_layers
        assert package.tensor_count() == config.num_hidden_layers * 7

    def test_convert_single_layer_gqa(self):
        from models.llama_config import tiny_gqa_debug_config

        config = tiny_gqa_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        package, report = converter.convert()
        assert report.ok
        assert package is not None
        assert package.num_layers() == config.num_hidden_layers
        qkv_meta = package.layers[0].tensors["qkv"]
        expected_rows = config.q_output_dim() + 2 * config.kv_output_dim()
        assert qkv_meta.original_shape[0] == expected_rows

    def test_convert_with_layers_param(self):
        from models.llama_config import tiny_debug_config

        config = tiny_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        package, report = converter.convert()
        assert report.ok
        assert package is not None
        assert package.num_layers() == 2

    def test_convert_with_explicit_layer_indices(self):
        from models.llama_config import tiny_debug_config

        config = tiny_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        packages = converter.convert_layers([0])
        assert len(packages) == 1
        assert packages[0].layer_idx == 0

    def test_convert_multiple_layers(self):
        from models.llama_config import tiny_gqa_debug_config

        config = tiny_gqa_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        packages = converter.convert_layers([0, 1])
        assert len(packages) == 2

    def test_package_config_includes_head_counts(self):
        from models.llama_config import tiny_gqa_debug_config

        config = tiny_gqa_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        package, _ = converter.convert()
        assert package is not None
        assert package.config["num_attention_heads"] == config.num_attention_heads
        assert package.config["num_key_value_heads"] == config.num_key_value_heads

    def test_invalid_bits_raises(self):
        from models.llama_config import tiny_debug_config

        config = tiny_debug_config()
        adapter = _build_adapter(config)
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            CheckpointConverter(adapter, CheckpointConverterConfig(bits=3))

    def test_save_tensor_data_writes_tensors(self, tmp_path):
        from models.llama_config import tiny_debug_config

        config = tiny_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(
            adapter, CheckpointConverterConfig(bits=4, group_size=32, save_tensor_data=True)
        )
        out_dir = tmp_path / "pkg"
        out_dir.mkdir()
        package, report = converter.convert(output_path=str(out_dir))
        assert report.ok
        assert package is not None
        assert package.has_tensor_data(require_all=True)
        json_path = out_dir / "package.json"
        assert json_path.exists()
        tensor_dir = out_dir / "tensors"
        assert tensor_dir.is_dir()
        npy_files = list(tensor_dir.glob("*.npy"))
        assert len(npy_files) > 0

    def test_layer_out_of_range_raises(self):
        from models.llama_config import tiny_debug_config

        config = tiny_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4))
        with pytest.raises(ValueError, match="out of range"):
            converter.convert_layers([99])

    def test_save_package_to_path(self, tmp_path):
        from models.llama_config import tiny_gqa_debug_config

        config = tiny_gqa_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        package, _ = converter.convert()
        assert package is not None
        out = tmp_path / "test_pkg.json"
        converter.save_package(package, str(out))
        assert out.exists()

    def test_convert_with_output_path(self, tmp_path):
        from models.llama_config import tiny_gqa_debug_config

        config = tiny_gqa_debug_config()
        adapter = _build_adapter(config)
        converter = CheckpointConverter(adapter, CheckpointConverterConfig(bits=4, group_size=32))
        out = tmp_path / "output.json"
        _, report = converter.convert(output_path=str(out))
        assert report.ok
        assert out.exists()


class TestCheckpointConverterConfigValidation:
    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            CheckpointConverterConfig(bits=3).validate()

    def test_negative_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size must be positive"):
            CheckpointConverterConfig(group_size=0).validate()

    def test_save_tensor_data_valid(self):
        config = CheckpointConverterConfig(save_tensor_data=True)
        validated = config.validate()
        assert validated.save_tensor_data is True
