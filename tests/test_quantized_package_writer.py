from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from models.quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage
from models.quantized_package_io import QuantizedCheckpointPackage, QuantizedLayerMetadata, QuantizedTensorMetadata
from models.quantized_package_writer import PackageWriterConfig, PackageWriterReport, QuantizedPackageWriter


def _sample_linear(name, bits=4, group_size=32, out_dim=64, in_dim=128):
    packed_cols = in_dim // 2 if bits == 4 else in_dim
    groups = (in_dim + group_size - 1) // group_size
    return QuantizedLinearPackage(
        name=name,
        weight=np.random.default_rng(0).normal(size=(out_dim, packed_cols)).astype(np.float16),
        scales=np.random.default_rng(1).normal(size=(out_dim, groups)).astype(np.float16),
        zeros=None,
        bits=bits,
        group_size=group_size,
        original_shape=(out_dim, in_dim),
    )


def _sample_layer_package(layer_idx=0):
    return QuantizedLlamaLayerPackage(
        layer_idx=layer_idx,
        input_layernorm_weight=np.ones((64,), dtype=np.float16),
        post_attention_layernorm_weight=np.ones((64,), dtype=np.float16),
        qkv=_sample_linear(f"layers.{layer_idx}.qkv", out_dim=128, in_dim=64),
        o_proj=_sample_linear(f"layers.{layer_idx}.o_proj", out_dim=64, in_dim=128),
        gate_proj=_sample_linear(f"layers.{layer_idx}.gate_proj", out_dim=128, in_dim=64),
        up_proj=_sample_linear(f"layers.{layer_idx}.up_proj", out_dim=128, in_dim=64),
        down_proj=_sample_linear(f"layers.{layer_idx}.down_proj", out_dim=64, in_dim=128),
    )


def _sample_tensor_meta(name, role="qkv", bits=4, group_size=32, out_dim=128, in_dim=64):
    packed_cols = in_dim // 2 if bits == 4 else in_dim
    groups = (in_dim + group_size - 1) // group_size
    return QuantizedTensorMetadata(
        name=name,
        role=role,
        bits=bits,
        group_size=group_size,
        original_shape=(out_dim, in_dim),
        packed_shape=(out_dim, packed_cols),
        scales_shape=(out_dim, groups),
    )


def _sample_layer_meta(layer_idx=0):
    return QuantizedLayerMetadata(
        layer_idx=layer_idx,
        tensors={
            "input_layernorm": QuantizedTensorMetadata(
                name=f"layers.{layer_idx}.input_layernorm",
                role="norm", bits=0, group_size=0,
                original_shape=(64,), packed_shape=(64,), scales_shape=(0,),
            ),
            "post_attention_layernorm": QuantizedTensorMetadata(
                name=f"layers.{layer_idx}.post_attention_layernorm",
                role="norm", bits=0, group_size=0,
                original_shape=(64,), packed_shape=(64,), scales_shape=(0,),
            ),
            "qkv": _sample_tensor_meta(f"layers.{layer_idx}.qkv", "qkv", out_dim=128, in_dim=64),
            "o_proj": _sample_tensor_meta(f"layers.{layer_idx}.o_proj", "o_proj", out_dim=64, in_dim=128),
            "gate_proj": _sample_tensor_meta(f"layers.{layer_idx}.gate_proj", "gate_proj", out_dim=128, in_dim=64),
            "up_proj": _sample_tensor_meta(f"layers.{layer_idx}.up_proj", "up_proj", out_dim=128, in_dim=64),
            "down_proj": _sample_tensor_meta(f"layers.{layer_idx}.down_proj", "down_proj", out_dim=64, in_dim=128),
        },
    )


class TestPackageWriterConfig:
    def test_default_validation(self):
        config = PackageWriterConfig().validate()
        assert config.tensor_subdir == "tensors"
        assert config.checksum_algorithm == "sha256"

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="checksum_algorithm"):
            PackageWriterConfig(checksum_algorithm="invalid").validate()

    def test_empty_subdir_raises(self):
        with pytest.raises(ValueError, match="tensor_subdir"):
            PackageWriterConfig(tensor_subdir="").validate()


class TestQuantizedPackageWriter:
    def test_write_tensors_two_layers(self):
        layer_packages = [_sample_layer_package(0), _sample_layer_package(1)]
        meta_layers = [_sample_layer_meta(0), _sample_layer_meta(1)]
        package = QuantizedCheckpointPackage(
            format_version="0.1.0",
            model_type="llama_like",
            config={"hidden_size": 64, "num_hidden_layers": 2},
            quantization={"bits": 4, "group_size": 32, "symmetric": True},
            layers=meta_layers,
        )
        writer = QuantizedPackageWriter()
        with tempfile.TemporaryDirectory() as d:
            report = writer.write_tensors(package, layer_packages, d)
            assert report.ok
            assert report.tensor_count > 0
            assert len(report.files_written) > 0
            assert report.total_bytes > 0
            assert report.package_path is not None
            assert Path(report.package_path).exists()
            for layer_meta in package.layers:
                for key, tensor_meta in layer_meta.tensors.items():
                    assert tensor_meta.data_file is not None, f"{key}.data_file"
                    if tensor_meta.bits != 0:
                        assert tensor_meta.scales_file is not None, f"{key}.scales_file"
                    assert tensor_meta.checksum is not None, f"{key}.checksum"

    def test_write_tensors_includes_checksums(self):
        layer_packages = [_sample_layer_package(0)]
        meta_layers = [_sample_layer_meta(0)]
        package = QuantizedCheckpointPackage(
            format_version="0.1.0",
            model_type="llama_like",
            config={"hidden_size": 64, "num_hidden_layers": 1},
            quantization={"bits": 4, "group_size": 32, "symmetric": True},
            layers=meta_layers,
        )
        writer = QuantizedPackageWriter()
        with tempfile.TemporaryDirectory() as d:
            writer.write_tensors(package, layer_packages, d)
            for layer_meta in package.layers:
                for tensor_meta in layer_meta.tensors.values():
                    assert tensor_meta.checksum is not None
                    assert len(tensor_meta.checksum) == 64

    def test_write_tensors_creates_tensor_subdir(self):
        layer_packages = [_sample_layer_package(0)]
        meta_layers = [_sample_layer_meta(0)]
        package = QuantizedCheckpointPackage(
            format_version="0.1.0",
            model_type="llama_like",
            config={"hidden_size": 64, "num_hidden_layers": 1},
            quantization={"bits": 4, "group_size": 32, "symmetric": True},
            layers=meta_layers,
        )
        writer = QuantizedPackageWriter(PackageWriterConfig(tensor_subdir="mytensors"))
        with tempfile.TemporaryDirectory() as d:
            report = writer.write_tensors(package, layer_packages, d)
            assert report.ok
            tensor_dir = Path(d) / "mytensors"
            assert tensor_dir.is_dir()
            assert any(f.endswith(".npy") for f in report.files_written)

    def test_write_tensors_updates_package_json(self):
        layer_packages = [_sample_layer_package(0)]
        meta_layers = [_sample_layer_meta(0)]
        package = QuantizedCheckpointPackage(
            format_version="0.1.0",
            model_type="llama_like",
            config={"hidden_size": 64, "num_hidden_layers": 1},
            quantization={"bits": 4, "group_size": 32, "symmetric": True},
            layers=meta_layers,
        )
        writer = QuantizedPackageWriter()
        with tempfile.TemporaryDirectory() as d:
            report = writer.write_tensors(package, layer_packages, d)
            json_path = Path(report.package_path)
            loaded = QuantizedCheckpointPackage.load_json(json_path)
            assert loaded.has_tensor_data(require_all=True)
            assert loaded.tensor_files(base_dir=d)

    def test_write_tensors_dry_run(self):
        layer_packages = [_sample_layer_package(0)]
        meta_layers = [_sample_layer_meta(0)]
        package = QuantizedCheckpointPackage(
            format_version="0.1.0",
            model_type="llama_like",
            config={"hidden_size": 64, "num_hidden_layers": 1},
            quantization={"bits": 4, "group_size": 32, "symmetric": True},
            layers=meta_layers,
        )
        writer = QuantizedPackageWriter()
        with tempfile.TemporaryDirectory() as d:
            report = writer.write_package(package, layer_packages, d, dry_run=True)
            assert report.ok
            assert report.tensor_count > 0
            assert not Path(d).joinpath("tensors").exists()

    def test_write_tensors_with_global_tensors(self):
        layer_packages = [_sample_layer_package(0)]
        meta_layers = [_sample_layer_meta(0)]
        package = QuantizedCheckpointPackage(
            format_version="0.1.0",
            model_type="llama_like",
            config={"hidden_size": 64, "num_hidden_layers": 1},
            quantization={"bits": 4, "group_size": 32, "symmetric": True},
            layers=meta_layers,
            global_tensors={
                "embedding": QuantizedTensorMetadata(
                    name="embedding", role="embedding", bits=0, group_size=0,
                    original_shape=(64, 64), packed_shape=(64, 64), scales_shape=(0,),
                ),
                "norm": QuantizedTensorMetadata(
                    name="norm", role="norm", bits=0, group_size=0,
                    original_shape=(64,), packed_shape=(64,), scales_shape=(0,),
                ),
            },
        )
        global_tensors_data = {
            "embedding": np.random.default_rng(0).normal(size=(64, 64)).astype(np.float16),
            "norm": np.ones((64,), dtype=np.float16),
        }
        writer = QuantizedPackageWriter()
        with tempfile.TemporaryDirectory() as d:
            report = writer.write_tensors(
                package, layer_packages, d, global_tensors=global_tensors_data
            )
            assert report.ok
            assert report.tensor_count > 0

    def test_report_raise_for_errors(self):
        report = PackageWriterReport(ok=True, files_written=[], errors=[])
        report.raise_for_errors()

        report = PackageWriterReport(ok=False, errors=["something failed"])
        with pytest.raises(ValueError, match="something failed"):
            report.raise_for_errors()
