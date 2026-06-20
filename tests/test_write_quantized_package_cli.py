from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


def _create_synthetic_package_json(path: Path, num_layers=2):
    package = {
        "format_version": "0.1.0",
        "model_type": "llama_like",
        "config": {
            "hidden_size": 64,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "intermediate_size": 128,
        },
        "quantization": {"bits": 4, "group_size": 32, "symmetric": True},
        "layers": [
            {
                "layer_idx": i,
                "tensors": {
                    "input_layernorm": {
                        "name": f"layers.{i}.input_layernorm",
                        "role": "norm",
                        "bits": 0, "group_size": 0,
                        "original_shape": [64], "packed_shape": [64],
                        "scales_shape": [0],
                    },
                    "post_attention_layernorm": {
                        "name": f"layers.{i}.post_attention_layernorm",
                        "role": "norm",
                        "bits": 0, "group_size": 0,
                        "original_shape": [64], "packed_shape": [64],
                        "scales_shape": [0],
                    },
                    "qkv": {
                        "name": f"layers.{i}.qkv",
                        "role": "qkv", "bits": 4, "group_size": 32,
                        "original_shape": [128, 64], "packed_shape": [128, 32],
                        "scales_shape": [128, 2],
                    },
                    "o_proj": {
                        "name": f"layers.{i}.o_proj",
                        "role": "o_proj", "bits": 4, "group_size": 32,
                        "original_shape": [64, 128], "packed_shape": [64, 64],
                        "scales_shape": [64, 4],
                    },
                    "gate_proj": {
                        "name": f"layers.{i}.gate_proj",
                        "role": "gate_proj", "bits": 4, "group_size": 32,
                        "original_shape": [128, 64], "packed_shape": [128, 32],
                        "scales_shape": [128, 2],
                    },
                    "up_proj": {
                        "name": f"layers.{i}.up_proj",
                        "role": "up_proj", "bits": 4, "group_size": 32,
                        "original_shape": [128, 64], "packed_shape": [128, 32],
                        "scales_shape": [128, 2],
                    },
                    "down_proj": {
                        "name": f"layers.{i}.down_proj",
                        "role": "down_proj", "bits": 4, "group_size": 32,
                        "original_shape": [64, 128], "packed_shape": [64, 64],
                        "scales_shape": [64, 4],
                    },
                },
            }
            for i in range(num_layers)
        ],
        "global_tensors": {},
        "metadata": {},
    }
    path.write_text(json.dumps(package, indent=2), encoding="utf-8")


class TestWriteQuantizedPackageCLI:
    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            pkg_path = Path(d) / "package.json"
            _create_synthetic_package_json(pkg_path, num_layers=2)
            from scripts.write_quantized_package import main

            exit_code = main([str(pkg_path), "--dry-run"])
            assert exit_code == 0

    def test_synthetic_write(self):
        with tempfile.TemporaryDirectory() as d:
            pkg_path = Path(d) / "package.json"
            _create_synthetic_package_json(pkg_path, num_layers=2)
            from scripts.write_quantized_package import main

            exit_code = main([str(pkg_path), "--synthetic", "--seed", "42"])
            assert exit_code == 0
            tensor_dir = pkg_path.parent / "tensors"
            assert tensor_dir.is_dir()
            npy_files = list(tensor_dir.glob("*.npy"))
            assert len(npy_files) > 0

    def test_synthetic_write_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            pkg_path = Path(d) / "package.json"
            _create_synthetic_package_json(pkg_path, num_layers=1)
            out_dir = Path(d) / "output"
            from scripts.write_quantized_package import main

            exit_code = main(
                [str(pkg_path), "--synthetic", "--output-dir", str(out_dir)]
            )
            assert exit_code == 0
            assert (out_dir / "package.json").exists()
            assert (out_dir / "tensors").is_dir()

    def test_package_not_found(self):
        from scripts.write_quantized_package import main

        exit_code = main(["/nonexistent/package.json", "--dry-run"])
        assert exit_code == 1

    def test_invalid_package_json(self):
        with tempfile.TemporaryDirectory() as d:
            pkg_path = Path(d) / "bad.json"
            pkg_path.write_text("not json", encoding="utf-8")
            from scripts.write_quantized_package import main

            exit_code = main([str(pkg_path), "--dry-run"])
            assert exit_code == 1
