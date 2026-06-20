import json

import pytest

from models.checkpoint_manifest import CheckpointManifest, TensorInfo


def test_tensor_info_roundtrip():
    info = TensorInfo(name="foo", shape=(2, 3), dtype="float16", source="mock", nbytes=12)
    restored = TensorInfo.from_dict(info.to_dict())
    assert restored == info


def test_manifest_from_dict_tensor_dict():
    manifest = CheckpointManifest.from_dict(
        {
            "model_type": "llama_like",
            "tensors": {
                "foo": {"shape": [2, 3], "dtype": "float16"},
            },
        }
    )
    assert manifest.has("foo")


def test_manifest_from_dict_tensor_list():
    manifest = CheckpointManifest.from_dict(
        {
            "model_type": "llama_like",
            "tensors": [
                {"name": "foo", "shape": [2, 3], "dtype": "float16"},
                {"name": "bar", "shape": [3], "dtype": "float16"},
            ],
        }
    )
    assert manifest.tensor_names() == ["bar", "foo"]


def test_manifest_find_and_require(tmp_path):
    manifest = CheckpointManifest.from_dict(
        {
            "model_type": "llama_like",
            "tensors": {
                "model.layers.0.self_attn.q_proj.weight": {"shape": [2, 3], "dtype": "float16"},
                "model.layers.0.self_attn.k_proj.weight": {"shape": [2, 3], "dtype": "float16"},
            },
        }
    )
    assert len(manifest.find(prefix="model.layers.0")) == 2
    assert manifest.require("model.layers.0.self_attn.q_proj.weight").name.endswith("q_proj.weight")
    with pytest.raises(KeyError):
        manifest.require("missing")
    path = tmp_path / "manifest.json"
    manifest.save_json(path)
    loaded = CheckpointManifest.load_json(path)
    assert loaded.to_dict() == manifest.to_dict()


def test_invalid_shape_and_dtype_raise():
    with pytest.raises(ValueError, match="positive integers"):
        TensorInfo(name="foo", shape=(2, 0), dtype="float16")
    with pytest.raises(ValueError, match="non-empty string"):
        TensorInfo(name="foo", shape=(2, 3), dtype="")
