import pytest

from models import CheckpointManifest
from models.tensor_store import InMemoryTensorStore, ManifestTensorStore, SafeTensorsTensorStore

np = pytest.importorskip("numpy")


def test_in_memory_tensor_store_basic_behavior():
    tensors = {
        "a": np.zeros((2, 3), dtype=np.float16),
        "b": np.ones((4,), dtype=np.float32),
    }
    store = InMemoryTensorStore(tensors)
    assert store.keys() == ["a", "b"]
    assert store.has("a") is True
    assert store.get_shape("a") == (2, 3)
    assert "float16" in store.get_dtype("a")
    assert store.load("b").shape == (4,)
    with pytest.raises(KeyError, match="Missing tensor"):
        store.load("missing")


def test_manifest_tensor_store_shape_only_behavior():
    manifest = CheckpointManifest.from_dict(
        {
            "model_type": "llama_like",
            "tensors": {
                "x": {"shape": [2, 3], "dtype": "float16"},
                "y": {"shape": [4], "dtype": "float32"},
            },
        }
    )
    store = ManifestTensorStore(manifest)
    assert store.keys() == ["x", "y"]
    assert store.get_shape("x") == (2, 3)
    assert store.get_dtype("y") == "float32"
    with pytest.raises(NotImplementedError, match="shape-only"):
        store.load("x")


def test_safetensors_tensor_store_missing_dependency_is_clear(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "safetensors":
            raise ImportError("missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="safetensors is optional"):
        SafeTensorsTensorStore("dummy.safetensors")
