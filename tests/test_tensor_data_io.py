from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from models.tensor_data_io import (
    TensorDataInfo,
    compute_file_checksum,
    load_tensor_npy,
    save_tensor_npy,
    tensor_dtype,
    tensor_nbytes,
    tensor_shape,
    validate_tensor_file,
)


class TestComputeFileChecksum:
    def test_sha256_of_small_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"hello world")
            tmp = f.name
        try:
            h = compute_file_checksum(tmp)
            assert isinstance(h, str)
            assert len(h) == 64
            assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_sha1(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"hello")
            tmp = f.name
        try:
            h = compute_file_checksum(tmp, algorithm="sha1")
            assert len(h) == 40
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_md5(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"hello")
            tmp = f.name
        try:
            h = compute_file_checksum(tmp, algorithm="md5")
            assert len(h) == 32
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp = f.name
        try:
            h = compute_file_checksum(tmp)
            assert len(h) == 64
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestSaveTensorNpy:
    def test_save_and_load_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.npy"
            info = save_tensor_npy(arr, path)
            assert isinstance(info, TensorDataInfo)
            assert info.shape == (2, 2)
            assert info.dtype == "float32"
            assert info.nbytes > 0
            assert len(info.checksum) == 64
            loaded = load_tensor_npy(path)
            np.testing.assert_array_equal(loaded, arr)

    def test_save_integer_tensor(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "int.npy"
            info = save_tensor_npy(arr, path)
            assert info.dtype == "int32"
            assert info.shape == (3,)

    def test_save_creates_parent_dirs(self):
        arr = np.array([1.0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "nested" / "tensor.npy"
            info = save_tensor_npy(arr, path)
            assert Path(info.file_path).exists()

    def test_save_float16(self):
        arr = np.array([1.0, 2.0], dtype=np.float16)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f16.npy"
            info = save_tensor_npy(arr, path)
            assert info.dtype == "float16"


class TestLoadTensorNpy:
    def test_load_from_file(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "arr.npy"
            save_tensor_npy(arr, path)
            loaded = load_tensor_npy(path)
            np.testing.assert_array_equal(loaded, arr)

    def test_load_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_tensor_npy("/nonexistent/path.npy")


class TestTensorMetadata:
    def test_tensor_shape(self):
        arr = np.zeros((3, 4, 5))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "shape.npy"
            save_tensor_npy(arr, path)
            assert tensor_shape(path) == (3, 4, 5)

    def test_tensor_dtype(self):
        arr = np.zeros((2, 2), dtype=np.float64)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dtype.npy"
            save_tensor_npy(arr, path)
            assert tensor_dtype(path) == "float64"

    def test_tensor_nbytes(self):
        arr = np.zeros((10, 10), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nbytes.npy"
            save_tensor_npy(arr, path)
            assert tensor_nbytes(path) == 400


class TestValidateTensorFile:
    def test_validate_ok(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "valid.npy"
            save_tensor_npy(arr, path)
            issues = validate_tensor_file(
                path,
                expected_shape=(2, 3),
                expected_dtype="float32",
            )
            assert issues == []

    def test_validate_shape_mismatch(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "shape_mismatch.npy"
            save_tensor_npy(arr, path)
            issues = validate_tensor_file(path, expected_shape=(4, 5))
            assert any("shape mismatch" in i for i in issues)

    def test_validate_dtype_mismatch(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dtype_mismatch.npy"
            save_tensor_npy(arr, path)
            issues = validate_tensor_file(path, expected_dtype="int32")
            assert any("dtype mismatch" in i for i in issues)

    def test_validate_checksum(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checksum.npy"
            save_tensor_npy(arr, path)
            wrong = "0" * 64
            issues = validate_tensor_file(path, expected_checksum=wrong)
            assert any("checksum mismatch" in i for i in issues)

    def test_validate_file_not_found(self):
        issues = validate_tensor_file("/nonexistent/file.npy")
        assert any("not found" in i for i in issues)

    def test_validate_without_expected_params(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "no_check.npy"
            save_tensor_npy(arr, path)
            issues = validate_tensor_file(path)
            assert issues == []
