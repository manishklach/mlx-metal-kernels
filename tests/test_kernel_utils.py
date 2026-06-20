from __future__ import annotations

from pathlib import Path

import pytest

from ops.kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source


def test_kernel_dir_is_absolute():
    assert KERNEL_DIR.is_absolute()
    assert KERNEL_DIR.name == "kernels"


def test_make_metal_header_float16():
    import mlx.core as mx
    header = make_metal_header(mx.float16)
    assert "#define ELEM_TYPE half" in header
    assert "#include <metal_stdlib>" in header


def test_make_metal_header_bfloat16():
    import mlx.core as mx
    header = make_metal_header(mx.bfloat16)
    assert "#define ELEM_TYPE bfloat" in header


def test_make_metal_header_extra_defines():
    import mlx.core as mx
    header = make_metal_header(mx.float16, MAX_HEAD_DIM=128, TG_THREADS=64)
    assert "#define MAX_HEAD_DIM 128" in header
    assert "#define TG_THREADS 64" in header


def test_make_metal_header_raises_on_bad_dtype():
    import mlx.core as mx
    with pytest.raises(TypeError, match="only float16/bfloat16"):
        make_metal_header(mx.float32)


def test_load_metal_source_raises_on_missing():
    nonexistent = KERNEL_DIR / "nonexistent_kernel.metal"
    with pytest.raises(FileNotFoundError, match="Missing Metal kernel source"):
        load_metal_source(nonexistent)


def test_load_metal_source_known_kernel():
    kernel = KERNEL_DIR / "rms_norm.metal"
    source = load_metal_source(kernel)
    assert isinstance(source, str)
    assert len(source) > 0
    assert "kernel" in source
