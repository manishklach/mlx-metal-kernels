from __future__ import annotations

from pathlib import Path

import mlx.core as mx

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"


def make_metal_header(dtype: mx.Dtype, **extra_defines: int | str) -> str:
    """Generate Metal kernel header with ELEM_TYPE and optional #define lines.

    Parameters
    ----------
    dtype:
        MLX dtype; must be float16 or bfloat16.
    **extra_defines:
        Additional defines as keyword arguments, e.g.
        ``MAX_HEAD_DIM=128`` produces ``#define MAX_HEAD_DIM 128``.
        Values can be int, float, or string.
    """
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"Metal kernels support only float16/bfloat16, got {dtype}")
    lines = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        f"#define ELEM_TYPE {elem_type}",
    ]
    for name, val in extra_defines.items():
        lines.append(f"#define {name} {val}")
    return "\n".join(lines) + "\n"


def load_metal_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing Metal kernel source: {path}")
    return path.read_text()
