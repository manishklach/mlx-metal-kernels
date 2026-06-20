from __future__ import annotations

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - compatibility is only needed when MLX exists
    mx = None


def apply_mlx_compat() -> None:
    if mx is None:
        return
    _patch_random_shape_tuple("normal")
    _patch_random_shape_tuple("uniform")


def _patch_random_shape_tuple(name: str) -> None:
    fn = getattr(mx.random, name, None)
    if fn is None or getattr(fn, "_mlx_shape_tuple_compat", False):
        return

    def wrapper(*args, **kwargs):
        if "shape" not in kwargs and len(args) == 1 and isinstance(args[0], tuple):
            return fn(shape=args[0], **kwargs)
        return fn(*args, **kwargs)

    wrapper._mlx_shape_tuple_compat = True  # type: ignore[attr-defined]
    setattr(mx.random, name, wrapper)


apply_mlx_compat()
