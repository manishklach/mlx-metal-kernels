from __future__ import annotations

from copy import deepcopy


BACKEND_REGISTRY = {
    "fast_attention": {
        "function": "ops.attention_ops.fast_attention",
        "reference_backend": "reference",
        "candidate_backends": [
            "baseline",
            "row_parallel",
            "tiled_kv",
            "threadgroup",
            "baseline_d64",
            "baseline_d128",
            "simdgroup_d64",
        ],
    },
    "decode_attention": {
        "function": "ops.decode_ops.decode_attention",
        "reference_backend": "reference",
        "candidate_backends": [
            "metal",
            "metal_threadgroup",
            "metal_d64",
            "metal_d128",
            "simdgroup_d64",
        ],
    },
    "paged_decode_attention": {
        "function": "ops.paged_kv_ops.paged_decode_attention",
        "reference_backend": "reference",
        "candidate_backends": [
            "metal",
            "metal_threadgroup",
            "metal_d64",
            "metal_d128",
        ],
    },
    "q4_matvec_decode": {
        "function": "ops.quant_ops.q4_matvec_decode",
        "reference_backend": "reference",
        "candidate_backends": [
            "metal",
            "metal_parallel",
            "metal_tiled",
        ],
    },
    "q8_matvec_decode": {
        "function": "ops.quant_ops.q8_matvec_decode",
        "reference_backend": "reference",
        "candidate_backends": [
            "metal",
            "metal_parallel",
            "metal_tiled",
        ],
    },
}


_EXPERIMENTAL_BACKENDS = {
    "row_parallel",
    "tiled_kv",
    "threadgroup",
    "simdgroup_d64",
    "metal_threadgroup",
    "metal_tiled",
}


def list_ops() -> list[str]:
    return sorted(BACKEND_REGISTRY)


def get_candidate_backends(op_name: str) -> list[str]:
    if op_name not in BACKEND_REGISTRY:
        raise KeyError(f"Unknown op: {op_name}")
    return list(BACKEND_REGISTRY[op_name]["candidate_backends"])


def get_reference_backend(op_name: str) -> str:
    if op_name not in BACKEND_REGISTRY:
        raise KeyError(f"Unknown op: {op_name}")
    return str(BACKEND_REGISTRY[op_name]["reference_backend"])


def get_registry_entry(op_name: str) -> dict:
    if op_name not in BACKEND_REGISTRY:
        raise KeyError(f"Unknown op: {op_name}")
    return deepcopy(BACKEND_REGISTRY[op_name])


def validate_backend(op_name: str, backend: str) -> str:
    candidates = set(get_candidate_backends(op_name))
    candidates.add(get_reference_backend(op_name))
    if backend not in candidates:
        raise ValueError(
            f"Unsupported backend={backend!r} for op={op_name!r}. "
            f"Expected one of {sorted(candidates)}."
        )
    return backend


def is_experimental_backend(backend: str) -> bool:
    return backend in _EXPERIMENTAL_BACKENDS


def _dtype_name(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    name = getattr(dtype, "__name__", None)
    if name:
        return name
    return str(dtype)


def filter_backends_for_shape(op_name: str, shape: dict, dtype, backends) -> list[str]:
    if op_name not in BACKEND_REGISTRY:
        raise KeyError(f"Unknown op: {op_name}")
    dtype_name = _dtype_name(dtype).lower()
    dim = shape.get("D")
    filtered = []
    for backend in backends:
        validate_backend(op_name, backend)
        if "d64" in backend and dim != 64:
            continue
        if "d128" in backend and dim != 128:
            continue
        if backend == "simdgroup_d64":
            if dim != 64:
                continue
            if dtype_name != "float16":
                continue
        filtered.append(backend)
    return filtered
