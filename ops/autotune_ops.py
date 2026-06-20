from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


DEFAULT_AUTOTUNE_PATH = "~/.cache/mlx-metal-kernels/autotune_results.json"

_DEFAULT_BACKENDS = {
    "fast_attention": "baseline",
    "decode_attention": "metal",
    "paged_decode_attention": "metal",
    "q4_matvec_decode": "metal_parallel",
    "q8_matvec_decode": "metal_parallel",
}


def _try_command(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _system_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": sys.version.replace("\n", " "),
        "macos_version": platform.mac_ver()[0] or None,
        "chip_info": _try_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or _try_command(["system_profiler", "SPHardwareDataType"]),
    }


def _normalize_shape(shape: dict | None) -> dict:
    if shape is None:
        return {}
    return {str(key): shape[key] for key in sorted(shape)}


def _normalize_extra(extra: dict | None) -> dict:
    if extra is None:
        return {}
    return {str(key): extra[key] for key in sorted(extra)}


def _normalize_dtype(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    name = getattr(dtype, "__name__", None)
    if name:
        return name
    text = str(dtype)
    if "." in text:
        return text.split(".")[-1]
    return text


def _normalize_path(path=None) -> Path:
    raw = DEFAULT_AUTOTUNE_PATH if path is None else path
    return Path(os.path.expanduser(str(raw)))


def load_autotune_results(path=None) -> dict:
    result_path = _normalize_path(path)
    if not result_path.exists():
        return {"version": 1, "entries": {}}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if "entries" not in payload:
        payload = {"version": 1, "entries": payload}
    payload.setdefault("version", 1)
    payload.setdefault("entries", {})
    return payload


def save_autotune_results(results, path=None) -> Path:
    result_path = _normalize_path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def make_tuning_key(op_name, shape, dtype, extra=None) -> str:
    payload = {
        "op": op_name,
        "shape": _normalize_shape(shape),
        "dtype": _normalize_dtype(dtype),
        "system": _system_info(),
        "extra": _normalize_extra(extra),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def lookup_best_backend(op_name, shape, dtype, path=None, extra=None):
    results = load_autotune_results(path)
    key = make_tuning_key(op_name, shape, dtype, extra=extra)
    entry = results.get("entries", {}).get(key)
    if not entry or entry.get("status") != "ok":
        raise KeyError(f"No saved autotune result for op={op_name!r}, shape={_normalize_shape(shape)}, dtype={_normalize_dtype(dtype)}")
    return entry["best_backend"]


def record_best_backend(op_name, shape, dtype, backend, timing, path=None, extra=None):
    results = load_autotune_results(path)
    key = make_tuning_key(op_name, shape, dtype, extra=extra)
    timings = timing if isinstance(timing, dict) and backend in timing else {backend: timing}
    entry = {
        "op": op_name,
        "shape": _normalize_shape(shape),
        "dtype": _normalize_dtype(dtype),
        "system": _system_info(),
        "best_backend": backend,
        "timings": timings,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "extra": _normalize_extra(extra),
    }
    results.setdefault("entries", {})[key] = entry
    save_autotune_results(results, path=path)
    return entry


def _default_backend(op_name: str) -> str:
    if op_name not in _DEFAULT_BACKENDS:
        raise KeyError(f"No conservative default backend registered for op={op_name!r}")
    return _DEFAULT_BACKENDS[op_name]


def select_backend(
    op_name,
    shape,
    dtype,
    *,
    default_backend=None,
    path=None,
    extra=None,
    require_tuned=False,
):
    try:
        return lookup_best_backend(op_name, shape, dtype, path=path, extra=extra)
    except KeyError:
        if require_tuned:
            raise KeyError(
                f"No saved autotune backend for op={op_name!r}, shape={_normalize_shape(shape)}, "
                f"dtype={_normalize_dtype(dtype)}."
            ) from None
        if default_backend is not None:
            return default_backend
        return _default_backend(op_name)


def explain_backend_choice(
    op_name,
    shape,
    dtype,
    *,
    default_backend=None,
    path=None,
    extra=None,
    require_tuned=False,
):
    try:
        backend = lookup_best_backend(op_name, shape, dtype, path=path, extra=extra)
        return (
            f"Using autotuned backend {backend!r} for op={op_name!r}, "
            f"shape={_normalize_shape(shape)}, dtype={_normalize_dtype(dtype)}."
        )
    except KeyError:
        if require_tuned:
            raise
        chosen = default_backend if default_backend is not None else _default_backend(op_name)
        return (
            f"No saved autotune result for op={op_name!r}, shape={_normalize_shape(shape)}, "
            f"dtype={_normalize_dtype(dtype)}; using conservative default {chosen!r}."
        )
