from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def sync(value=None):
    if value is None:
        return None
    if isinstance(value, tuple):
        mx.eval(*value)
        return value
    if isinstance(value, list):
        mx.eval(*value)
        return value
    mx.eval(value)
    return value


def time_fn(fn, warmup: int = 5, iters: int = 20):
    if warmup < 0 or iters <= 0:
        raise ValueError(f"warmup must be >= 0 and iters must be > 0, got warmup={warmup}, iters={iters}")
    for _ in range(warmup):
        sync(fn())
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        sync(fn())
        end = time.perf_counter()
        samples.append((end - start) * 1e3)
    return summarize_times(samples)


def summarize_times(times):
    if not times:
        raise ValueError("times must be non-empty")
    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "iters": len(times),
    }


def dtype_from_string(dtype_name: str):
    if dtype_name == "float16":
        return mx.float16
    if dtype_name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"Unsupported dtype string: {dtype_name}")


def make_random(shape, dtype, seed: int = 0):
    mx.random.seed(seed)
    return mx.random.normal(shape).astype(dtype)


def safe_run_case(case_name: str, fn):
    try:
        return {"status": "ok", "case_name": case_name, "result": fn(), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "case_name": case_name, "result": None, "error": str(exc)}


def write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "suite",
                "kernel",
                "backend",
                "dtype",
                "shape_json",
                "status",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "std_ms",
                "iters",
                "error",
            ],
        )
        writer.writeheader()
        for item in results:
            timing = item.get("timing") or {}
            writer.writerow(
                {
                    "suite": item.get("suite"),
                    "kernel": item.get("kernel"),
                    "backend": item.get("backend"),
                    "dtype": item.get("dtype"),
                    "shape_json": json.dumps(item.get("shape", {}), sort_keys=True),
                    "status": item.get("status"),
                    "mean_ms": timing.get("mean_ms"),
                    "median_ms": timing.get("median_ms"),
                    "min_ms": timing.get("min_ms"),
                    "max_ms": timing.get("max_ms"),
                    "std_ms": timing.get("std_ms"),
                    "iters": timing.get("iters"),
                    "error": item.get("error"),
                }
            )
