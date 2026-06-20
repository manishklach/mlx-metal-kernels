from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from backend_registry import (
    filter_backends_for_shape,
    get_candidate_backends,
    get_reference_backend,
    is_experimental_backend,
    list_ops,
)
from benchmark_utils import dtype_from_string, safe_run_case, time_fn
from collect_system_info import collect_system_info
from ops.attention_ops import fast_attention, reference_attention
from ops.autotune_ops import load_autotune_results, make_tuning_key, record_best_backend, save_autotune_results
from ops.decode_ops import decode_attention, reference_decode_attention
from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention, reference_paged_decode_attention
from ops.quant_ops import pack_q4, q4_matvec_decode, q8_matvec_decode


def _shapes_for_op(op_name: str, quick: bool):
    if op_name == "fast_attention":
        return [{"B": 1, "S": 64, "H": 4, "D": 64}] if quick else [
            {"B": 1, "S": 128, "H": 8, "D": 64},
            {"B": 1, "S": 128, "H": 8, "D": 128},
        ]
    if op_name == "decode_attention":
        return [{"B": 1, "MAX_S": 64, "H": 4, "D": 64, "length": 64}] if quick else [
            {"B": 1, "MAX_S": 128, "H": 8, "D": 64, "length": 128},
            {"B": 1, "MAX_S": 128, "H": 8, "D": 128, "length": 128},
        ]
    if op_name == "paged_decode_attention":
        return [{"B": 1, "MAX_S": 64, "PAGE_SIZE": 16, "H": 4, "D": 64, "length": 64}] if quick else [
            {"B": 1, "MAX_S": 128, "PAGE_SIZE": 16, "H": 8, "D": 64, "length": 128},
            {"B": 1, "MAX_S": 128, "PAGE_SIZE": 16, "H": 8, "D": 128, "length": 128},
        ]
    if op_name in ("q4_matvec_decode", "q8_matvec_decode"):
        return [{"B": 1, "K": 512, "N": 512, "group_size": 32}] if quick else [
            {"B": 1, "K": 4096, "N": 4096, "group_size": 32},
            {"B": 1, "K": 4096, "N": 11008, "group_size": 32},
        ]
    raise KeyError(f"Unsupported op: {op_name}")


def _is_close(a, b, dtype):
    atol, rtol = (3e-2, 3e-2) if dtype == mx.bfloat16 else (3e-2, 3e-2)
    return mx.allclose(a, b, atol=atol, rtol=rtol).item()


def _filter_experimental(backends, include_experimental: bool):
    if include_experimental:
        return list(backends)
    return [backend for backend in backends if not is_experimental_backend(backend)]


def _build_case(op_name: str, shape: dict, dtype, seed: int):
    mx.random.seed(seed)
    if op_name == "fast_attention":
        B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
        q = mx.random.normal((B, S, H, D)).astype(dtype)
        k = mx.random.normal((B, S, H, D)).astype(dtype)
        v = mx.random.normal((B, S, H, D)).astype(dtype)
        return {
            "extra": {"causal": False},
            "reference": lambda: reference_attention(q, k, v, causal=False),
            "run": lambda backend: fast_attention(q, k, v, causal=False, backend=backend),
        }
    if op_name == "decode_attention":
        B, MAX_S, H, D, length = shape["B"], shape["MAX_S"], shape["H"], shape["D"], shape["length"]
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        k_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        v_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        return {
            "extra": {"length": length},
            "reference": lambda: reference_decode_attention(q, k_cache, v_cache, lengths=length),
            "run": lambda backend: decode_attention(q, k_cache, v_cache, lengths=length, backend=backend),
        }
    if op_name == "paged_decode_attention":
        B, MAX_S, PAGE_SIZE, H, D, length = shape["B"], shape["MAX_S"], shape["PAGE_SIZE"], shape["H"], shape["D"], shape["length"]
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        k_pages, v_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
        k_pages = mx.random.normal(k_pages.shape).astype(dtype)
        v_pages = mx.random.normal(v_pages.shape).astype(dtype)
        return {
            "extra": {"length": length, "PAGE_SIZE": PAGE_SIZE},
            "reference": lambda: reference_paged_decode_attention(q, k_pages, v_pages, block_table, length),
            "run": lambda backend: paged_decode_attention(q, k_pages, v_pages, block_table, length, backend=backend),
        }
    if op_name in ("q4_matvec_decode", "q8_matvec_decode"):
        B, K, N, group_size = shape["B"], shape["K"], shape["N"], shape["group_size"]
        groups = (K + group_size - 1) // group_size
        x = mx.random.normal((B, K)).astype(dtype)
        scales = mx.random.normal((N, groups)).astype(mx.float32)
        if op_name == "q4_matvec_decode":
            q = (mx.random.uniform((N, K)) * 16).astype(mx.uint8)
            packed = pack_q4(q)
            return {
                "extra": {"bits": 4, "group_size": group_size},
                "reference": lambda: q4_matvec_decode(x, packed, scales, group_size=group_size, backend="reference"),
                "run": lambda backend: q4_matvec_decode(x, packed, scales, group_size=group_size, backend=backend),
            }
        q8 = (mx.random.uniform((N, K)) * 255).astype(mx.uint8)
        return {
            "extra": {"bits": 8, "group_size": group_size},
            "reference": lambda: q8_matvec_decode(x, q8, scales, group_size=group_size, backend="reference"),
            "run": lambda backend: q8_matvec_decode(x, q8, scales, group_size=group_size, backend=backend),
        }
    raise KeyError(f"Unsupported op: {op_name}")


def _validate_candidate(case, backend: str, dtype):
    if backend == "reference":
        return
    got = case["run"](backend)
    ref = case["reference"]()
    mx.eval(got, ref)
    if not _is_close(got, ref, dtype):
        raise AssertionError(f"backend={backend} failed reference validation")


def _time_candidate(case, backend: str, warmup: int, iters: int):
    fn = case["reference"] if backend == "reference" else lambda: case["run"](backend)
    return time_fn(fn, warmup=warmup, iters=iters)


def _autotune_one(op_name: str, shape: dict, dtype, dtype_name: str, args):
    case = _build_case(op_name, shape, dtype, args.seed)
    reference_backend = get_reference_backend(op_name)
    candidates = get_candidate_backends(op_name)
    candidates = _filter_experimental(candidates, args.include_experimental)
    candidates = filter_backends_for_shape(op_name, shape, dtype_name, candidates)

    timings = {}
    failures = {}
    if reference_backend not in candidates:
        candidates = [reference_backend] + candidates

    for backend in candidates:
        result = safe_run_case(
            f"{op_name}:{backend}",
            lambda b=backend: (
                None if b == reference_backend else _validate_candidate(case, b, dtype),
                _time_candidate(case, b, args.warmup, args.iters),
            )[1],
        )
        if result["status"] == "ok":
            timings[backend] = result["result"]
        else:
            failures[backend] = result["error"]
            if args.fail_fast:
                raise RuntimeError(result["error"])

    valid_non_reference = {backend: timing for backend, timing in timings.items() if backend != reference_backend}
    if valid_non_reference:
        best_backend = min(valid_non_reference, key=lambda name: valid_non_reference[name]["mean_ms"])
        status = "ok"
    elif reference_backend in timings:
        best_backend = reference_backend
        status = "reference_only"
    else:
        best_backend = None
        status = "error"

    return {
        "op": op_name,
        "shape": shape,
        "dtype": dtype_name,
        "system": collect_system_info(),
        "reference_backend": reference_backend,
        "best_backend": best_backend,
        "timings": timings,
        "failures": failures,
        "timestamp_utc": collect_system_info()["timestamp_utc"],
        "status": status,
        "extra": case.get("extra", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=list_ops() + ["all"], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", default="benchmarks/results/autotune_results.json")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-cache", action="store_true")
    args = parser.parse_args()

    quick = True if not args.quick and not args.full else args.quick
    dtype = dtype_from_string(args.dtype)
    ops = list_ops() if args.op == "all" else [args.op]

    entries = []
    for op_name in ops:
        for shape in _shapes_for_op(op_name, quick):
            entries.append(_autotune_one(op_name, shape, dtype, args.dtype, args))

    payload = {
        "system_info": collect_system_info(),
        "config": {
            "op": args.op,
            "mode": "quick" if quick else "full",
            "dtype": args.dtype,
            "iters": args.iters,
            "warmup": args.warmup,
            "include_experimental": args.include_experimental,
            "fail_fast": args.fail_fast,
            "seed": args.seed,
            "write_cache": args.write_cache,
        },
        "results": entries,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote autotune results to {out_path}")

    if args.write_cache:
        cache = load_autotune_results()
        for entry in entries:
            if entry["best_backend"] is None:
                continue
            key = make_tuning_key(entry["op"], entry["shape"], entry["dtype"], extra=entry.get("extra"))
            cache.setdefault("entries", {})[key] = entry
        save_autotune_results(cache)
        print("Updated local autotune cache")


if __name__ == "__main__":
    main()
