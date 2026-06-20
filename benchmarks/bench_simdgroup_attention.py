from __future__ import annotations

import argparse
import math

import mlx.core as mx

from benchmark_utils import safe_run_case, time_fn
from ops.attention_ops import fast_attention, reference_attention
from ops.decode_ops import decode_attention, reference_decode_attention


def _dtype_from_string(dtype_name: str):
    if dtype_name == "float16":
        return mx.float16
    if dtype_name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _attention_backends(mode: str):
    if mode == "reference":
        return ["reference"]
    if mode == "baseline":
        return ["baseline"]
    if mode == "threadgroup":
        return ["threadgroup"]
    if mode == "simdgroup":
        return ["simdgroup_d64"]
    return ["reference", "baseline", "threadgroup", "simdgroup_d64"]


def _decode_backends(mode: str):
    if mode == "reference":
        return ["reference"]
    if mode == "baseline":
        return ["metal"]
    if mode == "threadgroup":
        return ["metal_threadgroup"]
    if mode == "simdgroup":
        return ["simdgroup_d64"]
    return ["reference", "metal", "metal_threadgroup", "metal_d64", "simdgroup_d64"]


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 3e-2, 3e-2


def _validate_prefill(backend: str, Q, K, V, scale, causal):
    if backend == "reference":
        return
    got = fast_attention(Q, K, V, scale=scale, causal=causal, backend=backend)
    ref = reference_attention(Q, K, V, scale=scale, causal=causal)
    mx.eval(got, ref)
    atol, rtol = _tol(Q.dtype)
    if not mx.allclose(got, ref, atol=atol, rtol=rtol).item():
        raise AssertionError(f"backend={backend} failed validation for prefill attention")


def _validate_decode(backend: str, q, K_cache, V_cache, lengths, scale):
    if backend == "reference":
        return
    got = decode_attention(q, K_cache, V_cache, lengths=lengths, scale=scale, backend=backend)
    ref = reference_decode_attention(q, K_cache, V_cache, lengths=lengths, scale=scale)
    mx.eval(got, ref)
    atol, rtol = _tol(q.dtype)
    if not mx.allclose(got, ref, atol=atol, rtol=rtol).item():
        raise AssertionError(f"backend={backend} failed validation for decode attention")


def _baseline_ms(rows, backend_name: str):
    for row in rows:
        if row["backend"] == backend_name and row["status"] == "ok":
            return row["mean_ms"]
    return None


def _print_rows(rows):
    baseline_ms = _baseline_ms(rows, "baseline") or _baseline_ms(rows, "metal")
    threadgroup_ms = _baseline_ms(rows, "threadgroup") or _baseline_ms(rows, "metal_threadgroup")
    reference_ms = _baseline_ms(rows, "reference")
    for row in rows:
        if row["status"] != "ok":
            print(
                f"mode={row['mode']} backend={row['backend']} B={row['B']} "
                f"S={row.get('S', '-')}/MAX_S={row.get('MAX_S', '-')} H={row['H']} D={row['D']} "
                f"dtype={row['dtype']} causal={row['causal']} status={row['status']} error={row['error']}"
            )
            continue
        mean_ms = row["mean_ms"]
        speedup_vs_baseline = baseline_ms / mean_ms if baseline_ms and mean_ms > 0 else float("nan")
        speedup_vs_threadgroup = threadgroup_ms / mean_ms if threadgroup_ms and mean_ms > 0 else float("nan")
        speedup_vs_reference = reference_ms / mean_ms if reference_ms and mean_ms > 0 else float("nan")
        print(
            f"mode={row['mode']} backend={row['backend']} B={row['B']} "
            f"S={row.get('S', '-')}/MAX_S={row.get('MAX_S', '-')} H={row['H']} D={row['D']} "
            f"dtype={row['dtype']} causal={row['causal']} mean_ms={mean_ms:.3f} "
            f"speedup_vs_baseline={speedup_vs_baseline:.3f} "
            f"speedup_vs_threadgroup={speedup_vs_threadgroup:.3f} "
            f"speedup_vs_reference={speedup_vs_reference:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=128)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backend", choices=["reference", "baseline", "threadgroup", "simdgroup", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = _dtype_from_string(args.dtype)
    mx.random.seed(217)
    rows = []

    if args.mode == "prefill":
        Q = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
        K = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
        V = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
        scale = 1.0 / math.sqrt(args.D)
        for backend in _attention_backends(args.backend):
            result = safe_run_case(
                f"prefill:{backend}",
                lambda b=backend: (
                    _validate_prefill(b, Q, K, V, scale, args.causal),
                    time_fn(lambda: fast_attention(Q, K, V, scale=scale, causal=args.causal, backend=b), warmup=3, iters=args.iters),
                )[1],
            )
            row = {
                "mode": "prefill",
                "backend": backend,
                "B": args.B,
                "S": args.S,
                "H": args.H,
                "D": args.D,
                "dtype": args.dtype,
                "causal": args.causal,
                "status": result["status"],
                "error": result["error"],
            }
            if result["status"] == "ok":
                row["mean_ms"] = result["result"]["mean_ms"]
            rows.append(row)
    else:
        q = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
        K_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
        V_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
        scale = 1.0 / math.sqrt(args.D)
        for backend in _decode_backends(args.backend):
            if backend == "simdgroup_d64":
                rows.append(
                    {
                        "mode": "decode",
                        "backend": backend,
                        "B": args.B,
                        "MAX_S": args.MAX_S,
                        "H": args.H,
                        "D": args.D,
                        "dtype": args.dtype,
                        "causal": False,
                        "status": "skipped",
                        "error": "simdgroup decode backend not implemented in PR #15",
                    }
                )
                continue
            result = safe_run_case(
                f"decode:{backend}",
                lambda b=backend: (
                    _validate_decode(b, q, K_cache, V_cache, args.length, scale),
                    time_fn(lambda: decode_attention(q, K_cache, V_cache, lengths=args.length, scale=scale, backend=b), warmup=3, iters=args.iters),
                )[1],
            )
            row = {
                "mode": "decode",
                "backend": backend,
                "B": args.B,
                "MAX_S": args.MAX_S,
                "H": args.H,
                "D": args.D,
                "dtype": args.dtype,
                "causal": False,
                "status": result["status"],
                "error": result["error"],
            }
            if result["status"] == "ok":
                row["mean_ms"] = result["result"]["mean_ms"]
            rows.append(row)

    _print_rows(rows)


if __name__ == "__main__":
    main()
