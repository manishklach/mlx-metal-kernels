import argparse
import math
import time

import mlx.core as mx

from ops.attention_ops import fast_attention, reference_attention


def tolerances(dtype):
    if dtype == mx.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2


def time_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(y)
    end = time.perf_counter()
    return (end - start) / iters


def run_attention_backend(backend, Q, K, V, scale, causal):
    return fast_attention(Q, K, V, scale=scale, causal=causal, backend=backend)


def validate_backend(backend, Q, K, V, scale, causal):
    if backend == "reference":
        return

    got = run_attention_backend(backend, Q, K, V, scale, causal)
    ref = reference_attention(Q, K, V, scale=scale, causal=causal)
    mx.eval(got, ref)
    atol, rtol = tolerances(Q.dtype)
    if not mx.allclose(got, ref, atol=atol, rtol=rtol).item():
        raise AssertionError(
            f"backend={backend} failed validation against reference_attention "
            f"for shape={Q.shape}, causal={causal}, dtype={Q.dtype}"
        )


def bench_case(backend, B, S, H, D, dtype_name, causal, iters):
    dtype = mx.float16 if dtype_name == "float16" else mx.bfloat16
    mx.random.seed(0)
    Q = mx.random.normal((B, S, H, D)).astype(dtype)
    K = mx.random.normal((B, S, H, D)).astype(dtype)
    V = mx.random.normal((B, S, H, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)
    validate_backend(backend, Q, K, V, scale, causal)

    ref_ms = time_fn(
        lambda: reference_attention(Q, K, V, scale=scale, causal=causal),
        iters=iters,
    ) * 1e3
    cur_ms = time_fn(
        lambda: run_attention_backend(backend, Q, K, V, scale, causal),
        iters=iters,
    ) * 1e3
    speedup = ref_ms / cur_ms if cur_ms > 0 else float("inf")
    return {
        "backend": backend,
        "B": B,
        "S": S,
        "H": H,
        "D": D,
        "dtype": dtype_name,
        "causal": causal,
        "milliseconds": cur_ms,
        "speedup_vs_reference": speedup,
    }


def print_table(rows):
    headers = [
        "backend",
        "B",
        "S",
        "H",
        "D",
        "dtype",
        "causal",
        "milliseconds",
        "speedup_vs_reference",
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "backend": row["backend"],
                "B": str(row["B"]),
                "S": str(row["S"]),
                "H": str(row["H"]),
                "D": str(row["D"]),
                "dtype": row["dtype"],
                "causal": str(row["causal"]),
                "milliseconds": f"{row['milliseconds']:.3f}",
                "speedup_vs_reference": f"{row['speedup_vs_reference']:.2f}x",
            }
        )

    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(sep_line)
    for row in formatted_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=128)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--backend",
        choices=["baseline", "row_parallel", "tiled_kv", "reference", "all"],
        default="baseline",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()

    backends = ["baseline", "row_parallel", "tiled_kv", "reference"] if args.backend == "all" else [args.backend]
    rows = []
    if args.matrix:
        for S in (32, 64, 128, 256):
            for D in (32, 64, 128):
                for causal in (False, True):
                    for backend in backends:
                        rows.append(
                            bench_case(
                                backend=backend,
                                B=args.B,
                                S=S,
                                H=args.H,
                                D=D,
                                dtype_name=args.dtype,
                                causal=causal,
                                iters=args.iters,
                            )
                        )
    else:
        for backend in backends:
            rows.append(
                bench_case(
                    backend=backend,
                    B=args.B,
                    S=args.S,
                    H=args.H,
                    D=args.D,
                    dtype_name=args.dtype,
                    causal=args.causal,
                    iters=args.iters,
                )
            )

    print_table(rows)


if __name__ == "__main__":
    main()
