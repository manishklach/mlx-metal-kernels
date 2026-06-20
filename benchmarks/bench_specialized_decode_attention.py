import argparse
import time

import mlx.core as mx

from ops.decode_ops import decode_attention


def _run(backend, q, K_cache, V_cache, lengths, D):
    if backend == "generic":
        return decode_attention(q, K_cache, V_cache, lengths=lengths, backend="metal")
    if backend == "specialized":
        return decode_attention(q, K_cache, V_cache, lengths=lengths, backend=f"metal_d{D}")
    return decode_attention(q, K_cache, V_cache, lengths=lengths, backend="reference")


def _time_fn(fn, iters):
    for _ in range(5):
        y = fn()
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(y)
    end = time.perf_counter()
    return (end - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, choices=[64, 128], default=64)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["generic", "specialized", "all"], default="all")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(111)
    q = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
    K_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
    V_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
    lengths = min(args.length, args.MAX_S)
    backends = ["reference", "generic", "specialized"] if args.backend == "all" else [args.backend]
    timings = {}
    for backend in backends:
        timings[backend] = _time_fn(lambda b=backend: _run(b, q, K_cache, V_cache, lengths, args.D), args.iters)
        ms = timings[backend] * 1e3
        speed_generic = f"{timings['generic'] / timings[backend]:.3f}" if backend != "generic" and "generic" in timings else "n/a"
        speed_ref = f"{timings['reference'] / timings[backend]:.3f}" if backend != "reference" and "reference" in timings else "n/a"
        print(
            f"backend={backend} B={args.B} MAX_S={args.MAX_S} H={args.H} D={args.D} "
            f"length={lengths} dtype={args.dtype} milliseconds={ms:.3f} "
            f"speedup_vs_generic={speed_generic} speedup_vs_reference={speed_ref}"
        )


if __name__ == "__main__":
    main()
