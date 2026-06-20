from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from benchmark_utils import safe_run_case, time_fn, dtype_from_string, write_csv
from collect_system_info import collect_system_info
from ops.activation_ops import swiglu
from ops.attention_ops import fast_attention
from ops.decode_block_ops import decode_block_from_qkv, paged_decode_block_from_qkv
from ops.decode_ops import decode_attention
from ops.fused_ops import fused_decode_step_from_qkv, qkv_rope_cache_update
from ops.layout_ops import qkv_split, qkv_split_rope
from ops.norm_ops import rms_norm
from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention
from ops.quant_ops import dequant_q4, dequant_q8, pack_q4, q4_matvec_decode, q8_matvec_decode
from ops.quantized_decode_block_ops import paged_quantized_decode_block, quantized_decode_block
from ops.rope_ops import apply_rope


def _backend_set(mode: str, stable, experimental):
    if mode == "all":
        return stable + experimental
    if mode == "stable":
        return stable
    if mode == "experimental":
        return experimental
    raise ValueError(f"Unsupported backends mode: {mode}")


def _record(results, suite, kernel, backend, dtype_name, shape, status, timing=None, error=None):
    results.append(
        {
            "suite": suite,
            "kernel": kernel,
            "backend": backend,
            "dtype": dtype_name,
            "shape": shape,
            "status": status,
            "timing": timing,
            "error": error,
        }
    )


def _run(results, suite, kernel, backend, dtype_name, shape, fn, fail_fast):
    wrapped = safe_run_case(f"{suite}:{kernel}:{backend}", fn)
    if wrapped["status"] == "ok":
        _record(results, suite, kernel, backend, dtype_name, shape, "ok", timing=wrapped["result"])
        return
    _record(results, suite, kernel, backend, dtype_name, shape, "error", error=wrapped["error"])
    if fail_fast:
        raise RuntimeError(wrapped["error"])


def _skip(results, suite, kernel, backend, dtype_name, shape, error):
    _record(results, suite, kernel, backend, dtype_name, shape, "skipped", error=error)


def _attention_cases(results, dtype, dtype_name, quick, include_slow, backends_mode, iters, fail_fast):
    shapes = [{"B": 1, "S": 64, "H": 4, "D": 64}] if quick else [
        {"B": 1, "S": 128, "H": 8, "D": 64},
        {"B": 1, "S": 128, "H": 8, "D": 128},
        {"B": 2, "S": 256, "H": 8, "D": 64},
    ]
    backends = _backend_set(backends_mode, ["reference", "baseline"], ["row_parallel", "tiled_kv", "baseline_d64", "baseline_d128"])
    for shape in shapes:
        B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
        Q = mx.random.normal((B, S, H, D)).astype(dtype)
        K = mx.random.normal((B, S, H, D)).astype(dtype)
        V = mx.random.normal((B, S, H, D)).astype(dtype)
        for backend in backends:
            if backend == "baseline_d64" and D != 64:
                _skip(results, "attention", "fast_attention", backend, dtype_name, shape, "requires D == 64")
                continue
            if backend == "baseline_d128" and D != 128:
                _skip(results, "attention", "fast_attention", backend, dtype_name, shape, "requires D == 128")
                continue
            if backend == "tiled_kv" and not include_slow and not quick:
                _skip(results, "attention", "fast_attention", backend, dtype_name, shape, "include with --include-slow")
                continue
            _run(results, "attention", "fast_attention", backend, dtype_name, shape, lambda b=backend, q=Q, k=K, v=V: time_fn(lambda: fast_attention(q, k, v, backend=b), warmup=3, iters=iters), fail_fast)


def _decode_cases(results, dtype, dtype_name, quick, backends_mode, iters, fail_fast):
    shapes = [{"B": 1, "MAX_S": 64, "H": 4, "D": 64, "length": 64}] if quick else [
        {"B": 1, "MAX_S": 128, "H": 8, "D": 64, "length": 128},
        {"B": 1, "MAX_S": 128, "H": 8, "D": 128, "length": 128},
        {"B": 2, "MAX_S": 512, "H": 8, "D": 64, "length": 512},
    ]
    backends = _backend_set(backends_mode, ["reference", "metal"], ["metal_d64", "metal_d128"])
    for shape in shapes:
        B, MAX_S, H, D, length = shape["B"], shape["MAX_S"], shape["H"], shape["D"], shape["length"]
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        for backend in backends:
            if backend == "metal_d64" and D != 64:
                _skip(results, "decode", "decode_attention", backend, dtype_name, shape, "requires D == 64")
                continue
            if backend == "metal_d128" and D != 128:
                _skip(results, "decode", "decode_attention", backend, dtype_name, shape, "requires D == 128")
                continue
            _run(results, "decode", "decode_attention", backend, dtype_name, shape, lambda b=backend, qq=q, kk=K_cache, vv=V_cache, ll=length: time_fn(lambda: decode_attention(qq, kk, vv, lengths=ll, backend=b), warmup=3, iters=iters), fail_fast)


def _paged_decode_cases(results, dtype, dtype_name, quick, backends_mode, iters, fail_fast):
    shapes = [{"B": 1, "MAX_S": 64, "PAGE_SIZE": 16, "H": 4, "D": 64, "length": 64}] if quick else [
        {"B": 1, "MAX_S": 128, "PAGE_SIZE": 16, "H": 8, "D": 64, "length": 128},
        {"B": 2, "MAX_S": 512, "PAGE_SIZE": 16, "H": 8, "D": 64, "length": 512},
    ]
    backends = _backend_set(backends_mode, ["reference", "metal"], ["metal_d64", "metal_d128"])
    for shape in shapes:
        B, MAX_S, PAGE_SIZE, H, D, length = shape["B"], shape["MAX_S"], shape["PAGE_SIZE"], shape["H"], shape["D"], shape["length"]
        K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
        K_pages = mx.random.normal(K_pages.shape).astype(dtype)
        V_pages = mx.random.normal(V_pages.shape).astype(dtype)
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        for backend in backends:
            if backend == "metal_d64" and D != 64:
                _skip(results, "paged_decode", "paged_decode_attention", backend, dtype_name, shape, "requires D == 64")
                continue
            if backend == "metal_d128" and D != 128:
                _skip(results, "paged_decode", "paged_decode_attention", backend, dtype_name, shape, "requires D == 128")
                continue
            _run(results, "paged_decode", "paged_decode_attention", backend, dtype_name, shape, lambda b=backend, qq=q, kp=K_pages, vp=V_pages, bt=block_table, ll=length: time_fn(lambda: paged_decode_attention(qq, kp, vp, bt, ll, backend=b), warmup=3, iters=iters), fail_fast)


def _misc_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    norm_shapes = [{"B": 2, "S": 16, "D": 1024}] if quick else [{"B": 2, "S": 128, "D": 4096}, {"B": 1, "S": 1, "D": 4096}]
    for shape in norm_shapes:
        x = mx.random.normal((shape["B"], shape["S"], shape["D"])).astype(dtype)
        weight = mx.ones((shape["D"],), dtype=dtype)
        for backend in ("reference", "metal"):
            _run(results, "norm", "rms_norm", backend, dtype_name, shape, lambda b=backend, xx=x, ww=weight: time_fn(lambda: rms_norm(xx, ww, backend=b), warmup=3, iters=iters), fail_fast)

    rope_shapes = [{"B": 1, "S": 32, "H": 8, "D": 64}] if quick else [{"B": 1, "S": 128, "H": 8, "D": 64}, {"B": 1, "S": 128, "H": 8, "D": 128}]
    for shape in rope_shapes:
        x = mx.random.normal((shape["B"], shape["S"], shape["H"], shape["D"])).astype(dtype)
        cos = mx.random.normal((shape["S"] + 32, shape["D"] // 2)).astype(mx.float32)
        sin = mx.random.normal((shape["S"] + 32, shape["D"] // 2)).astype(mx.float32)
        for backend in ("reference", "metal"):
            _run(results, "rope", "apply_rope", backend, dtype_name, shape, lambda b=backend, xx=x, cc=cos, ss=sin: time_fn(lambda: apply_rope(xx, cc, ss, backend=b), warmup=3, iters=iters), fail_fast)

    swiglu_shapes = [{"B": 2, "S": 16, "D": 1024}] if quick else [{"B": 2, "S": 128, "D": 11008}]
    for shape in swiglu_shapes:
        gate = mx.random.normal((shape["B"], shape["S"], shape["D"])).astype(dtype)
        up = mx.random.normal((shape["B"], shape["S"], shape["D"])).astype(dtype)
        for backend in ("reference", "metal"):
            _run(results, "activation", "swiglu", backend, dtype_name, shape, lambda b=backend, gg=gate, uu=up: time_fn(lambda: swiglu(gg, uu, backend=b), warmup=3, iters=iters), fail_fast)


def _layout_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = [{"B": 1, "S": 16, "H": 4, "D": 64}] if quick else [{"B": 2, "S": 32, "H": 8, "D": 64}]
    for shape in shapes:
        qkv = mx.random.normal((shape["B"], shape["S"], 3 * shape["H"] * shape["D"])).astype(dtype)
        cos = mx.random.normal((shape["S"] + 16, shape["D"] // 2)).astype(mx.float32)
        sin = mx.random.normal((shape["S"] + 16, shape["D"] // 2)).astype(mx.float32)
        K_cache = mx.zeros((shape["B"], shape["S"] + 16, shape["H"], shape["D"]), dtype=dtype)
        V_cache = mx.zeros((shape["B"], shape["S"] + 16, shape["H"], shape["D"]), dtype=dtype)
        _run(results, "layout", "qkv_split", "metal", dtype_name, shape, lambda qq=qkv, hh=shape["H"], dd=shape["D"]: time_fn(lambda: qkv_split(qq, H=hh, D=dd, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "layout", "qkv_split_rope", "metal", dtype_name, shape, lambda qq=qkv, cc=cos, ss=sin, hh=shape["H"], dd=shape["D"]: time_fn(lambda: qkv_split_rope(qq, cc, ss, H=hh, D=dd, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "layout", "qkv_rope_cache_update", "metal", dtype_name, shape, lambda qq=qkv[:, :1, :], kk=K_cache, vv=V_cache, cc=cos, ss=sin, hh=shape["H"], dd=shape["D"]: time_fn(lambda: qkv_rope_cache_update(qq, kk, vv, cc, ss, 0, H=hh, D=dd, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "layout", "fused_decode_step_from_qkv", "metal", dtype_name, shape, lambda qq=qkv[:, :1, :], kk=K_cache, vv=V_cache, cc=cos, ss=sin, hh=shape["H"], dd=shape["D"]: time_fn(lambda: fused_decode_step_from_qkv(qq, kk, vv, cc, ss, 0, H=hh, D=dd, backend="metal"), warmup=3, iters=iters), fail_fast)


def _quant_cases(results, dtype, dtype_name, quick, backends_mode, iters, fail_fast):
    shapes = [{"B": 1, "K": 512, "N": 512, "group_size": 32}] if quick else [{"B": 1, "K": 4096, "N": 4096, "group_size": 32}, {"B": 1, "K": 4096, "N": 11008, "group_size": 32}]
    backends = _backend_set(backends_mode, ["reference", "metal"], ["metal_parallel", "metal_tiled"])
    for shape in shapes:
        B, K, N, group_size = shape["B"], shape["K"], shape["N"], shape["group_size"]
        groups = (K + group_size - 1) // group_size
        x = mx.random.normal((B, K)).astype(dtype)
        q4 = (mx.random.uniform((N, K)) * 16).astype(mx.uint8)
        packed = pack_q4(q4)
        q8 = (mx.random.uniform((N, K)) * 255).astype(mx.uint8)
        scales = mx.random.normal((N, groups)).astype(mx.float32)
        _run(results, "quant", "dequant_q4", "metal", dtype_name, shape, lambda pp=packed, ss=scales, gs=group_size, dt=dtype: time_fn(lambda: dequant_q4(pp, ss, group_size=gs, out_dtype=dt, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "quant", "dequant_q8", "metal", dtype_name, shape, lambda qq=q8, ss=scales, gs=group_size, dt=dtype: time_fn(lambda: dequant_q8(qq, ss, group_size=gs, out_dtype=dt, backend="metal"), warmup=3, iters=iters), fail_fast)
        for backend in backends:
            _run(results, "quant", "q4_matvec_decode", backend, dtype_name, shape, lambda b=backend, xx=x, pp=packed, ss=scales, gs=group_size: time_fn(lambda: q4_matvec_decode(xx, pp, ss, group_size=gs, backend=b), warmup=3, iters=iters), fail_fast)
            _run(results, "quant", "q8_matvec_decode", backend, dtype_name, shape, lambda b=backend, xx=x, qq=q8, ss=scales, gs=group_size: time_fn(lambda: q8_matvec_decode(xx, qq, ss, group_size=gs, backend=b), warmup=3, iters=iters), fail_fast)


def _quant_matvec_tiled_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = (
        [{"bits": 4, "B": 1, "K": 512, "N": 512, "group_size": 32}, {"bits": 8, "B": 1, "K": 512, "N": 512, "group_size": 32}]
        if quick
        else [
            {"bits": 4, "B": 1, "K": 4096, "N": 4096, "group_size": 32},
            {"bits": 8, "B": 1, "K": 4096, "N": 4096, "group_size": 32},
            {"bits": 4, "B": 1, "K": 4096, "N": 11008, "group_size": 32},
        ]
    )
    for shape in shapes:
        bits = shape["bits"]
        B, K, N, group_size = shape["B"], shape["K"], shape["N"], shape["group_size"]
        groups = (K + group_size - 1) // group_size
        x = mx.random.normal((B, K)).astype(dtype)
        scales = mx.random.normal((N, groups)).astype(mx.float32)
        if bits == 4:
            q = (mx.random.uniform((N, K)) * 16).astype(mx.uint8)
            packed = pack_q4(q)
            _run(results, "quant_matvec_tiled", "q4_matvec_decode", "metal_tiled", dtype_name, shape, lambda xx=x, pp=packed, ss=scales, gs=group_size: time_fn(lambda: q4_matvec_decode(xx, pp, ss, group_size=gs, backend="metal_tiled"), warmup=3, iters=iters), fail_fast)
        else:
            q8 = (mx.random.uniform((N, K)) * 255).astype(mx.uint8)
            _run(results, "quant_matvec_tiled", "q8_matvec_decode", "metal_tiled", dtype_name, shape, lambda xx=x, qq=q8, ss=scales, gs=group_size: time_fn(lambda: q8_matvec_decode(xx, qq, ss, group_size=gs, backend="metal_tiled"), warmup=3, iters=iters), fail_fast)


def _decode_block_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = [{"B": 1, "T": 8, "MAX_S": 64, "H": 4, "D": 64}] if quick else [{"B": 1, "T": 32, "MAX_S": 128, "H": 8, "D": 64}]
    for shape in shapes:
        B, MAX_S, H, D = shape["B"], shape["MAX_S"], shape["H"], shape["D"]
        qkv = mx.random.normal((B, 1, 3 * H * D)).astype(dtype)
        cos = mx.random.normal((MAX_S + 32, D // 2)).astype(mx.float32)
        sin = mx.random.normal((MAX_S + 32, D // 2)).astype(mx.float32)
        K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
        V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
        K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, 16, dtype)
        _run(results, "decode_block", "decode_block_from_qkv", "metal", dtype_name, shape, lambda qq=qkv, kk=K_cache, vv=V_cache, cc=cos, ss=sin, hh=H, dd=D: time_fn(lambda: decode_block_from_qkv(qq, kk, vv, cc, ss, 0, H=hh, D=dd, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "decode_block", "paged_decode_block_from_qkv", "metal", dtype_name, shape, lambda qq=qkv, kk=K_pages, vv=V_pages, bt=block_table, cc=cos, ss=sin, hh=H, dd=D: time_fn(lambda: paged_decode_block_from_qkv(qq, kk, vv, bt, cc, ss, 0, H=hh, D=dd, backend="metal"), warmup=3, iters=iters), fail_fast)


def _quantized_decode_block_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = (
        [{"bits": 4, "B": 1, "K": 64, "H": 2, "D": 16, "MAX_S": 8, "PAGE_SIZE": 4, "T": 4, "group_size": 32}]
        if quick
        else [{"bits": 4, "B": 1, "K": 1024, "H": 8, "D": 64, "MAX_S": 64, "PAGE_SIZE": 16, "T": 8, "group_size": 32}]
    )
    for shape in shapes:
        bits = shape["bits"]
        B, K, H, D = shape["B"], shape["K"], shape["H"], shape["D"]
        MAX_S, PAGE_SIZE, group_size = shape["MAX_S"], shape["PAGE_SIZE"], shape["group_size"]
        qkv_groups = (K + group_size - 1) // group_size
        out_groups = (H * D + group_size - 1) // group_size
        qkv_scales = mx.random.normal((3 * H * D, qkv_groups)).astype(mx.float32)
        out_scales = mx.random.normal((K, out_groups)).astype(mx.float32)
        qkv_w = pack_q4((mx.random.uniform((3 * H * D, K)) * 16).astype(mx.uint8)) if bits == 4 else (mx.random.uniform((3 * H * D, K)) * 255).astype(mx.uint8)
        out_w = pack_q4((mx.random.uniform((K, H * D)) * 16).astype(mx.uint8)) if bits == 4 else (mx.random.uniform((K, H * D)) * 255).astype(mx.uint8)
        x = mx.random.normal((B, 1, K)).astype(dtype)
        cos = mx.random.normal((MAX_S + 8, D // 2)).astype(mx.float32)
        sin = mx.random.normal((MAX_S + 8, D // 2)).astype(mx.float32)
        K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
        V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
        K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
        _run(results, "quantized_decode_block", "quantized_decode_block", "parallel+metal", dtype_name, shape, lambda xx=x, qq=qkv_w, qs=qkv_scales, ow=out_w, os=out_scales, kk=K_cache, vv=V_cache, cc=cos, ss=sin, hh=H, dd=D, gs=group_size, bb=bits: time_fn(lambda: quantized_decode_block(xx, qq, qs, ow, os, kk, vv, cc, ss, 0, bits=bb, group_size=gs, H=hh, D=dd, matvec_backend="metal_parallel", block_backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "quantized_decode_block", "paged_quantized_decode_block", "parallel+metal", dtype_name, shape, lambda xx=x, qq=qkv_w, qs=qkv_scales, ow=out_w, os=out_scales, kk=K_pages, vv=V_pages, bt=block_table, cc=cos, ss=sin, hh=H, dd=D, gs=group_size, bb=bits: time_fn(lambda: paged_quantized_decode_block(xx, qq, qs, ow, os, kk, vv, bt, cc, ss, 0, bits=bb, group_size=gs, H=hh, D=dd, matvec_backend="metal_parallel", block_backend="metal"), warmup=3, iters=iters), fail_fast)


def _threadgroup_attention_v2_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    decode_shapes = [{"B": 1, "MAX_S": 64, "H": 4, "D": 64, "length": 64}] if quick else [
        {"B": 1, "MAX_S": 128, "H": 8, "D": 64, "length": 128},
        {"B": 1, "MAX_S": 128, "H": 8, "D": 128, "length": 128},
    ]
    for shape in decode_shapes:
        B, MAX_S, H, D, length = shape["B"], shape["MAX_S"], shape["H"], shape["D"], shape["length"]
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        _run(results, "threadgroup_attention_v2", "decode_attention", "metal", dtype_name, shape, lambda qq=q, kk=K_cache, vv=V_cache, ll=length: time_fn(lambda: decode_attention(qq, kk, vv, lengths=ll, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "threadgroup_attention_v2", "decode_attention", "metal_threadgroup", dtype_name, shape, lambda qq=q, kk=K_cache, vv=V_cache, ll=length: time_fn(lambda: decode_attention(qq, kk, vv, lengths=ll, backend="metal_threadgroup"), warmup=3, iters=iters), fail_fast)

    paged_shapes = [{"B": 1, "MAX_S": 64, "PAGE_SIZE": 16, "H": 4, "D": 64, "length": 64}] if quick else [
        {"B": 1, "MAX_S": 128, "PAGE_SIZE": 16, "H": 8, "D": 64, "length": 128},
    ]
    for shape in paged_shapes:
        B, MAX_S, PAGE_SIZE, H, D, length = shape["B"], shape["MAX_S"], shape["PAGE_SIZE"], shape["H"], shape["D"], shape["length"]
        K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
        K_pages = mx.random.normal(K_pages.shape).astype(dtype)
        V_pages = mx.random.normal(V_pages.shape).astype(dtype)
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        _run(results, "threadgroup_attention_v2", "paged_decode_attention", "metal", dtype_name, shape, lambda qq=q, kk=K_pages, vv=V_pages, bt=block_table, ll=length: time_fn(lambda: paged_decode_attention(qq, kk, vv, bt, ll, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "threadgroup_attention_v2", "paged_decode_attention", "metal_threadgroup", dtype_name, shape, lambda qq=q, kk=K_pages, vv=V_pages, bt=block_table, ll=length: time_fn(lambda: paged_decode_attention(qq, kk, vv, bt, ll, backend="metal_threadgroup"), warmup=3, iters=iters), fail_fast)

    prefill_shapes = [{"B": 1, "S": 64, "H": 4, "D": 64, "causal": False}] if quick else [
        {"B": 1, "S": 128, "H": 8, "D": 64, "causal": False},
    ]
    for shape in prefill_shapes:
        B, S, H, D, causal = shape["B"], shape["S"], shape["H"], shape["D"], shape["causal"]
        Q = mx.random.normal((B, S, H, D)).astype(dtype)
        K = mx.random.normal((B, S, H, D)).astype(dtype)
        V = mx.random.normal((B, S, H, D)).astype(dtype)
        _run(results, "threadgroup_attention_v2", "fast_attention", "baseline", dtype_name, shape, lambda q=Q, k=K, v=V, c=causal: time_fn(lambda: fast_attention(q, k, v, causal=c, backend="baseline"), warmup=3, iters=iters), fail_fast)
        _run(results, "threadgroup_attention_v2", "fast_attention", "threadgroup", dtype_name, shape, lambda q=Q, k=K, v=V, c=causal: time_fn(lambda: fast_attention(q, k, v, causal=c, backend="threadgroup"), warmup=3, iters=iters), fail_fast)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", default="benchmarks/results/benchmark_results.json")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--backends", choices=["all", "stable", "experimental"], default="all")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    quick = True if not args.quick and not args.full else args.quick
    dtype = dtype_from_string(args.dtype)
    mx.random.seed(args.seed)
    iters = 5 if quick else 10

    results = []
    _attention_cases(results, dtype, args.dtype, quick, args.include_slow, args.backends, iters, args.fail_fast)
    _decode_cases(results, dtype, args.dtype, quick, args.backends, iters, args.fail_fast)
    _paged_decode_cases(results, dtype, args.dtype, quick, args.backends, iters, args.fail_fast)
    _misc_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _layout_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _quant_cases(results, dtype, args.dtype, quick, args.backends, iters, args.fail_fast)
    _quant_matvec_tiled_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _decode_block_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _quantized_decode_block_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _threadgroup_attention_v2_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)

    payload = {
        "system_info": collect_system_info(),
        "config": {
            "mode": "quick" if quick else "full",
            "dtype": args.dtype,
            "include_slow": args.include_slow,
            "backends": args.backends,
            "fail_fast": args.fail_fast,
            "seed": args.seed,
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv:
        write_csv(results, args.csv)
    print(f"Wrote JSON benchmark results to {out_path}")
    if args.csv:
        print(f"Wrote CSV benchmark results to {args.csv}")


if __name__ == "__main__":
    main()
