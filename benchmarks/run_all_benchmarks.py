from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from benchmark_utils import safe_run_case, time_fn, dtype_from_string, write_csv
from collect_system_info import collect_system_info
from ops.activation_ops import swiglu
from ops.attention_ops import fast_attention
from ops.autotune_ops import select_backend
from ops.decode_block_ops import decode_block_from_qkv, paged_decode_block_from_qkv
from ops.decode_ops import decode_attention
from ops.fused_ops import fused_decode_step_from_qkv, qkv_rope_cache_update
from ops.gqa_ops import gqa_attention, gqa_decode_block_from_qkv, reference_gqa_attention, reference_gqa_decode_attention, reference_paged_gqa_decode_attention
from ops.layout_ops import qkv_split, qkv_split_rope
from ops.llama_layer_ops import create_random_quantized_llama_layer_weights, init_llama_layer_cache, llama_layer_decode_loop
from ops.llama_prefill_ops import _build_rope_tables_numpy, fused_experimental_prefill_backend_config, llama_stack_prefill, metal_prefill_backend_config, reference_prefill_backend_config, tiled_prefill_backend_config
from ops.llama_stack_ops import create_random_quantized_llama_stack_weights, init_llama_stack_cache, llama_stack_decode_loop
from ops.mlp_block_ops import quantized_mlp_block, reference_quantized_mlp_block
from ops.norm_ops import rms_norm
from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention
from ops.quant_ops import dequant_q4, dequant_q8, pack_q4, q4_matvec_decode, q8_matvec_decode
from ops.quantized_kv_cache_ops import QuantizedKVCacheConfig, quantize_kv_cache, quantized_kv_gqa_decode_attention
from ops.quantized_decode_block_ops import paged_quantized_decode_block, quantized_decode_block
from ops.rope_ops import apply_rope
from ops.sparse_attention_ops import SparseAttentionPattern, sparse_gqa_attention, sparse_gqa_decode_attention
from ops.toy_transformer_ops import make_toy_layer_weights, paged_toy_transformer_decode_layer, toy_transformer_decode_layer
from models.llama_config import LlamaLikeConfig
from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig


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


def _choose_backends(op_name: str, shape: dict, dtype_name: str, default_backend: str, use_autotune: bool, extra=None, candidates=None):
    if not use_autotune:
        return list(candidates or [])
    return [select_backend(op_name, shape, dtype_name, default_backend=default_backend, extra=extra)]


def _attention_cases(results, dtype, dtype_name, quick, include_slow, backends_mode, iters, fail_fast, use_autotune):
    shapes = [{"B": 1, "S": 64, "H": 4, "D": 64}] if quick else [
        {"B": 1, "S": 128, "H": 8, "D": 64},
        {"B": 1, "S": 128, "H": 8, "D": 128},
        {"B": 2, "S": 256, "H": 8, "D": 64},
    ]
    base_backends = _backend_set(backends_mode, ["reference", "baseline"], ["row_parallel", "tiled_kv", "baseline_d64", "baseline_d128"])
    for shape in shapes:
        B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
        Q = mx.random.normal((B, S, H, D)).astype(dtype)
        K = mx.random.normal((B, S, H, D)).astype(dtype)
        V = mx.random.normal((B, S, H, D)).astype(dtype)
        backends = _choose_backends("fast_attention", shape, dtype_name, "baseline", use_autotune, extra={"causal": False}, candidates=base_backends)
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


def _decode_cases(results, dtype, dtype_name, quick, backends_mode, iters, fail_fast, use_autotune):
    shapes = [{"B": 1, "MAX_S": 64, "H": 4, "D": 64, "length": 64}] if quick else [
        {"B": 1, "MAX_S": 128, "H": 8, "D": 64, "length": 128},
        {"B": 1, "MAX_S": 128, "H": 8, "D": 128, "length": 128},
        {"B": 2, "MAX_S": 512, "H": 8, "D": 64, "length": 512},
    ]
    base_backends = _backend_set(backends_mode, ["reference", "metal"], ["metal_d64", "metal_d128"])
    for shape in shapes:
        B, MAX_S, H, D, length = shape["B"], shape["MAX_S"], shape["H"], shape["D"], shape["length"]
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        K_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        V_cache = mx.random.normal((B, MAX_S, H, D)).astype(dtype)
        backends = _choose_backends("decode_attention", shape, dtype_name, "metal", use_autotune, extra={"length": length}, candidates=base_backends)
        for backend in backends:
            if backend == "metal_d64" and D != 64:
                _skip(results, "decode", "decode_attention", backend, dtype_name, shape, "requires D == 64")
                continue
            if backend == "metal_d128" and D != 128:
                _skip(results, "decode", "decode_attention", backend, dtype_name, shape, "requires D == 128")
                continue
            _run(results, "decode", "decode_attention", backend, dtype_name, shape, lambda b=backend, qq=q, kk=K_cache, vv=V_cache, ll=length: time_fn(lambda: decode_attention(qq, kk, vv, lengths=ll, backend=b), warmup=3, iters=iters), fail_fast)


def _paged_decode_cases(results, dtype, dtype_name, quick, backends_mode, iters, fail_fast, use_autotune):
    shapes = [{"B": 1, "MAX_S": 64, "PAGE_SIZE": 16, "H": 4, "D": 64, "length": 64}] if quick else [
        {"B": 1, "MAX_S": 128, "PAGE_SIZE": 16, "H": 8, "D": 64, "length": 128},
        {"B": 2, "MAX_S": 512, "PAGE_SIZE": 16, "H": 8, "D": 64, "length": 512},
    ]
    base_backends = _backend_set(backends_mode, ["reference", "metal"], ["metal_d64", "metal_d128"])
    for shape in shapes:
        B, MAX_S, PAGE_SIZE, H, D, length = shape["B"], shape["MAX_S"], shape["PAGE_SIZE"], shape["H"], shape["D"], shape["length"]
        K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
        K_pages = mx.random.normal(K_pages.shape).astype(dtype)
        V_pages = mx.random.normal(V_pages.shape).astype(dtype)
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        backends = _choose_backends("paged_decode_attention", shape, dtype_name, "metal", use_autotune, extra={"length": length, "PAGE_SIZE": PAGE_SIZE}, candidates=base_backends)
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


def _quant_cases(results, dtype, dtype_name, quick, backends_mode, iters, fail_fast, use_autotune):
    shapes = [{"B": 1, "K": 512, "N": 512, "group_size": 32}] if quick else [{"B": 1, "K": 4096, "N": 4096, "group_size": 32}, {"B": 1, "K": 4096, "N": 11008, "group_size": 32}]
    base_backends = _backend_set(backends_mode, ["reference", "metal"], ["metal_parallel", "metal_tiled"])
    for shape in shapes:
        B, K, N, group_size = shape["B"], shape["K"], shape["N"], shape["group_size"]
        groups = (K + group_size - 1) // group_size
        x = mx.random.normal((B, K)).astype(dtype)
        q4 = (mx.random.uniform((N, K)) * 16).astype(mx.uint8)
        packed = pack_q4(q4)
        q8 = (mx.random.uniform((N, K)) * 255).astype(mx.uint8)
        scales = mx.random.normal((N, groups)).astype(mx.float32)
        q4_backends = _choose_backends("q4_matvec_decode", shape, dtype_name, "metal_parallel", use_autotune, extra={"bits": 4, "group_size": group_size}, candidates=base_backends)
        q8_backends = _choose_backends("q8_matvec_decode", shape, dtype_name, "metal_parallel", use_autotune, extra={"bits": 8, "group_size": group_size}, candidates=base_backends)
        _run(results, "quant", "dequant_q4", "metal", dtype_name, shape, lambda pp=packed, ss=scales, gs=group_size, dt=dtype: time_fn(lambda: dequant_q4(pp, ss, group_size=gs, out_dtype=dt, backend="metal"), warmup=3, iters=iters), fail_fast)
        _run(results, "quant", "dequant_q8", "metal", dtype_name, shape, lambda qq=q8, ss=scales, gs=group_size, dt=dtype: time_fn(lambda: dequant_q8(qq, ss, group_size=gs, out_dtype=dt, backend="metal"), warmup=3, iters=iters), fail_fast)
        for backend in q4_backends:
            _run(results, "quant", "q4_matvec_decode", backend, dtype_name, shape, lambda b=backend, xx=x, pp=packed, ss=scales, gs=group_size: time_fn(lambda: q4_matvec_decode(xx, pp, ss, group_size=gs, backend=b), warmup=3, iters=iters), fail_fast)
        for backend in q8_backends:
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


def _quantized_mlp_block_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = (
        [{"bits": 4, "B": 1, "S": 1, "hidden_size": 64, "intermediate_size": 128, "group_size": 32}]
        if quick
        else [
            {"bits": 4, "B": 1, "S": 1, "hidden_size": 4096, "intermediate_size": 11008, "group_size": 32},
            {"bits": 4, "B": 1, "S": 16, "hidden_size": 4096, "intermediate_size": 11008, "group_size": 32},
        ]
    )
    presets = {
        "reference": {"norm_backend": "reference", "matvec_backend": "reference", "activation_backend": "reference", "residual_backend": "reference"},
        "metal": {"norm_backend": "metal", "matvec_backend": "metal", "activation_backend": "metal", "residual_backend": "metal"},
        "parallel": {"norm_backend": "metal", "matvec_backend": "metal_parallel", "activation_backend": "metal", "residual_backend": "metal"},
        "tiled": {"norm_backend": "metal", "matvec_backend": "metal_tiled", "activation_backend": "metal", "residual_backend": "metal"},
    }
    for shape in shapes:
        bits = shape["bits"]
        B, S = shape["B"], shape["S"]
        hidden_size, intermediate_size, group_size = shape["hidden_size"], shape["intermediate_size"], shape["group_size"]
        groups_hidden = (hidden_size + group_size - 1) // group_size
        groups_intermediate = (intermediate_size + group_size - 1) // group_size
        x = mx.random.normal((B, S, hidden_size)).astype(dtype)
        residual = mx.random.normal((B, S, hidden_size)).astype(dtype)
        norm_weight = mx.ones((hidden_size,), dtype=dtype)
        gate_w = pack_q4((mx.random.uniform((intermediate_size, hidden_size)) * 16).astype(mx.uint8))
        up_w = pack_q4((mx.random.uniform((intermediate_size, hidden_size)) * 16).astype(mx.uint8))
        down_w = pack_q4((mx.random.uniform((hidden_size, intermediate_size)) * 16).astype(mx.uint8))
        gate_scales = mx.random.normal((intermediate_size, groups_hidden)).astype(mx.float32)
        up_scales = mx.random.normal((intermediate_size, groups_hidden)).astype(mx.float32)
        down_scales = mx.random.normal((hidden_size, groups_intermediate)).astype(mx.float32)
        for preset, kwargs in presets.items():
            _run(results, "quantized_mlp_block", "quantized_mlp_block", preset, dtype_name, shape, lambda xx=x, rr=residual, nw=norm_weight, gw=gate_w, gs=gate_scales, uw=up_w, us=up_scales, dw=down_w, ds=down_scales, bb=bits, gsz=group_size, kk=kwargs: time_fn(lambda: quantized_mlp_block(xx, rr, nw, gw, gs, uw, us, dw, ds, bits=bb, group_size=gsz, **kk), warmup=3, iters=iters), fail_fast)


def _fused_quantized_mlp_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    shapes = (
        [{"bits": 4, "B": 1, "S": 1, "hidden_size": 64, "intermediate_size": 128, "group_size": 32}]
        if quick
        else [{"bits": 4, "B": 1, "S": 1, "hidden_size": 4096, "intermediate_size": 11008, "group_size": 32}]
    )
    base_backends = ["reference", "tiled", "fused_experimental"]
    for shape in shapes:
        bits = shape["bits"]
        B, S = shape["B"], shape["S"]
        hidden_size, intermediate_size, group_size = shape["hidden_size"], shape["intermediate_size"], shape["group_size"]
        groups_hidden = (hidden_size + group_size - 1) // group_size
        groups_intermediate = (intermediate_size + group_size - 1) // group_size
        x = mx.random.normal((B, S, hidden_size)).astype(dtype)
        residual = mx.random.normal((B, S, hidden_size)).astype(dtype)
        norm_weight = mx.ones((hidden_size,), dtype=dtype)
        gate_w = pack_q4((mx.random.uniform((intermediate_size, hidden_size)) * 16).astype(mx.uint8))
        up_w = pack_q4((mx.random.uniform((intermediate_size, hidden_size)) * 16).astype(mx.uint8))
        down_w = pack_q4((mx.random.uniform((hidden_size, intermediate_size)) * 16).astype(mx.uint8))
        gate_scales = mx.random.normal((intermediate_size, groups_hidden)).astype(mx.float32)
        up_scales = mx.random.normal((intermediate_size, groups_hidden)).astype(mx.float32)
        down_scales = mx.random.normal((hidden_size, groups_intermediate)).astype(mx.float32)
        backends = _choose_backends("quantized_mlp_block", shape, dtype_name, "tiled", use_autotune, extra={"bits": bits, "group_size": group_size}, candidates=base_backends)
        for backend in backends:
            if backend == "reference":
                fn = lambda xx=x, rr=residual, nw=norm_weight, gw=gate_w, gs=gate_scales, uw=up_w, us=up_scales, dw=down_w, ds=down_scales, bb=bits, gsz=group_size: time_fn(  # noqa: E731
                    lambda: reference_quantized_mlp_block(xx, rr, nw, gw, gs, uw, us, dw, ds, bits=bb, group_size=gsz),
                    warmup=3,
                    iters=iters,
                )
            else:
                fn = lambda b=backend, xx=x, rr=residual, nw=norm_weight, gw=gate_w, gs=gate_scales, uw=up_w, us=up_scales, dw=down_w, ds=down_scales, bb=bits, gsz=group_size: time_fn(  # noqa: E731
                    lambda: quantized_mlp_block(xx, rr, nw, gw, gs, uw, us, dw, ds, bits=bb, group_size=gsz, backend_preset=b),
                    warmup=3,
                    iters=iters,
                )
            _run(results, "fused_quantized_mlp", "quantized_mlp_block", backend, dtype_name, shape, fn, fail_fast)


def _gqa_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = (
        [{"B": 1, "MAX_S": 32, "PAGE_SIZE": 4, "Hq": 4, "Hkv": 2, "D": 16, "T": 4, "length": 32}]
        if quick
        else [{"B": 1, "MAX_S": 128, "PAGE_SIZE": 16, "Hq": 32, "Hkv": 8, "D": 128, "T": 8, "length": 128}]
    )
    for shape in shapes:
        B, MAX_S, PAGE_SIZE, Hq, Hkv, D, T, length = (
            shape["B"],
            shape["MAX_S"],
            shape["PAGE_SIZE"],
            shape["Hq"],
            shape["Hkv"],
            shape["D"],
            shape["T"],
            shape["length"],
        )
        q = mx.random.normal((B, 1, Hq, D)).astype(dtype)
        K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
        V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
        K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, Hkv, D, PAGE_SIZE, dtype)
        K_pages = mx.random.normal(K_pages.shape).astype(dtype)
        V_pages = mx.random.normal(V_pages.shape).astype(dtype)
        cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
        sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
        qkv = mx.random.normal((B, 1, Hq * D + 2 * Hkv * D)).astype(dtype)
        _run(results, "gqa", "decode_attention", "reference", dtype_name, shape, lambda qq=q, kk=K_cache, vv=V_cache, ll=length: time_fn(lambda: reference_gqa_decode_attention(qq, kk, vv, lengths=ll), warmup=3, iters=iters), fail_fast)
        _run(results, "gqa", "paged_decode_attention", "reference", dtype_name, shape, lambda qq=q, kk=K_pages, vv=V_pages, bt=block_table, ll=length: time_fn(lambda: reference_paged_gqa_decode_attention(qq, kk, vv, bt, lengths=ll), warmup=3, iters=iters), fail_fast)
        _run(results, "gqa", "decode_block", "reference", dtype_name, shape, lambda qq=qkv, kk=mx.zeros((B, MAX_S, Hkv, D), dtype=dtype), vv=mx.zeros((B, MAX_S, Hkv, D), dtype=dtype), cc=cos, ss=sin: time_fn(lambda: gqa_decode_block_from_qkv(qq, kk, vv, cc, ss, 0, num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=D, backend="reference"), warmup=3, iters=iters), fail_fast)


def _gqa_prefill_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    shapes = (
        [
            {"B": 1, "Sq": 32, "Sk": 32, "Hq": 4, "Hkv": 2, "D": 16, "causal": True},
            {"B": 1, "Sq": 32, "Sk": 32, "Hq": 4, "Hkv": 2, "D": 16, "causal": False},
        ]
        if quick
        else [
            {"B": 1, "Sq": 128, "Sk": 128, "Hq": 32, "Hkv": 8, "D": 128, "causal": True},
            {"B": 1, "Sq": 128, "Sk": 128, "Hq": 32, "Hkv": 8, "D": 128, "causal": False},
        ]
    )
    base_backends = ["reference", "metal_gqa", "metal_gqa_threadgroup"]
    for shape in shapes:
        B, Sq, Sk, Hq, Hkv, D, causal = shape["B"], shape["Sq"], shape["Sk"], shape["Hq"], shape["Hkv"], shape["D"], shape["causal"]
        Q = mx.random.normal((B, Sq, Hq, D)).astype(dtype)
        K = mx.random.normal((B, Sk, Hkv, D)).astype(dtype)
        V = mx.random.normal((B, Sk, Hkv, D)).astype(dtype)
        backends = _choose_backends("gqa_attention", shape, dtype_name, "metal_gqa", use_autotune, extra={"causal": causal}, candidates=base_backends)
        for backend in backends:
            fn = (
                lambda qq=Q, kk=K, vv=V, c=causal: time_fn(lambda: reference_gqa_attention(qq, kk, vv, causal=c), warmup=3, iters=iters)
                if backend == "reference"
                else lambda b=backend, qq=Q, kk=K, vv=V, c=causal: time_fn(lambda: gqa_attention(qq, kk, vv, causal=c, backend=b), warmup=3, iters=iters)
            )
            if backend == "reference":
                _run(results, "gqa_prefill_attention", "gqa_attention", backend, dtype_name, shape, lambda qq=Q, kk=K, vv=V, c=causal: time_fn(lambda: reference_gqa_attention(qq, kk, vv, causal=c), warmup=3, iters=iters), fail_fast)
            else:
                _run(results, "gqa_prefill_attention", "gqa_attention", backend, dtype_name, shape, lambda b=backend, qq=Q, kk=K, vv=V, c=causal: time_fn(lambda: gqa_attention(qq, kk, vv, causal=c, backend=b), warmup=3, iters=iters), fail_fast)


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


def _sparse_attention_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    shapes = (
        [{"B": 1, "S": 32, "Hq": 4, "Hkv": 2, "D": 16, "window_size": 8, "sink_tokens": 2}]
        if quick
        else [{"B": 1, "S": 512, "Hq": 32, "Hkv": 8, "D": 128, "window_size": 128, "sink_tokens": 4}]
    )
    base_backends = ["reference", "metal_sliding_window", "metal_sliding_window_sink"]
    for shape in shapes:
        B, S, Hq, Hkv, D = shape["B"], shape["S"], shape["Hq"], shape["Hkv"], shape["D"]
        Q = mx.random.normal((B, S, Hq, D)).astype(dtype)
        K = mx.random.normal((B, S, Hkv, D)).astype(dtype)
        V = mx.random.normal((B, S, Hkv, D)).astype(dtype)
        backends = _choose_backends("sparse_gqa_attention", shape, dtype_name, "metal_sliding_window", use_autotune, extra={"window_size": shape["window_size"], "sink_tokens": shape["sink_tokens"]}, candidates=base_backends)
        for backend in backends:
            sink_tokens = shape["sink_tokens"] if backend == "metal_sliding_window_sink" else 0
            pattern = SparseAttentionPattern(
                pattern="sliding_window_sink" if sink_tokens > 0 else "sliding_window",
                window_size=shape["window_size"],
                sink_tokens=sink_tokens,
            )
            _run(results, "sparse_attention", "sparse_gqa_attention", backend, dtype_name, shape, lambda b=backend, q=Q, k=K, v=V, p=pattern: time_fn(lambda: sparse_gqa_attention(q, k, v, p, backend=b), warmup=3, iters=iters), fail_fast)


def _sparse_decode_attention_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    shapes = (
        [{"B": 1, "MAX_S": 32, "length": 32, "Hq": 4, "Hkv": 2, "D": 16, "window_size": 8, "sink_tokens": 2}]
        if quick
        else [{"B": 1, "MAX_S": 4096, "length": 4096, "Hq": 32, "Hkv": 8, "D": 128, "window_size": 512, "sink_tokens": 4}]
    )
    base_backends = ["reference", "metal_sliding_window", "metal_sliding_window_sink"]
    for shape in shapes:
        B, MAX_S, Hq, Hkv, D = shape["B"], shape["MAX_S"], shape["Hq"], shape["Hkv"], shape["D"]
        q = mx.random.normal((B, 1, Hq, D)).astype(dtype)
        K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
        V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
        backends = _choose_backends("sparse_gqa_decode_attention", shape, dtype_name, "metal_sliding_window", use_autotune, extra={"window_size": shape["window_size"], "sink_tokens": shape["sink_tokens"], "length": shape["length"]}, candidates=base_backends)
        for backend in backends:
            sink_tokens = shape["sink_tokens"] if backend == "metal_sliding_window_sink" else 0
            pattern = SparseAttentionPattern(
                pattern="sliding_window_sink" if sink_tokens > 0 else "sliding_window",
                window_size=shape["window_size"],
                sink_tokens=sink_tokens,
            )
            _run(results, "sparse_decode_attention", "sparse_gqa_decode_attention", backend, dtype_name, shape, lambda b=backend, qq=q, kk=K_cache, vv=V_cache, p=pattern, ll=shape["length"]: time_fn(lambda: sparse_gqa_decode_attention(qq, kk, vv, ll, p, backend=b), warmup=3, iters=iters), fail_fast)


def _simdgroup_attention_cases(results, dtype, dtype_name, quick, iters):
    shapes = [{"B": 1, "S": 32, "H": 4, "D": 64, "causal": False}] if quick else [
        {"B": 1, "S": 128, "H": 8, "D": 64, "causal": False},
    ]
    for shape in shapes:
        if dtype != mx.float16:
            _skip(results, "simdgroup_attention", "fast_attention", "simdgroup_d64", dtype_name, shape, "requires dtype == float16")
            continue
        B, S, H, D, causal = shape["B"], shape["S"], shape["H"], shape["D"], shape["causal"]
        Q = mx.random.normal((B, S, H, D)).astype(dtype)
        K = mx.random.normal((B, S, H, D)).astype(dtype)
        V = mx.random.normal((B, S, H, D)).astype(dtype)
        _run(results, "simdgroup_attention", "fast_attention", "baseline", dtype_name, shape, lambda q=Q, k=K, v=V, c=causal: time_fn(lambda: fast_attention(q, k, v, causal=c, backend="baseline"), warmup=3, iters=iters), False)
        _run(results, "simdgroup_attention", "fast_attention", "threadgroup", dtype_name, shape, lambda q=Q, k=K, v=V, c=causal: time_fn(lambda: fast_attention(q, k, v, causal=c, backend="threadgroup"), warmup=3, iters=iters), False)
        _run(results, "simdgroup_attention", "fast_attention", "simdgroup_d64", dtype_name, shape, lambda q=Q, k=K, v=V, c=causal: time_fn(lambda: fast_attention(q, k, v, causal=c, backend="simdgroup_d64"), warmup=3, iters=iters), False)


def _toy_transformer_decode_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    shapes = (
        [{"cache": "contiguous", "bits": 4, "B": 1, "K": 256, "H": 4, "D": 64, "INTERMEDIATE": 512, "MAX_S": 32, "PAGE_SIZE": 8, "T": 4}]
        if quick
        else [
            {"cache": "contiguous", "bits": 4, "B": 1, "K": 512, "H": 8, "D": 64, "INTERMEDIATE": 1024, "MAX_S": 64, "PAGE_SIZE": 16, "T": 8},
            {"cache": "paged", "bits": 4, "B": 1, "K": 512, "H": 8, "D": 64, "INTERMEDIATE": 1024, "MAX_S": 64, "PAGE_SIZE": 16, "T": 8},
        ]
    )
    for shape in shapes:
        cache = shape["cache"]
        bits = shape["bits"]
        B, K, H, D = shape["B"], shape["K"], shape["H"], shape["D"]
        intermediate = shape["INTERMEDIATE"]
        MAX_S, PAGE_SIZE, T = shape["MAX_S"], shape["PAGE_SIZE"], shape["T"]
        weights = make_toy_layer_weights(K, intermediate, bits=bits, group_size=32, num_attention_heads=H, head_dim=D)
        cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
        sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
        x = mx.random.normal((B, 1, K)).astype(dtype)
        if cache == "contiguous":
            K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
            V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
            _run(results, "toy_transformer_decode", "toy_transformer_decode_layer", "parallel+metal", dtype_name, shape, lambda xx=x, kk=K_cache, vv=V_cache, ww=weights, cc=cos, ss=sin: time_fn(lambda: toy_transformer_decode_layer(xx, ww["attn_norm_weight"].astype(dtype), ww["ffn_norm_weight"].astype(dtype), ww["qkv_w"], ww["qkv_scales"], ww["out_w"], ww["out_scales"], ww["gate_w"], ww["gate_scales"], ww["up_w"], ww["up_scales"], ww["down_w"], ww["down_scales"], kk, vv, cc, ss, 0, bits=bits, group_size=32, H=H, D=D, matvec_backend="metal_parallel", block_backend="metal"), warmup=3, iters=iters), fail_fast)
        else:
            K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
            _run(results, "toy_transformer_decode", "paged_toy_transformer_decode_layer", "parallel+metal", dtype_name, shape, lambda xx=x, kk=K_pages, vv=V_pages, bt=block_table, ww=weights, cc=cos, ss=sin: time_fn(lambda: paged_toy_transformer_decode_layer(xx, ww["attn_norm_weight"].astype(dtype), ww["ffn_norm_weight"].astype(dtype), ww["qkv_w"], ww["qkv_scales"], ww["out_w"], ww["out_scales"], ww["gate_w"], ww["gate_scales"], ww["up_w"], ww["up_scales"], ww["down_w"], ww["down_scales"], kk, vv, bt, cc, ss, 0, bits=bits, group_size=32, H=H, D=D, matvec_backend="metal_parallel", block_backend="metal"), warmup=3, iters=iters), fail_fast)


def _llama_layer_decode_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    shapes = (
        [{"bits": 4, "B": 1, "T": 4, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "MAX_S": 8, "cache": "contiguous"}]
        if quick
        else [{"bits": 4, "B": 1, "T": 16, "hidden_size": 512, "intermediate_size": 2048, "num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "MAX_S": 128, "cache": "contiguous"}]
    )
    base_backends = ["reference", "metal", "tiled", "fused_experimental"]
    for shape in shapes:
        cfg = LlamaLikeConfig(
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            num_attention_heads=shape["num_heads"],
            num_key_value_heads=shape["num_kv_heads"],
            head_dim=shape["head_dim"],
            num_hidden_layers=1,
            max_position_embeddings=shape["MAX_S"],
        ).validate()
        weights = create_random_quantized_llama_layer_weights(cfg, bits=shape["bits"], group_size=32, dtype=dtype, seed=shape["bits"] + shape["head_dim"])
        inputs = mx.random.normal((shape["B"], shape["T"], shape["hidden_size"])).astype(dtype)
        cos = mx.random.normal((shape["MAX_S"] + 4, shape["head_dim"] // 2)).astype(mx.float32)
        sin = mx.random.normal((shape["MAX_S"] + 4, shape["head_dim"] // 2)).astype(mx.float32)
        backends = _choose_backends("llama_layer_decode", shape, dtype_name, "tiled", use_autotune, extra={"cache": shape["cache"], "bits": shape["bits"]}, candidates=base_backends)
        for backend in backends:
            _run(
                results,
                "llama_layer_decode",
                "llama_layer_decode_loop",
                backend,
                dtype_name,
                shape,
                lambda b=backend, xx=inputs, ww=weights, co=cos, si=sin, cf=cfg, cl=shape["cache"], sh=shape: time_fn(
                    lambda: llama_layer_decode_loop(
                        xx,
                        ww,
                        init_llama_layer_cache(cf, sh["B"], sh["MAX_S"], cache_layout=cl, dtype=dtype),
                        co,
                        si,
                        cf,
                        backend_preset=b,
                        cache_layout=cl,
                    ),
                    warmup=3,
                    iters=iters,
                ),
                fail_fast,
            )


def _llama_stack_decode_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    shapes = (
        [{"bits": 4, "B": 1, "T": 4, "num_layers": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "MAX_S": 8, "cache": "contiguous", "vocab_size": 64}]
        if quick
        else [{"bits": 4, "B": 1, "T": 16, "num_layers": 2, "hidden_size": 512, "intermediate_size": 2048, "num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "MAX_S": 128, "cache": "contiguous", "vocab_size": 128}]
    )
    base_backends = ["reference", "metal", "tiled", "fused_experimental"]
    for shape in shapes:
        cfg = LlamaLikeConfig(
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            num_attention_heads=shape["num_heads"],
            num_key_value_heads=shape["num_kv_heads"],
            head_dim=shape["head_dim"],
            num_hidden_layers=shape["num_layers"],
            max_position_embeddings=shape["MAX_S"],
            vocab_size=shape["vocab_size"],
        ).validate()
        weights = create_random_quantized_llama_stack_weights(cfg, vocab_size=shape["vocab_size"], bits=shape["bits"], group_size=32, dtype=dtype, seed=shape["bits"] + shape["head_dim"])
        inputs = mx.random.normal((shape["B"], shape["T"], shape["hidden_size"])).astype(dtype)
        cos = mx.random.normal((shape["MAX_S"] + 4, shape["head_dim"] // 2)).astype(mx.float32)
        sin = mx.random.normal((shape["MAX_S"] + 4, shape["head_dim"] // 2)).astype(mx.float32)
        backends = _choose_backends("llama_stack_decode", shape, dtype_name, "tiled", use_autotune, extra={"cache": shape["cache"], "bits": shape["bits"]}, candidates=base_backends)
        for backend in backends:
            _run(
                results,
                "llama_stack_decode",
                "llama_stack_decode_loop",
                backend,
                dtype_name,
                shape,
                lambda b=backend, xx=inputs, ww=weights, co=cos, si=sin, cf=cfg, cl=shape["cache"], sh=shape: time_fn(
                    lambda: llama_stack_decode_loop(
                        xx,
                        ww,
                        init_llama_stack_cache(cf, sh["B"], sh["MAX_S"], cache_layout=cl, dtype=dtype),
                        co,
                        si,
                        cf,
                        backend_preset=b,
                        cache_layout=cl,
                        return_logits=True,
                    ),
                    warmup=3,
                    iters=iters,
                ),
                fail_fast,
            )


def _tiny_generation_pipeline_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    _ = dtype
    _ = dtype_name
    shapes = (
        [{"bits": 4, "prompt_len": 4, "max_new_tokens": 4, "num_layers": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "vocab_size": 128}]
        if quick
        else [{"bits": 4, "prompt_len": 8, "max_new_tokens": 16, "num_layers": 2, "hidden_size": 512, "intermediate_size": 2048, "num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "vocab_size": 128}]
    )
    backends = ["reference", "metal", "tiled", "fused_experimental"]
    for shape in shapes:
        prompt = "a" * shape["prompt_len"]
        for backend in backends:
            config = TinyGenerationPipelineConfig(
                hidden_size=shape["hidden_size"],
                intermediate_size=shape["intermediate_size"],
                num_attention_heads=shape["num_heads"],
                num_key_value_heads=shape["num_kv_heads"],
                head_dim=shape["head_dim"],
                num_hidden_layers=shape["num_layers"],
                max_position_embeddings=shape["prompt_len"] + shape["max_new_tokens"] + 16,
                vocab_size=shape["vocab_size"],
                bits=shape["bits"],
                backend_preset=backend,
            ).validate()
            pipeline = TinyGenerationPipeline(config=config)
            _run(
                results,
                "tiny_generation_pipeline",
                "generate",
                backend,
                dtype_name,
                shape,
                lambda p=pipeline, text=prompt, max_new=shape["max_new_tokens"]: time_fn(
                    lambda: (p.generate(text, max_new_tokens=max_new, greedy=True), mx.array(0.0))[1],
                    warmup=1,
                    iters=iters,
                ),
                fail_fast,
            )


def _llama_stack_prefill_cases(results, dtype, dtype_name, quick, iters, fail_fast, use_autotune):
    _ = use_autotune
    shapes = (
        [{"bits": 4, "B": 1, "S": 8, "num_layers": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "MAX_S": 16, "cache": "contiguous", "vocab_size": 64}]
        if quick
        else [{"bits": 4, "B": 1, "S": 64, "num_layers": 2, "hidden_size": 512, "intermediate_size": 2048, "num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "MAX_S": 128, "cache": "contiguous", "vocab_size": 128}]
    )
    backend_map = {
        "reference": reference_prefill_backend_config,
        "metal": metal_prefill_backend_config,
        "tiled": tiled_prefill_backend_config,
        "fused_experimental": fused_experimental_prefill_backend_config,
    }
    for shape in shapes:
        cfg = LlamaLikeConfig(
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            num_attention_heads=shape["num_heads"],
            num_key_value_heads=shape["num_kv_heads"],
            head_dim=shape["head_dim"],
            num_hidden_layers=shape["num_layers"],
            max_position_embeddings=shape["MAX_S"],
            vocab_size=shape["vocab_size"],
        ).validate()
        weights = create_random_quantized_llama_stack_weights(cfg, vocab_size=shape["vocab_size"], bits=shape["bits"], group_size=32, dtype=dtype, seed=shape["bits"] + shape["head_dim"])
        inputs = mx.random.normal((shape["B"], shape["S"], shape["hidden_size"])).astype(dtype)
        cos = mx.random.normal((shape["MAX_S"] + 4, shape["head_dim"] // 2)).astype(mx.float32)
        sin = mx.random.normal((shape["MAX_S"] + 4, shape["head_dim"] // 2)).astype(mx.float32)
        for backend, backend_fn in backend_map.items():
            _run(
                results,
                "llama_stack_prefill",
                "llama_stack_prefill",
                backend,
                dtype_name,
                shape,
                lambda b=backend_fn, xx=inputs, ww=weights, co=cos, si=sin, cf=cfg, sh=shape: time_fn(
                    lambda: llama_stack_prefill(
                        xx,
                        ww,
                        init_llama_stack_cache(cf, sh["B"], sh["MAX_S"], cache_layout=sh["cache"], dtype=dtype),
                        co,
                        si,
                        cf,
                        backend_config=b(),
                        return_logits=True,
                    ),
                    warmup=3,
                    iters=iters,
                ),
                fail_fast,
            )


def _prefix_cache_reuse_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    from models.prefix_cache import prefill_with_prefix_reuse, InMemoryPrefixCache, compute_fingerprint
    from models.generation import GenerationConfig

    shapes = (
        [{"prompt_tokens": 4, "reused_tokens": 3}]
        if quick
        else [{"prompt_tokens": 8, "reused_tokens": 6}]
    )
    for shape in shapes:
        from models import create_synthetic_stack_generation_model
        model = create_synthetic_stack_generation_model(seed=42)
        gen_config = GenerationConfig(max_new_tokens=2, backend_preset="reference")
        prompt_ids = list(range(shape["prompt_tokens"]))
        cache = InMemoryPrefixCache(max_size=16)
        prefill_with_prefix_reuse(prompt_ids, model, prefix_cache=cache, generation_config=gen_config)
        reused_ids = list(range(shape["reused_tokens"]))
        _run(
            results,
            "prefix_cache_reuse",
            "prefill_with_prefix_reuse",
            "in_memory",
            dtype_name,
            shape,
            lambda m=model, ids=prompt_ids, c=cache, gc=gen_config: time_fn(
                lambda: (prefill_with_prefix_reuse(ids, m, prefix_cache=c, generation_config=gc), 0)[1],
                warmup=1,
                iters=iters,
            ),
            fail_fast,
        )


def _kv_offload_tier_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    if quick:
        shapes = [{"seq_len": 256, "block_size": 64, "num_layers": 1, "num_kv_heads": 2, "head_dim": 16}]
    else:
        shapes = [
            {"seq_len": 256, "block_size": 64, "num_layers": 1, "num_kv_heads": 2, "head_dim": 16},
            {"seq_len": 4096, "block_size": 128, "num_layers": 2, "num_kv_heads": 8, "head_dim": 128},
        ]
    import numpy as np
    from models.kv_offload import KVResidencyMap, partition_sequence_into_blocks
    from models.kv_offload_policy import KVOffloadPolicyConfig, plan_offload_blocks
    from models.kv_offload_store import InMemoryKVOffloadStore
    from ops.kv_offload_ops import apply_offload_plan

    for shape in shapes:
        rng = np.random.default_rng(0)
        seq_len = shape["seq_len"]
        block_size = shape["block_size"]
        num_layers = shape["num_layers"]
        num_kv_heads = shape["num_kv_heads"]
        head_dim = shape["head_dim"]

        caches = []
        for layer in range(num_layers):
            K = rng.normal(0, 0.02, (1, seq_len, num_kv_heads, head_dim)).astype(np.float32)
            V = rng.normal(0, 0.02, (1, seq_len, num_kv_heads, head_dim)).astype(np.float32)
            caches.append((K, V))

        rmap = KVResidencyMap()
        for layer in range(num_layers):
            blocks = partition_sequence_into_blocks(
                layer_idx=layer, batch_idx=0,
                seq_len=seq_len, block_size=block_size,
                num_kv_heads=num_kv_heads, head_dim=head_dim,
                dtype="float32",
            )
            for b in blocks:
                rmap.add_block(b)

        store = InMemoryKVOffloadStore()
        policy = KVOffloadPolicyConfig(
            block_size=block_size,
            keep_sink_blocks=1,
            keep_recent_blocks=2,
            max_resident_blocks=max(1, (seq_len // block_size) // 2),
        ).validate()

        plan = plan_offload_blocks(rmap, current_position=seq_len - 1, policy_config=policy)
        _run(
            results,
            "kv_offload_tier",
            "apply_offload_plan",
            "memory",
            dtype_name,
            shape,
            lambda cs=caches, rm=rmap, st=store, pl=plan: time_fn(
                lambda: apply_offload_plan(cs, rm, st, pl),
                warmup=1,
                iters=iters,
            ),
            fail_fast,
        )


def _quantized_kv_decode_attention_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    if quick:
        shapes = [{"bits": 8, "B": 1, "MAX_S": 32, "length": 32, "Hq": 4, "Hkv": 2, "D": 16, "group_size": 16}]
    else:
        shapes = [
            {"bits": 8, "B": 1, "MAX_S": 4096, "length": 4096, "Hq": 32, "Hkv": 8, "D": 128, "group_size": 32},
            {"bits": 4, "B": 1, "MAX_S": 4096, "length": 4096, "Hq": 32, "Hkv": 8, "D": 128, "group_size": 32},
        ]
    backends = ["reference", "metal_q8", "metal_q4"]
    for shape in shapes:
        B, MAX_S, length, Hq, Hkv, D = shape["B"], shape["MAX_S"], shape["length"], shape["Hq"], shape["Hkv"], shape["D"]
        bits, group_size = shape["bits"], shape["group_size"]
        q = mx.random.normal((B, 1, Hq, D)).astype(dtype)
        K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
        V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
        qkv = quantize_kv_cache(K_cache, V_cache, QuantizedKVCacheConfig(bits=bits, group_size=group_size))
        for backend in backends:
            if backend == "metal_q8" and bits != 8:
                _skip(results, "quantized_kv_decode_attention", f"q{bits}_decode_attention", backend, dtype_name, shape, "bits mismatch")
                continue
            if backend == "metal_q4" and bits != 4:
                _skip(results, "quantized_kv_decode_attention", f"q{bits}_decode_attention", backend, dtype_name, shape, "bits mismatch")
                continue
            if D > 128 and backend != "reference":
                _skip(results, "quantized_kv_decode_attention", f"q{bits}_decode_attention", backend, dtype_name, shape, "D > 128 not supported")
                continue
            _run(results, "quantized_kv_decode_attention", f"q{bits}_decode_attention", backend, dtype_name, shape,
                 lambda b=backend, qq=q, qqkv=qkv, ll=length: time_fn(
                     lambda: quantized_kv_gqa_decode_attention(qq, qqkv, lengths=ll, backend=b),
                     warmup=3, iters=iters,
                 ), fail_fast)


def _speculative_decoding_cases(results, dtype, dtype_name, quick, iters, fail_fast):
    if quick:
        shapes = [{"prompt_tokens": 4, "max_new_tokens": 4, "draft_length": 2}]
    else:
        shapes = [
            {"prompt_tokens": 4, "max_new_tokens": 8, "draft_length": 2},
            {"prompt_tokens": 8, "max_new_tokens": 16, "draft_length": 4},
        ]
    for shape in shapes:
        from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig
        from models.speculative_decoding import FixedDraftProposer, SpeculativeConfig, SpeculativeGenerator

        pipe_cfg = TinyGenerationPipelineConfig(
            hidden_size=32, intermediate_size=64,
            num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, num_hidden_layers=1,
            max_position_embeddings=max(shape["prompt_tokens"] + shape["max_new_tokens"] + 8, 32),
            vocab_size=64, bits=4, group_size=32,
            backend_preset="reference",
        ).validate()
        pipeline = TinyGenerationPipeline(config=pipe_cfg)
        prompt_ids = list(range(shape["prompt_tokens"]))
        spec_cfg = SpeculativeConfig(
            draft_length=shape["draft_length"],
            max_new_tokens=shape["max_new_tokens"],
            temperature=1.0,
            greedy_verify=True,
            seed=0,
            backend_preset="reference",
        ).validate()
        for draft_mode in ("fixed", "random"):
            if draft_mode == "fixed":
                proposer = FixedDraftProposer(list(range(pipeline.vocab_size))[:shape["draft_length"]])
            else:
                from models.speculative_decoding import RandomDraftProposer
                proposer = RandomDraftProposer(pipeline.vocab_size, seed=0)
            gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=spec_cfg)
            _run(
                results,
                "speculative_decoding",
                f"generate_ids_{draft_mode}",
                "reference",
                dtype_name,
                shape,
                lambda g=gen, ids=prompt_ids, sc=spec_cfg: time_fn(
                    lambda: g.generate_ids(ids, speculative_config=sc),
                    warmup=1,
                    iters=iters,
                ),
                fail_fast,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", default="benchmarks/results/benchmark_results.json")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--backends", choices=["all", "stable", "experimental"], default="all")
    parser.add_argument("--use-autotune", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    quick = True if not args.quick and not args.full else args.quick
    dtype = dtype_from_string(args.dtype)
    mx.random.seed(args.seed)
    iters = 5 if quick else 10

    results = []
    _attention_cases(results, dtype, args.dtype, quick, args.include_slow, args.backends, iters, args.fail_fast, args.use_autotune)
    _decode_cases(results, dtype, args.dtype, quick, args.backends, iters, args.fail_fast, args.use_autotune)
    _paged_decode_cases(results, dtype, args.dtype, quick, args.backends, iters, args.fail_fast, args.use_autotune)
    _misc_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _layout_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _quant_cases(results, dtype, args.dtype, quick, args.backends, iters, args.fail_fast, args.use_autotune)
    _quant_matvec_tiled_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _decode_block_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _quantized_decode_block_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _quantized_mlp_block_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _fused_quantized_mlp_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _gqa_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _gqa_prefill_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _sparse_attention_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _sparse_decode_attention_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _threadgroup_attention_v2_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _simdgroup_attention_cases(results, dtype, args.dtype, quick, iters)
    _toy_transformer_decode_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _llama_layer_decode_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _llama_stack_decode_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _llama_stack_prefill_cases(results, dtype, args.dtype, quick, iters, args.fail_fast, args.use_autotune)
    _tiny_generation_pipeline_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _prefix_cache_reuse_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _kv_offload_tier_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _quantized_kv_decode_attention_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)
    _speculative_decoding_cases(results, dtype, args.dtype, quick, iters, args.fail_fast)

    payload = {
        "system_info": collect_system_info(),
        "config": {
            "mode": "quick" if quick else "full",
            "dtype": args.dtype,
            "include_slow": args.include_slow,
            "backends": args.backends,
            "use_autotune": args.use_autotune,
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
