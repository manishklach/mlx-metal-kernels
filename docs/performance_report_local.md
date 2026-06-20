# MLX Metal Kernels — Local Evidence Report

This file records the exact local audit performed in this checkout. It does not contain Apple Silicon benchmark numbers because the required MLX runtime was not available on this machine. In line with the repo's design principles, unmeasured rows are marked `not yet measured` rather than estimated.

## 1. Machine & environment

| Field | Value |
|---|---|
| Machine | Acer Aspire Lite AL15-41 |
| Chip / GPU | AMD Ryzen-based Windows laptop with `AMD Radeon(TM) Graphics`; not an Apple Silicon machine |
| Memory | 16,471,355,392 bytes physical RAM; `TotalVisibleMemorySize` reported as `16085308` KiB |
| Operating system | Microsoft Windows 11 Home Single Language, version `10.0.26200` |
| Python version | `Python 3.11.9` |
| MLX version | not yet measured; `pip install mlx --break-system-packages` failed with `No matching distribution found for mlx` |
| Repo commit | `e03927a` |
| Date audited | `2026-06-20` |
| Power state | not measured |
| Other apps running | not measured |

## 2. Correctness gate status

The required Phase 0 commands were attempted in this checkout. Because `mlx` could not be installed, the correctness gate was never reached and no benchmark or autotune command was run.

Commands attempted:

```bash
pip install mlx pytest --break-system-packages
pip install -e . --break-system-packages
pytest tests -q
python examples/run_basic.py
```

Observed output:

```text
> pip install mlx pytest --break-system-packages
Defaulting to user installation because normal site-packages is not writeable
ERROR: Could not find a version that satisfies the requirement mlx (from versions: none)
ERROR: No matching distribution found for mlx
```

```text
> pip install -e . --break-system-packages
Defaulting to user installation because normal site-packages is not writeable
Obtaining file:///C:/Users/ManishKL/Documents/Playground/mlx-flash-attention-metal
INFO: pip is looking at multiple versions of mlx-flash-attention-metal to determine which version is compatible with other requirements. This could take a while.
ERROR: Could not find a version that satisfies the requirement mlx (from mlx-flash-attention-metal) (from versions: none)
ERROR: No matching distribution found for mlx
```

```text
> pytest tests -q
pytest:
Line |
   2 |  pytest tests -q
     |  ~~~~~~
     | The term 'pytest' is not recognized as a name of a cmdlet, function, script file, or executable program.
Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
```

```text
> python examples/run_basic.py
Traceback (most recent call last):
  File "C:\Users\ManishKL\Documents\Playground\mlx-flash-attention-metal\examples\run_basic.py", line 3, in <module>
    import mlx.core as mx
ModuleNotFoundError: No module named 'mlx'
```

## 3. Methodology for a future Apple Silicon run

When this repo is audited on a supported Apple Silicon machine, the sequence below is the intended evidence path:

1. Install `mlx`, install the repo editable, and run `pytest tests -q`.
2. Run `python examples/run_basic.py`.
3. Only after correctness passes, run:

```bash
python benchmarks/run_all_benchmarks.py --full --output benchmarks/results/local_results.json --csv benchmarks/results/local_results.csv
python scripts/save_benchmark_report.py benchmarks/results/local_results.json --output docs/performance_report_local.md
python benchmarks/autotune.py --op all --quick --dtype float16 --write-cache
```

4. Publish only rows backed by:
   - the benchmark command,
   - the generated JSON/CSV artifacts, and
   - a same-machine correctness pass.

## 4. Results

### 4.1 Attention

not yet measured

### 4.2 Decode and KV-cache

not yet measured

### 4.3 Quantization

not yet measured

### 4.4 Transformer blocks

not yet measured

### 4.5 End-to-end generation from real weights

not yet measured

The README already states the current honest status: end-to-end generation from real model weights has not yet been benchmarked, and current results remain kernel- and block-level only once an Apple Silicon measurement run exists.

## 5. Raw artifacts

- `benchmarks/results/local_results.json`: not generated in this environment
- `benchmarks/results/local_results.csv`: not generated in this environment
- `benchmarks/results/.gitkeep`: present so future measured artifacts have a tracked directory
