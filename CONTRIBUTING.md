# Contributing

## Quick start

```bash
pip install mlx pytest
pip install -e .
pytest tests -q
```

`mlx` is Apple Silicon specific in practice for this repo's intended workflow. If install or import fails, stop before benchmarking and record the environment blocker rather than publishing unverified numbers.

## How to add a new backend

1. **Reference path first** — implement a pure-MLX version of the op in the appropriate `ops/` module.
2. **Correctness test** — add a test in `tests/` that compares the reference output against the existing backends within dtype-appropriate tolerance.
3. **Optimized kernel** — write the Metal kernel in `kernels/` and wire it into the `ops/` Python interface.
4. **Test the new backend** — extend the correctness test to include the new backend.
5. **Benchmark** — add or extend a benchmark in `benchmarks/` and run it locally.
6. **Update docs** — update README and any relevant docs.

## Publishing results

- Do not add benchmark numbers to `README.md` until `pytest tests -q` passes on the same machine.
- Keep the exact benchmark command next to every published claim.
- Commit the generated JSON/CSV artifacts under `benchmarks/results/` only when they come from a real local run you intend to preserve.
- Update `docs/performance_report_local.md` with machine metadata, methodology, and links to the raw artifacts.

## Code style

- Python: follow existing patterns in the codebase. Use type annotations.
- Metal kernels: follow the conventions in `kernels/`.
- Tests: use `pytest` with descriptive test names. Every optimized backend should have a correctness gate.
- No performance claims without local benchmark data.
