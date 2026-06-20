# Benchmarking

This repo includes both individual benchmark CLIs and a unified benchmark suite for Apple Silicon testing.

## Quick vs full

- `--quick` runs a smaller representative benchmark sweep.
- `--full` runs larger shapes and broader coverage intended for a more complete machine report.

If neither flag is provided, the unified runner defaults to `--quick`.

## Run the unified suite

```bash
python benchmarks/run_all_benchmarks.py --quick
python benchmarks/run_all_benchmarks.py --full --output benchmarks/results/my_mac.json --csv benchmarks/results/my_mac.csv
python scripts/save_benchmark_report.py benchmarks/results/my_mac.json --output docs/performance_report_my_mac.md
```

## Compare two benchmark runs

```bash
python benchmarks/compare_results.py old.json new.json --output comparison.md
```

## What gets captured

The unified runner records:

- machine and platform information
- Python version
- macOS version if available
- MLX version if available
- Apple chip information when it can be collected
- benchmark configuration
- per-case timing data
- per-case errors or skips

## Submitting results

When sharing results, include:

- Mac model or chip
- macOS version
- MLX version
- exact command used
- whether experimental backends were enabled

## Notes

Benchmarks in this repo are not official performance claims unless they are reproduced on a specific Apple Silicon machine with the same configuration and command line.
