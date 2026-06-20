from __future__ import annotations

import argparse
import json
from pathlib import Path


_SUITES = ["attention", "decode", "paged_decode", "norm", "rope", "activation", "layout", "quant", "quant_matvec_tiled", "decode_block", "quantized_decode_block", "threadgroup_attention_v2", "simdgroup_attention", "toy_transformer_decode"]


def _shape_text(shape):
    return ", ".join(f"{key}={value}" for key, value in sorted(shape.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    results = payload.get("results", [])
    system_info = payload.get("system_info", {})
    counts = {
        "ok": sum(1 for item in results if item.get("status") == "ok"),
        "error": sum(1 for item in results if item.get("status") == "error"),
        "skipped": sum(1 for item in results if item.get("status") == "skipped"),
    }
    lines = [
        "# Performance Report",
        "",
        f"- Timestamp: `{system_info.get('timestamp_utc')}`",
        f"- Platform: `{system_info.get('platform')}`",
        f"- Machine: `{system_info.get('machine')}`",
        f"- Processor: `{system_info.get('processor')}`",
        f"- Python: `{system_info.get('python_version')}`",
        f"- macOS: `{system_info.get('macos_version')}`",
        f"- MLX: `{system_info.get('mlx_version')}`",
        f"- Chip Info: `{system_info.get('chip_info')}`",
        "",
        "## Summary",
        "",
        f"- ok: {counts['ok']}",
        f"- error: {counts['error']}",
        f"- skipped: {counts['skipped']}",
        "",
    ]
    for suite in _SUITES:
        rows = [item for item in results if item.get("suite") == suite]
        if not rows:
            continue
        lines.extend(
            [
                f"## {suite.replace('_', ' ').title()}",
                "",
                "| Kernel | Backend | Dtype | Shape | Mean (ms) | Median (ms) | Min (ms) | Status |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in rows:
            timing = item.get("timing") or {}
            lines.append(
                f"| {item.get('kernel')} | {item.get('backend')} | {item.get('dtype')} | "
                f"`{_shape_text(item.get('shape', {}))}` | {timing.get('mean_ms', '')} | "
                f"{timing.get('median_ms', '')} | {timing.get('min_ms', '')} | {item.get('status')} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "Results are hardware-specific and should not be generalized across Apple Silicon machines without reproducing the same benchmark command, software stack, and backend configuration.",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote performance report to {args.output}")


if __name__ == "__main__":
    main()
