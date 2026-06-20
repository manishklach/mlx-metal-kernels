from __future__ import annotations

import argparse
import json
from pathlib import Path


def _key(item):
    return (
        item.get("suite"),
        item.get("kernel"),
        item.get("backend"),
        item.get("dtype"),
        json.dumps(item.get("shape", {}), sort_keys=True),
    )


def _label(speedup):
    if speedup > 1.05:
        return "faster"
    if speedup < 0.95:
        return "slower"
    return "same"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_json")
    parser.add_argument("new_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    old_payload = json.loads(Path(args.old_json).read_text(encoding="utf-8"))
    new_payload = json.loads(Path(args.new_json).read_text(encoding="utf-8"))
    old_map = {_key(item): item for item in old_payload.get("results", []) if item.get("status") == "ok"}
    new_map = {_key(item): item for item in new_payload.get("results", []) if item.get("status") == "ok"}

    rows = [
        "# Benchmark Comparison",
        "",
        "| Suite | Kernel | Backend | Dtype | Shape | Old Mean (ms) | New Mean (ms) | Speedup | Regression % | Status |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(set(old_map) & set(new_map)):
        old_item = old_map[key]
        new_item = new_map[key]
        old_mean = old_item["timing"]["mean_ms"]
        new_mean = new_item["timing"]["mean_ms"]
        speedup = old_mean / new_mean if new_mean else 0.0
        regression_pct = ((new_mean - old_mean) / old_mean * 100.0) if old_mean else 0.0
        rows.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | `{key[4]}` | "
            f"{old_mean:.3f} | {new_mean:.3f} | {speedup:.3f} | {regression_pct:.2f}% | {_label(speedup)} |"
        )
    Path(args.output).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {args.output}")


if __name__ == "__main__":
    main()
