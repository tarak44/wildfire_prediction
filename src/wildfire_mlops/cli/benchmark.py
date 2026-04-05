from __future__ import annotations

import argparse
from pathlib import Path

from wildfire_mlops.training import (
    load_benchmark_record,
    render_markdown_table,
    save_benchmark_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wildfire model comparison table")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Benchmark entry in the format model_name=path/to/metrics.json",
    )
    parser.add_argument("--output", default=None, help="Optional path to save markdown output")
    args = parser.parse_args()

    records = []
    for entry in args.entry:
        if "=" not in entry:
            raise ValueError("Each --entry value must use model_name=metrics_path format")
        model_name, metrics_path = entry.split("=", maxsplit=1)
        records.append(load_benchmark_record(model_name, Path(metrics_path)))

    table = render_markdown_table(records)
    print(table)
    if args.output:
        save_benchmark_table(records, Path(args.output))


if __name__ == "__main__":
    main()
