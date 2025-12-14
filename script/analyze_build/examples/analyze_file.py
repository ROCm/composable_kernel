#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Analyze a single -ftime-trace JSON file.

Simple script to analyze one trace file and display summary statistics.

Usage:
    python analyze_file.py <path_to_trace_file.json>

Example:
    python analyze_file.py ../../build-trace/some_file.json
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trace_analysis import TraceFile, TraceParser, TraceTransformer


def analyze_trace_file(trace_path: Path):
    """Analyze a single trace file and print summary statistics."""

    print(f"\n{'=' * 80}")
    print(f"Analyzing: {trace_path.name}")
    print(f"{'=' * 80}\n")

    # Parse the file
    trace_file = TraceFile.from_path(trace_path)
    events = TraceParser.parse(trace_file)
    events_df = TraceTransformer.to_events_dataframe(events)
    templates_df = TraceTransformer.to_templates_dataframe(events)

    # Basic statistics
    total_duration_us = events_df["dur"].sum()

    print(f"Total events: {len(events_df):,}")
    print(f"Total duration: {total_duration_us / 1e6:.2f}s")
    print(f"File size: {trace_file.size_bytes / 1024:.1f} KB\n")

    # Top event types by duration
    print("Top 10 Event Types by Total Duration:")
    print(f"{'Event Type':<40} {'Count':>10} {'Total (s)':>12} {'Avg (ms)':>12}")
    print("-" * 80)

    event_stats = events_df.groupby("name", observed=True)["dur"].agg(
        ["sum", "count", "mean"]
    )
    event_stats = event_stats.sort_values("sum", ascending=False).head(10)

    for name, row in event_stats.iterrows():
        total_sec = row["sum"] / 1e6
        avg_ms = row["mean"] / 1e3
        print(f"{name:<40} {int(row['count']):>10,} {total_sec:>12.2f} {avg_ms:>12.2f}")

    # Template analysis
    if len(templates_df) > 0:
        total_template_time = templates_df["dur"].sum()

        print("\n\nTemplate Instantiation Summary:")
        print("-" * 80)
        print(f"Total template instantiations: {len(templates_df):,}")
        print(f"Total template time: {total_template_time / 1e6:.2f}s")
        print(
            f"Template time percentage: {(total_template_time / total_duration_us) * 100:.1f}%"
        )

        # Top 10 slowest individual template instantiations
        print("\n\nTop 10 Slowest Template Instantiations:")
        print(f"{'Duration (s)':>12} {'Template'}")
        print("-" * 80)

        slowest = templates_df.nlargest(10, "dur")
        for _, row in slowest.iterrows():
            duration_sec = row["dur"] / 1e6
            template = row["template_detail"]
            if len(template) > 65:
                template = template[:62] + "..."
            print(f"{duration_sec:>12.2f} {template}")

        # Most frequently instantiated templates
        print("\n\nTop 10 Most Frequently Instantiated Templates:")
        print(f"{'Count':>10} {'Template'}")
        print("-" * 80)

        template_counts = templates_df["template_detail"].value_counts().head(10)
        for template, count in template_counts.items():
            if len(template) > 65:
                template = template[:62] + "..."
            print(f"{count:>10,} {template}")

        # Most expensive templates by total time
        print("\n\nTop 10 Most Expensive Templates by Total Duration:")
        print(f"{'Total (s)':>12} {'Count':>10} {'Avg (ms)':>12} {'Template'}")
        print("-" * 80)

        template_totals = templates_df.groupby("template_detail")["dur"].agg(
            ["sum", "count", "mean"]
        )
        template_totals = template_totals.sort_values("sum", ascending=False).head(10)

        for template, row in template_totals.iterrows():
            total_sec = row["sum"] / 1e6
            avg_ms = row["mean"] / 1e3
            display = template if len(template) <= 40 else template[:37] + "..."
            print(
                f"{total_sec:>12.2f} {int(row['count']):>10,} {avg_ms:>12.2f} {display}"
            )

    print(f"\n{'=' * 80}\n")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    trace_path = Path(sys.argv[1])

    if not trace_path.exists():
        print(f"Error: File not found: {trace_path}")
        sys.exit(1)

    if not trace_path.suffix == ".json":
        print(f"Warning: File does not have .json extension: {trace_path}")

    try:
        analyze_trace_file(trace_path)
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
