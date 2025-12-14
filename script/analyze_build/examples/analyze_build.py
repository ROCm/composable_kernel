#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Analyze build trace files from Clang -ftime-trace.

Fast parallel analysis of all trace files in a directory. No caching, no complexity -
just straightforward I/O-limited parallel processing with in-memory aggregation.

Usage:
    python analyze_build.py [trace_directory]

Example:
    python analyze_build.py ../../build-trace
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trace_analysis import TraceFile, TraceParser, TraceTransformer

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def process_file(json_path: Path) -> tuple:
    """
    Process a single trace file and return DataFrames.

    Args:
        json_path: Path to JSON trace file

    Returns:
        Tuple of (file_name, events_df, templates_df, stats...)
    """
    trace_file = TraceFile.from_path(json_path)
    events = TraceParser.parse(trace_file)
    events_df = TraceTransformer.to_events_dataframe(events)
    templates_df = TraceTransformer.to_templates_dataframe(events)

    return (
        str(json_path.name),
        events_df,
        templates_df,
        len(events_df),
        int(events_df["dur"].sum()) if len(events_df) > 0 else 0,
        len(templates_df),
        int(templates_df["dur"].sum()) if len(templates_df) > 0 else 0,
    )


def main():
    """Main entry point."""
    # Get trace directory
    if len(sys.argv) > 1:
        trace_dir = Path(sys.argv[1])
    else:
        trace_dir = Path(__file__).parent.parent.parent / "build-trace"

    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        print(f"\nUsage: {sys.argv[0]} [trace_directory]")
        sys.exit(1)

    # Find all JSON files recursively
    json_files = list(trace_dir.rglob("*.json"))

    if not json_files:
        print(f"No trace files found in {trace_dir}")
        sys.exit(1)

    print(f"Found {len(json_files):,} trace files in {trace_dir}")

    # Process all files in parallel
    start_time = time.time()
    all_events = []
    all_templates = []
    file_stats = []

    workers = cpu_count()
    print(f"Processing with {workers} workers...\n")

    # Submit all files for parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f): f for f in json_files}

        # Collect results with progress bar
        if HAS_TQDM:
            pbar = tqdm(total=len(json_files), desc="Processing", unit="files")

        for future in as_completed(futures):
            (
                file_name,
                events_df,
                templates_df,
                event_count,
                event_dur,
                template_count,
                template_dur,
            ) = future.result()

            all_events.append(events_df)
            all_templates.append(templates_df)
            file_stats.append(
                {
                    "file_name": file_name,
                    "total_events": event_count,
                    "total_duration_us": event_dur,
                    "template_event_count": template_count,
                    "template_duration_us": template_dur,
                }
            )

            if HAS_TQDM:
                pbar.update(1)

        if HAS_TQDM:
            pbar.close()

    elapsed = time.time() - start_time
    print(
        f"\nParsing complete in {elapsed:.2f}s ({len(json_files) / elapsed:.1f} files/sec)"
    )

    # Combine all DataFrames
    print("Combining results...")
    combine_start = time.time()

    # Filter out empty DataFrames to avoid FutureWarning
    non_empty_events = [df for df in all_events if len(df) > 0]
    non_empty_templates = [df for df in all_templates if len(df) > 0]

    events_df = (
        pd.concat(non_empty_events, ignore_index=True)
        if non_empty_events
        else pd.DataFrame()
    )
    templates_df = (
        pd.concat(non_empty_templates, ignore_index=True)
        if non_empty_templates
        else pd.DataFrame()
    )
    file_stats_df = pd.DataFrame(file_stats)

    combine_time = time.time() - combine_start
    print(f"Combined in {combine_time:.2f}s")

    # Display statistics
    total_duration_us = events_df["dur"].sum()

    print(f"\n{'=' * 80}")
    print("ANALYSIS RESULTS")
    print(f"{'=' * 80}")
    print(f"Total files processed: {len(json_files):,}")
    print(f"Total events: {len(events_df):,}")
    print(f"Total build time: {total_duration_us / 1e6:.2f} seconds")
    print(f"{'=' * 80}\n")

    # Top event types by duration
    print("Top 10 Event Types by Total Duration:")
    print("-" * 80)
    event_totals = (
        events_df.groupby("name", observed=True)["dur"]
        .sum()
        .sort_values(ascending=False)
    )
    for event_type, duration in event_totals.head(10).items():
        print(f"{event_type:<50} {duration / 1e6:>12.2f}s")

    # Slowest files
    print("\nTop 10 Slowest Files:")
    print("-" * 80)
    slowest = file_stats_df.nlargest(10, "total_duration_us")
    for _, row in slowest.iterrows():
        print(f"{row['file_name']:<50} {row['total_duration_us'] / 1e6:>12.2f}s")

    # Template analysis
    if len(templates_df) > 0:
        total_template_time = templates_df["dur"].sum()

        print("\nTemplate Instantiation Summary:")
        print("-" * 80)
        print(f"Total template instantiations: {len(templates_df):,}")
        print(f"Total template time: {total_template_time / 1e6:.2f}s")
        print(
            f"Template time percentage: {(total_template_time / total_duration_us) * 100:.1f}%"
        )

        # Most common templates
        print("\nTop 10 Most Frequently Instantiated Templates:")
        print("-" * 80)
        template_counts = templates_df["template_detail"].value_counts()
        for template, count in template_counts.head(10).items():
            display = template if len(template) <= 60 else template[:57] + "..."
            print(f"{count:>8,} {display}")

        # Most expensive templates
        print("\nTop 10 Most Expensive Templates by Total Duration:")
        print("-" * 80)
        template_totals = templates_df.groupby("template_detail")["dur"].agg(
            ["sum", "count"]
        )
        template_totals["avg"] = template_totals["sum"] / template_totals["count"]
        template_totals = template_totals.sort_values("sum", ascending=False)

        for template, row in template_totals.head(10).iterrows():
            display = template if len(template) <= 50 else template[:47] + "..."
            print(
                f"{display:<50} {row['sum'] / 1e6:>10.2f}s (avg: {row['avg'] / 1e3:>8.2f}ms)"
            )

    print(f"\n{'=' * 80}\n")
    print(f"Total analysis time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
