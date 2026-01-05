#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example: Visualizing Build Timeline in Perfetto UI

This example demonstrates how to:
1. Parse a ninja .ninja_log file
2. Export to Chrome Trace format
3. Display in Perfetto UI (for Jupyter notebooks)
4. Save to file for manual upload

Usage:
    python perfetto_visualization_example.py path/to/.ninja_log
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trace_analysis import NinjaLogParser, ChromeTraceExporter
from trace_analysis.perfetto_display import (
    print_trace_summary,
)


def main():
    """Main example function."""
    if len(sys.argv) < 2:
        print("Usage: python perfetto_visualization_example.py path/to/.ninja_log")
        sys.exit(1)

    ninja_log_path = Path(sys.argv[1])

    if not ninja_log_path.exists():
        print(f"Error: {ninja_log_path} not found")
        sys.exit(1)

    print(f"Parsing {ninja_log_path}...")

    # Step 1: Parse ninja log
    builds = NinjaLogParser.parse(ninja_log_path)
    builds_df = NinjaLogParser.to_dataframe(builds)

    print(f"Found {len(builds_df):,} build targets")

    # Step 2: Assign workers (for parallelism visualization)
    builds_df = NinjaLogParser.assign_workers(builds_df)

    print(f"Assigned {builds_df['worker_id'].max() + 1} workers")

    # Step 3: Export to Chrome Trace format
    trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)

    print(f"\nGenerated {len(trace_data['traceEvents']):,} trace events")

    # Step 4: Print summary
    print_trace_summary(trace_data)

    # Step 5: Save to file
    output_path = ninja_log_path.parent / "build_trace.json"
    ChromeTraceExporter.export_to_file(trace_data, str(output_path))

    print(f"\n✓ Trace saved to: {output_path}")
    print("\nTo view in Perfetto UI:")
    print("  1. Go to https://ui.perfetto.dev")
    print("  2. Click 'Open trace file'")
    print(f"  3. Select: {output_path}")
    print("\nOr drag and drop the file directly into Perfetto UI")

    # For Jupyter notebook usage, you would use:
    # display_perfetto(trace_data)
    # or for large traces:
    # save_and_link(trace_data, str(output_path))


if __name__ == "__main__":
    main()
