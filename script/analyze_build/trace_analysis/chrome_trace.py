# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Chrome Trace Event Format export for Perfetto visualization.

Exports ninja build timeline data to Chrome Trace Event Format
for visualization in Perfetto UI within Jupyter notebooks.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd


class ChromeTraceExporter:
    """Export trace analysis data to Chrome Trace Event Format."""

    @staticmethod
    def categorize_target(target: str) -> str:
        """
        Categorize a build target based on file extension.

        Args:
            target: Build target name (e.g., "obj/foo.o")

        Returns:
            Category string for Chrome Trace format
        """
        ext = Path(target).suffix.lower()

        if ext in [".o", ".obj"]:
            return "compile"
        elif ext in [".a", ".lib"]:
            return "archive"
        elif ext in [".so", ".dll", ".dylib"]:
            return "link_shared"
        elif ext in [".exe", ".out"]:
            return "link_executable"
        elif "test" in target.lower():
            return "test"
        else:
            return "other"

    @staticmethod
    def export_ninja_timeline(
        builds_df: pd.DataFrame, process_id: int = 1, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Export ninja build timeline to Chrome Trace format.

        Creates trace events compatible with Perfetto UI for visualization
        in Jupyter notebooks or chrome://tracing.

        Args:
            builds_df: DataFrame with columns: target, start_ms, end_ms,
                      duration_ms, worker_id, cmd_hash
            process_id: Process ID for trace events (default: 1)
            include_metadata: Include trace metadata (default: True)

        Returns:
            Dictionary in Chrome Trace Event Format:
            {
                'traceEvents': [...],
                'displayTimeUnit': 'ms',
                'otherData': {...}
            }

        Example:
            >>> from trace_analysis import NinjaLogParser, ChromeTraceExporter
            >>> builds = NinjaLogParser.parse(Path('build/.ninja_log'))
            >>> builds_df = NinjaLogParser.to_dataframe(builds)
            >>> builds_df = NinjaLogParser.assign_workers(builds_df)
            >>> trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)
            >>> # Display in notebook or save to file
        """
        if len(builds_df) == 0:
            return {
                "traceEvents": [],
                "displayTimeUnit": "ms",
                "otherData": {"version": "1.0", "generator": "analyze_build"},
            }

        events = []

        for _, row in builds_df.iterrows():
            # Categorize based on file extension
            category = ChromeTraceExporter.categorize_target(row["target"])

            # Create Chrome Trace event
            event = {
                "name": row["target"],
                "cat": category,
                "ph": "X",  # Complete event (has duration)
                "ts": int(row["start_ms"] * 1000),  # Convert to microseconds
                "dur": int(row["duration_ms"] * 1000),  # Convert to microseconds
                "pid": process_id,
                "tid": int(row["worker_id"]),
                "args": {
                    "output": row["target"],
                    "duration_ms": int(row["duration_ms"]),
                    "cmd_hash": row["cmd_hash"],
                },
            }

            events.append(event)

        if include_metadata:
            return {
                "traceEvents": events,
                "displayTimeUnit": "ms",
                "otherData": {"version": "1.0", "generator": "analyze_build"},
            }
        else:
            # Simple format (just events array)
            return {"traceEvents": events}

    @staticmethod
    def export_to_file(trace_data: Dict[str, Any], output_path: str) -> None:
        """
        Export trace data to a JSON file.

        Args:
            trace_data: Chrome Trace format dictionary
            output_path: Path to output file

        Example:
            >>> ChromeTraceExporter.export_to_file(trace_data, 'trace.json')
        """
        import json

        with open(output_path, "w") as f:
            json.dump(trace_data, f, indent=2)
